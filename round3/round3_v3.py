
import json
import statistics
from abc import abstractmethod
from collections import deque
from typing import Any, TypeAlias, List, Dict, Tuple

# We still import the same data model classes, with the same names used in your environment.
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

############################################
# Type Definitions
############################################
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


############################################
# Reordered & Renamed Logger
############################################
class ActivityRecorder:
    def __init__(self) -> None:
        self.cached_logs = ""
        self.maximum_log_size = 3750

    def output(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        self.cached_logs += sep.join(map(str, args)) + end

    def finalize(self, current_state: TradingState, order_bundle: Dict[Symbol, List[Order]], conv_count: int, data_state: str) -> None:
        prelim = len(
            self.condense_to_json(
                [
                    self.state_brief(current_state, ""),
                    self.minify_orders(order_bundle),
                    conv_count,
                    "",
                    "",
                ]
            )
        )
        slice_len = (self.maximum_log_size - prelim) // 3

        print(
            self.condense_to_json(
                [
                    self.state_brief(current_state, self.trim(current_state.traderData, slice_len)),
                    self.minify_orders(order_bundle),
                    conv_count,
                    self.trim(data_state, slice_len),
                    self.trim(self.cached_logs, slice_len),
                ]
            )
        )

        self.cached_logs = ""

    def trim(self, data: str, cutoff: int) -> str:
        if len(data) <= cutoff:
            return data
        return data[: cutoff - 3] + "..."

    def state_brief(self, current_state: TradingState, textual_data: str) -> List[Any]:
        return [
            current_state.timestamp,
            textual_data,
            self.mini_listings(current_state.listings),
            self.mini_depths(current_state.order_depths),
            self.mini_trades(current_state.own_trades),
            self.mini_trades(current_state.market_trades),
            current_state.position,
            self.mini_observations(current_state.observations),
        ]

    def mini_listings(self, all_listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed_info = []
        for lst in all_listings.values():
            compressed_info.append([lst.symbol, lst.product, lst.denomination])
        return compressed_info

    def mini_depths(self, depth_map: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        shrunk = {}
        for smb, dp in depth_map.items():
            shrunk[smb] = [dp.buy_orders, dp.sell_orders]
        return shrunk

    def mini_trades(self, trade_map: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        comp = []
        for trades_arr in trade_map.values():
            for t in trades_arr:
                comp.append([
                    t.symbol,
                    t.price,
                    t.quantity,
                    t.buyer,
                    t.seller,
                    t.timestamp
                ])
        return comp

    def mini_observations(self, obs: Observation) -> List[Any]:
        conv_data = {}
        for prod, obv in obs.conversionObservations.items():
            conv_data[prod] = [
                obv.bidPrice,
                obv.askPrice,
                obv.transportFees,
                obv.exportTariff,
                obv.importTariff,
                obv.sugarPrice,
                obv.sunlightIndex,
            ]
        return [obs.plainValueObservations, conv_data]

    def minify_orders(self, request_map: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        mini = []
        for arr in request_map.values():
            for od in arr:
                mini.append([od.symbol, od.price, od.quantity])
        return mini

    def condense_to_json(self, val: Any) -> str:
        return json.dumps(val, cls=ProsperityEncoder, separators=(",", ":"))


# Global logger instance
activity_logger = ActivityRecorder()


############################################
# Base Approach
############################################
class ApproachBase:
    def __init__(self, ticker: str, capacity: int) -> None:
        self.ticker = ticker
        self.capacity = capacity

    @abstractmethod
    def behave(self, snapshot: TradingState) -> None:
        pass

    def process(self, snapshot: TradingState) -> Tuple[List[Order], int]:
        self.current_orders: List[Order] = []
        self.num_conversions = 0
        self.behave(snapshot)
        return self.current_orders, self.num_conversions

    def buy(self, price_val: int, qty: int) -> None:
        self.current_orders.append(Order(self.ticker, price_val, qty))

    def sell(self, price_val: int, qty: int) -> None:
        self.current_orders.append(Order(self.ticker, price_val, -qty))

    def convert(self, conv_amt: int) -> None:
        self.num_conversions += conv_amt

    def preserve(self) -> JSON:
        return None

    def restore(self, data: JSON) -> None:
        pass


############################################
# MarketBalancer (dynamic market-maker)
############################################
class MarketBalancer(ApproachBase):
    def __init__(self, ticker: Symbol, capacity: int) -> None:
        super().__init__(ticker, capacity)
        self.flags_window = deque()
        self.window_size = 10

    @abstractmethod
    def get_valuation(self, snapshot: TradingState) -> int:
        pass

    def behave(self, snapshot: TradingState) -> None:
        # Evaluate fair price
        valuation = self.get_valuation(snapshot)

        if self.ticker not in snapshot.order_depths:
            return

        od = snapshot.order_depths[self.ticker]
        sorted_buys = sorted(od.buy_orders.items(), reverse=True)
        sorted_sells = sorted(od.sell_orders.items())

        pos = snapshot.position.get(self.ticker, 0)
        remain_buy = self.capacity - pos
        remain_sell = self.capacity + pos

        # Keep track of how often we are pinned at limit
        self.flags_window.append(abs(pos) == self.capacity)
        if len(self.flags_window) > self.window_size:
            self.flags_window.popleft()

        half_filled_count = sum(self.flags_window)
        soft_liquid = (len(self.flags_window) == self.window_size and half_filled_count >= self.window_size / 2 and self.flags_window[-1])
        hard_liquid = (len(self.flags_window) == self.window_size and all(self.flags_window))

        # Adjust buy/sell range to reduce overfitting:
        # Instead of using exact thresholds, we do a small linear band around the valuation
        buy_ceiling = int(valuation * 0.999) if pos > self.capacity * 0.5 else valuation
        sell_floor = int(valuation * 1.001) if pos < self.capacity * -0.5 else valuation

        # Attempt to buy from the ask side
        for p, volume in sorted_sells:
            if remain_buy > 0 and p <= buy_ceiling:
                can_take = min(remain_buy, -volume)
                self.buy(p, can_take)
                remain_buy -= can_take

        # “Hard liquidation” if pinned at capacity
        if remain_buy > 0 and hard_liquid:
            self.buy(buy_ceiling, remain_buy // 2)
            remain_buy = remain_buy // 2

        # “Soft liquidation” if we are partial pinned
        if remain_buy > 0 and soft_liquid:
            self.buy(buy_ceiling - 2, remain_buy // 2)
            remain_buy = remain_buy // 2

        # If still leftover, place a passive buy
        if remain_buy > 0:
            if sorted_buys:
                top_bid = max(sorted_buys, key=lambda x: x[1])[0]
            else:
                top_bid = buy_ceiling
            final_buy = min(buy_ceiling, top_bid + 1)
            self.buy(final_buy, remain_buy)

        # Attempt to sell into the bid side
        for p, volume in sorted_buys:
            if remain_sell > 0 and p >= sell_floor:
                can_give = min(remain_sell, volume)
                self.sell(p, can_give)
                remain_sell -= can_give

        if remain_sell > 0 and hard_liquid:
            self.sell(sell_floor, remain_sell // 2)
            remain_sell = remain_sell // 2

        if remain_sell > 0 and soft_liquid:
            self.sell(sell_floor + 2, remain_sell // 2)
            remain_sell = remain_sell // 2

        if remain_sell > 0:
            if sorted_sells:
                bot_ask = min(sorted_sells, key=lambda x: x[1])[0]
            else:
                bot_ask = sell_floor
            final_ask = max(sell_floor, bot_ask - 1)
            self.sell(final_ask, remain_sell)

    def preserve(self) -> JSON:
        # Save the “flags_window” to preserve pinned-limit times
        return list(self.flags_window)

    def restore(self, data: JSON) -> None:
        if data is not None:
            self.flags_window = deque(data, maxlen=self.window_size)


############################################
# Example dynamic MarketMaker strategies
############################################
class AmethystsFlow(MarketBalancer):
    def get_valuation(self, snapshot: TradingState) -> int:
        # A simple “flat” fair value for example
        return 10_000

class StarfruitFlow(MarketBalancer):
    def get_valuation(self, snapshot: TradingState) -> int:
        if self.ticker not in snapshot.order_depths:
            return 5000  # Fallback

        depth_info = snapshot.order_depths[self.ticker]
        if not depth_info.buy_orders or not depth_info.sell_orders:
            return 5000

        b_arr = sorted(depth_info.buy_orders.items(), reverse=True)
        s_arr = sorted(depth_info.sell_orders.items())
        best_b = b_arr[0][0] if b_arr else 0
        best_s = s_arr[0][0] if s_arr else 0
        return int((best_b + best_s) / 2)


############################################
# OrchidsStrategy => FlowerArrangementMethod
############################################
class FlowerArrangementMethod(ApproachBase):
    def behave(self, snapshot: TradingState) -> None:
        pos = snapshot.position.get(self.ticker, 0)
        self.convert(-pos)

        ob = snapshot.observations.conversionObservations.get(self.ticker, None)
        if ob is None:
            return

        # dynamic approach: e.g., half-liner around the transportation
        baseline_cost = ob.askPrice + ob.transportFees + ob.importTariff
        # Sell near that cost
        self.sell(max(int(ob.bidPrice - 0.5), int(baseline_cost + 1)), self.capacity)


############################################
# GiftBasketStrategy => HamperDealApproach
# -> Add rolling average logic for the “offset”
#    so we are less reliant on a fixed threshold
############################################
class HamperDealApproach(ApproachBase):
    def __init__(self, ticker: str, capacity: int) -> None:
        super().__init__(ticker, capacity)
        # rolling history of “offset” for dynamic thresholds
        self.offset_history = deque(maxlen=50)

    def behave(self, snapshot: TradingState) -> None:
        needed_symbols = ["JAMS", "CROISSANTS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        if any(symb not in snapshot.order_depths for symb in needed_symbols):
            return

        choco_val = self._fetch_mid(snapshot, "JAMS")
        straw_val = self._fetch_mid(snapshot, "CROISSANTS")
        roses_val = self._fetch_mid(snapshot, "DJEMBES")
        gift1 = self._fetch_mid(snapshot, "PICNIC_BASKET1")
        gift2 = self._fetch_mid(snapshot, "PICNIC_BASKET2")

        # Basic offset logic
        if self.ticker == "PICNIC_BASKET2":
            offset = gift2 - 4 * straw_val - 2 * choco_val
        else:
            offset = gift1 - 4 * choco_val - 6 * straw_val - roses_val

        # store offset in rolling history
        self.offset_history.append(offset)

        # dynamic baseline from rolling average
        # if offset_history short, fallback to offset
        avg_offset = statistics.mean(self.offset_history) if len(self.offset_history) > 1 else offset
        # standard deviation
        stdev_offset = statistics.pstdev(self.offset_history) if len(self.offset_history) > 1 else 1

        # also keep your original thresholds for a fallback
        base_long, base_short = {
            "JAMS":           (230, 355),
            "CROISSANTS":     (195, 485),
            "DJEMBES":        (325, 370),
            "PICNIC_BASKET1": (290, 355),
            "PICNIC_BASKET2": (50,  100),
        }[self.ticker]

        # We do a “dynamic band” around the rolling average
        # e.g. if offset < avg_offset - stdev => buy, if offset > avg_offset + stdev => sell
        # but also check the original thresholds as a “second line”
        lower_band = avg_offset - stdev_offset
        upper_band = avg_offset + stdev_offset

        # We require offset < BOTH lower_band AND base_long to buy
        # or offset > BOTH upper_band AND base_short to sell
        if offset < lower_band and offset < base_long:
            self._do_buy(snapshot)
        elif offset > upper_band and offset > base_short:
            self._do_sell(snapshot)

    def _fetch_mid(self, snapshot: TradingState, symb: str) -> float:
        od = snapshot.order_depths.get(symb)
        if not od or not od.buy_orders or not od.sell_orders:
            return 0.0
        sorted_buys = sorted(od.buy_orders.items(), reverse=True)
        sorted_sells = sorted(od.sell_orders.items())
        best_bid = sorted_buys[0][0]
        best_ask = sorted_sells[0][0]
        return (best_bid + best_ask) / 2

    def _do_buy(self, snapshot: TradingState) -> None:
        depth = snapshot.order_depths[self.ticker]
        if not depth or not depth.sell_orders:
            return
        # buy at best ask
        best_ask = min(depth.sell_orders.keys())
        pos = snapshot.position.get(self.ticker, 0)
        to_buy = self.capacity - pos
        if to_buy > 0:
            self.buy(best_ask, to_buy)

    def _do_sell(self, snapshot: TradingState) -> None:
        depth = snapshot.order_depths[self.ticker]
        if not depth or not depth.buy_orders:
            return
        # sell at best bid
        best_bid = max(depth.buy_orders.keys())
        pos = snapshot.position.get(self.ticker, 0)
        to_sell = self.capacity + pos
        if to_sell > 0:
            self.sell(best_bid, to_sell)

    def preserve(self) -> JSON:
        return list(self.offset_history)

    def restore(self, data: JSON) -> None:
        if data is not None:
            self.offset_history = deque(data, maxlen=50)


############################################
# IgneousStoneCouponManager
# (VolcanicRockVoucherTrader renamed)
############################################
class IgneousStoneCouponManager:
    def run(self, snapshot: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        out_orders: Dict[str, List[Order]] = {}
        memo_state = ""
        total_convs = 0

        # figure out base quote
        base_rock_quote = None
        if "VOLCANIC_ROCK" in snapshot.order_depths:
            rocky_orders = snapshot.order_depths["VOLCANIC_ROCK"]
            if rocky_orders.buy_orders and rocky_orders.sell_orders:
                highest_bid = max(rocky_orders.buy_orders.keys())
                lowest_ask = min(rocky_orders.sell_orders.keys())
                base_rock_quote = (highest_bid + lowest_ask) / 2.0
            elif rocky_orders.buy_orders:
                base_rock_quote = max(rocky_orders.buy_orders.keys())
            elif rocky_orders.sell_orders:
                base_rock_quote = min(rocky_orders.sell_orders.keys())

        if base_rock_quote is None:
            return out_orders, total_convs, memo_state

        coupon_strikes = {
            "VOLCANIC_ROCK": 10312,
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }

        for cpn, strk in coupon_strikes.items():
            if cpn not in snapshot.order_depths:
                continue

            cpn_depth = snapshot.order_depths[cpn]
            intrinsic_eval = base_rock_quote - strk
            if intrinsic_eval < 0:
                intrinsic_eval = 0

            local_orders: List[Order] = []
            # buy logic
            if cpn_depth.sell_orders:
                lowest_offer = min(cpn_depth.sell_orders.keys())
                if lowest_offer < intrinsic_eval:
                    liquidity = abs(cpn_depth.sell_orders[lowest_offer])
                    local_orders.append(Order(cpn, lowest_offer, liquidity))
            # sell logic
            if cpn_depth.buy_orders:
                highest_deal = max(cpn_depth.buy_orders.keys())
                if highest_deal > intrinsic_eval:
                    liquidity = cpn_depth.buy_orders[highest_deal]
                    local_orders.append(Order(cpn, highest_deal, -liquidity))

            if local_orders:
                out_orders[cpn] = local_orders

        return out_orders, total_convs, memo_state


############################################
# Trader
# (Replaces OverarchingHandler with the name Trader,
#  so it can be imported as `from trader import Trader`)
############################################
class Trader:
    def __init__(self) -> None:
        caps = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "ORCHIDS": 100,
            "JAMS": 350,
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        # Connect each symbol to its approach
        self.strategies: Dict[Symbol, ApproachBase] = {
            sym: cls(sym, caps[sym])
            for sym, cls in {
                "RAINFOREST_RESIN": AmethystsFlow,
                "KELP": StarfruitFlow,
                "ORCHIDS": FlowerArrangementMethod,
                "JAMS": HamperDealApproach,
                "CROISSANTS": HamperDealApproach,
                "DJEMBES": HamperDealApproach,
                "PICNIC_BASKET1": HamperDealApproach,
                "PICNIC_BASKET2": HamperDealApproach,
            }.items()
        }
        # IgneousStoneCouponManager for volcanic rock & vouchers
        self.coupon_machine = IgneousStoneCouponManager()

    def run(self, snap: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        final_request: Dict[Symbol, List[Order]] = {}
        conv_sum = 0

        # Reconstruct old data if present
        old_persist = {}
        if snap.traderData != "":
            old_persist = json.loads(snap.traderData)

        new_persist = {}

        # 1) normal strategies
        for sm, st in self.strategies.items():
            if sm in old_persist:
                st.restore(old_persist[sm])
            if sm in snap.order_depths:
                st_orders, st_convs = st.process(snap)
                final_request[sm] = st_orders
                conv_sum += st_convs
            new_persist[sm] = st.preserve()

        # 2) run the coupon manager
        coupon_out, coupon_convs, _ = self.coupon_machine.run(snap)
        for pr_s, pr_ordz in coupon_out.items():
            if pr_s in final_request:
                final_request[pr_s].extend(pr_ordz)
            else:
                final_request[pr_s] = pr_ordz
        conv_sum += coupon_convs

        # final persistent data
        updated_data = json.dumps(new_persist, separators=(",", ":"))
        activity_logger.finalize(snap, final_request, conv_sum, updated_data)
        return final_request, conv_sum, updated_data