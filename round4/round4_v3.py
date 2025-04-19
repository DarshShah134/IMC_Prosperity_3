import json
import math
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

    def finalize(
        self,
        current_state: TradingState,
        order_bundle: Dict[Symbol, List[Order]],
        conv_count: int,
        data_state: str,
    ) -> None:
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
                    self.state_brief(
                        current_state, self.trim(current_state.traderData, slice_len)
                    ),
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
                    t.timestamp,
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
        soft_liquid = (
            len(self.flags_window) == self.window_size
            and half_filled_count >= self.window_size / 2
            and self.flags_window[-1]
        )
        hard_liquid = len(self.flags_window) == self.window_size and all(
            self.flags_window
        )

        # Adjust buy/sell range to reduce overfitting:
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
        return list(self.flags_window)

    def restore(self, data: JSON) -> None:
        if data is not None:
            self.flags_window = deque(data, maxlen=self.window_size)


############################################
# Example dynamic MarketMaker strategies
############################################
class AmethystsFlow(MarketBalancer):
    def get_valuation(self, snapshot: TradingState) -> int:
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

        baseline_cost = ob.askPrice + ob.transportFees + ob.importTariff
        self.sell(max(int(ob.bidPrice - 0.5), int(baseline_cost + 1)), self.capacity)


############################################
# GiftBasketStrategy => HamperDealApproach
############################################
class HamperDealApproach(ApproachBase):
    def __init__(self, ticker: str, capacity: int) -> None:
        super().__init__(ticker, capacity)
        self.offset_history = deque(maxlen=50)

    def behave(self, snapshot: TradingState) -> None:
        needed_symbols = [
            "JAMS",
            "CROISSANTS",
            "DJEMBES",
            "PICNIC_BASKET1",
            "PICNIC_BASKET2",
        ]
        if any(symb not in snapshot.order_depths for symb in needed_symbols):
            return

        choco_val = self._fetch_mid(snapshot, "JAMS")
        straw_val = self._fetch_mid(snapshot, "CROISSANTS")
        roses_val = self._fetch_mid(snapshot, "DJEMBES")
        gift1 = self._fetch_mid(snapshot, "PICNIC_BASKET1")
        gift2 = self._fetch_mid(snapshot, "PICNIC_BASKET2")

        if self.ticker == "PICNIC_BASKET2":
            offset = gift2 - 4 * straw_val - 2 * choco_val
        else:
            offset = gift1 - 4 * choco_val - 6 * straw_val - roses_val

        self.offset_history.append(offset)

        avg_offset = (
            statistics.mean(self.offset_history) if len(self.offset_history) > 1 else offset
        )
        stdev_offset = (
            statistics.pstdev(self.offset_history) if len(self.offset_history) > 1 else 1
        )

        base_long, base_short = {
            "JAMS": (230, 355),
            "CROISSANTS": (195, 485),
            "DJEMBES": (325, 370),
            "PICNIC_BASKET1": (290, 355),
            "PICNIC_BASKET2": (50, 100),
        }[self.ticker]

        lower_band = avg_offset - stdev_offset
        upper_band = avg_offset + stdev_offset

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
        best_ask = min(depth.sell_orders.keys())
        pos = snapshot.position.get(self.ticker, 0)
        to_buy = self.capacity - pos
        if to_buy > 0:
            self.buy(best_ask, to_buy)

    def _do_sell(self, snapshot: TradingState) -> None:
        depth = snapshot.order_depths[self.ticker]
        if not depth or not depth.buy_orders:
            return
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
# IgneousStoneCouponManager (Volcanic Logic)
############################################
class IgneousStoneCouponManager:
    """Volcanic‑Rock / Voucher market‑maker (unchanged)"""

    POSITION_LIMITS = {
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
    }

    COUPON_STRIKES = {
        "VOLCANIC_ROCK": 10312,
        "VOLCANIC_ROCK_VOUCHER_9500": 9500,
        "VOLCANIC_ROCK_VOUCHER_9750": 9750,
        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    }

    def run(self, snapshot: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        out_orders: Dict[str, List[Order]] = {}
        memo_state = ""
        total_convs = 0

        base_rock_quote = None
        rock_depth = snapshot.order_depths.get("VOLCANIC_ROCK")
        if rock_depth:
            if rock_depth.buy_orders and rock_depth.sell_orders:
                base_rock_quote = (
                    max(rock_depth.buy_orders) + min(rock_depth.sell_orders)
                ) / 2.0
            elif rock_depth.buy_orders:
                base_rock_quote = max(rock_depth.buy_orders)
            elif rock_depth.sell_orders:
                base_rock_quote = min(rock_depth.sell_orders)

        if base_rock_quote is None:
            return out_orders, total_convs, memo_state

        for cpn, strike in self.COUPON_STRIKES.items():
            depth = snapshot.order_depths.get(cpn)
            if not depth:
                continue

            intrinsic = max(base_rock_quote - strike, 0)
            pos_now = snapshot.position.get(cpn, 0)
            limit = self.POSITION_LIMITS[cpn]

            local: List[Order] = []

            if depth.sell_orders:
                best_ask = min(depth.sell_orders)
                if best_ask < intrinsic and pos_now < limit:
                    avail = min(-depth.sell_orders[best_ask], limit - pos_now)
                    if avail > 0:
                        local.append(Order(cpn, best_ask, avail))
                        pos_now += avail

            if depth.buy_orders:
                best_bid = max(depth.buy_orders)
                if best_bid > intrinsic and pos_now > -limit:
                    avail = min(depth.buy_orders[best_bid], limit + pos_now)
                    if avail > 0:
                        local.append(Order(cpn, best_bid, -avail))
                        pos_now -= avail

            if local:
                out_orders[cpn] = local

        return out_orders, total_convs, memo_state


############################################
# Replacement strategies for SQUID_INK, KELP, RAINFOREST_RESIN
############################################
############################################
# Replacement strategies for SQUID_INK, KELP, RAINFOREST_RESIN
############################################
class RainforestResinLogic(ApproachBase):
    """Exact Rainforest‑Resin logic from your original main code"""
    FV = 10000

    def _clear_position_order(
        self,
        fv: float,
        width: int,
        orders: List[Order],
        od: OrderDepth,
        position: int,
        buy_vol: int,
        sell_vol: int,
    ) -> Tuple[int, int]:
        pos_after = position + buy_vol - sell_vol
        fair_bid = round(fv - width)
        fair_ask = round(fv + width)

        buy_qty = self.capacity - (position + buy_vol)
        sell_qty = self.capacity + (position - sell_vol)

        # Liquidate longs
        if pos_after > 0:
            clear_qty = sum(
                vol for price, vol in od.buy_orders.items() if price >= fair_ask
            )
            clear_qty = min(clear_qty, pos_after)
            send_qty = min(sell_qty, clear_qty)
            if send_qty > 0:
                orders.append(Order(self.ticker, fair_ask, -abs(send_qty)))
                sell_vol += abs(send_qty)

        # Cover shorts
        if pos_after < 0:
            clear_qty = sum(
                abs(vol) for price, vol in od.sell_orders.items() if price <= fair_bid
            )
            clear_qty = min(clear_qty, abs(pos_after))
            send_qty = min(buy_qty, clear_qty)
            if send_qty > 0:
                orders.append(Order(self.ticker, fair_bid, abs(send_qty)))
                buy_vol += abs(send_qty)

        return buy_vol, sell_vol

    def behave(self, snapshot: TradingState) -> None:
        od = snapshot.order_depths.get(self.ticker)
        if not od:
            return

        orders: List[Order] = []
        pos = snapshot.position.get(self.ticker, 0)
        fv = self.FV

        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None

        buy_vol = 0
        sell_vol = 0

        if best_ask is not None and best_ask < fv:
            best_ask_amount = -od.sell_orders[best_ask]
            qty = min(best_ask_amount, self.capacity - pos)
            if qty > 0:
                self.buy(best_ask, qty)
                buy_vol += qty

        if best_bid is not None and best_bid > fv:
            best_bid_amount = od.buy_orders[best_bid]
            qty = min(best_bid_amount, self.capacity + pos)
            if qty > 0:
                self.sell(best_bid, qty)
                sell_vol += qty

        buy_vol, sell_vol = self._clear_position_order(
            fv,
            1,
            orders=self.current_orders,  # accumulate directly
            od=od,
            position=pos,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
        )

        # Passive edges
        baaf = min([p for p in od.sell_orders if p > fv + 1], default=fv + 2)
        bbbf = max([p for p in od.buy_orders if p < fv - 1], default=fv - 2)

        buy_qty = self.capacity - (pos + buy_vol)
        if buy_qty > 0:
            self.buy(bbbf + 1, buy_qty)

        sell_qty = self.capacity + (pos - sell_vol)
        if sell_qty > 0:
            self.sell(baaf - 1, sell_qty)


class KelpLogic(ApproachBase):
    def __init__(self, ticker: str, capacity: int) -> None:
        super().__init__(ticker, capacity)
        self.kelp_prices: List[float] = []
        self.kelp_vwap: List[Dict[str, float]] = []

    def _clear_position_order_kelp(
        self,
        od: OrderDepth,
        pos: int,
        limit: int,
        fv: float,
        buy_fill: int,
        sell_fill: int,
    ) -> Tuple[int, int]:
        pos_after = pos + buy_fill - sell_fill
        fair_bid = math.floor(fv)
        fair_ask = math.ceil(fv)

        buy_qty = limit - (pos + buy_fill)
        sell_qty = limit + (pos - sell_fill)

        if pos_after > 0 and fair_ask in od.buy_orders:
            clear_qty = min(od.buy_orders[fair_ask], pos_after, sell_qty)
            if clear_qty > 0:
                self.sell(fair_ask, clear_qty)
                sell_fill += clear_qty
        if pos_after < 0 and fair_bid in od.sell_orders:
            clear_qty = min(-od.sell_orders[fair_bid], -pos_after, buy_qty)
            if clear_qty > 0:
                self.buy(fair_bid, clear_qty)
                buy_fill += clear_qty
        return buy_fill, sell_fill

    def behave(self, snapshot: TradingState) -> None:
        od = snapshot.order_depths.get(self.ticker)
        if not od or not od.buy_orders or not od.sell_orders:
            return

        pos = snapshot.position.get(self.ticker, 0)
        limit = self.capacity

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())

        filt_ask = [p for p in od.sell_orders if abs(od.sell_orders[p]) >= 15]
        filt_bid = [p for p in od.buy_orders if abs(od.buy_orders[p]) >= 15]
        mm_ask = min(filt_ask) if filt_ask else best_ask
        mm_bid = max(filt_bid) if filt_bid else best_bid
        mmmid = (mm_ask + mm_bid) / 2

        self.kelp_prices.append(mmmid)
        if len(self.kelp_prices) > 10:
            self.kelp_prices.pop(0)

        volume = -od.sell_orders[best_ask] + od.buy_orders[best_bid]
        vwap = (
            (best_bid * (-od.sell_orders[best_ask]) + best_ask * od.buy_orders[best_bid]) / volume
            if volume != 0
            else mmmid
        )
        self.kelp_vwap.append({"vol": volume, "vwap": vwap})
        if len(self.kelp_vwap) > 10:
            self.kelp_vwap.pop(0)

        fv = mmmid
        buy_fill = 0
        sell_fill = 0

        take_width = 1
        if best_ask <= fv - take_width:
            ask_amt = -od.sell_orders[best_ask]
            qty = min(ask_amt, limit - pos, 20)
            if qty > 0:
                self.buy(best_ask, qty)
                buy_fill += qty

        if best_bid >= fv + take_width:
            bid_amt = od.buy_orders[best_bid]
            qty = min(bid_amt, limit + pos, 20)
            if qty > 0:
                self.sell(best_bid, qty)
                sell_fill += qty

        buy_fill, sell_fill = self._clear_position_order_kelp(
            od, pos, limit, fv, buy_fill, sell_fill
        )

        baaf = min([p for p in od.sell_orders if p > fv + 1], default=fv + 2)
        bbbf = max([p for p in od.buy_orders if p < fv - 1], default=fv - 2)

        buy_qty = limit - (pos + buy_fill)
        sell_qty = limit + (pos - sell_fill)

        if buy_qty > 0:
            self.buy(int(bbbf + 1), buy_qty)
        if sell_qty > 0:
            self.sell(int(baaf - 1), sell_qty)


class SquidInkLogic(ApproachBase):
    FV = 1850

    def behave(self, snapshot: TradingState) -> None:
        od = snapshot.order_depths.get(self.ticker)
        if not od:
            return
        pos = snapshot.position.get(self.ticker, 0)
        limit = self.capacity
        fv = self.FV

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_bid_vol = od.buy_orders[best_bid] if best_bid else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_ask_vol = od.sell_orders[best_ask] if best_ask else 0

        if best_ask is not None and best_ask < fv and pos < limit:
            qty = min(-best_ask_vol, limit - pos)
            if qty > 0:
                self.buy(best_ask, qty)
                pos += qty

        if best_bid is not None and best_bid > fv and pos > -limit:
            qty = min(best_bid_vol, limit + pos)
            if qty > 0:
                self.sell(best_bid, qty)
                pos -= qty


############################################
# Trader Class
############################################
class Trader:
    def __init__(self) -> None:
        caps = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "ORCHIDS": 100,
            "JAMS": 350,
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        # parameters for product‑specific logic (drawn from alternate code)
        self.params = {
            "RAINFOREST_RESIN": {"fair_value": 10000},
            "KELP": {"fair_value": 2026},
            "SQUID_INK": {"fair_value": 1850},
        }

        # Map symbols handled by generic strategies
        self.strategies: Dict[Symbol, ApproachBase] = {
            sym: cls(sym, caps[sym])
            for sym, cls in {
                # Generic strategies remain
                # "ORCHIDS": FlowerArrangementMethod,
                "JAMS": HamperDealApproach,
                "CROISSANTS": HamperDealApproach,
                "DJEMBES": HamperDealApproach,
                "PICNIC_BASKET1": HamperDealApproach,
                "PICNIC_BASKET2": HamperDealApproach,
            }.items()
        }

        # Custom logic strategies for replaced products
        self.special_strategies: Dict[Symbol, ApproachBase] = {
            "RAINFOREST_RESIN": RainforestResinLogic("RAINFOREST_RESIN", caps["RAINFOREST_RESIN"]),
            "KELP": KelpLogic("KELP", caps["KELP"]),
            "SQUID_INK": SquidInkLogic("SQUID_INK", caps["SQUID_INK"]),
        }

        self.coupon_machine = IgneousStoneCouponManager()

    def run(self, snap: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        final_request: Dict[Symbol, List[Order]] = {}
        conv_sum = 0

        # Reconstruct old data if present (only for generic strategies)
        old_persist = {}
        if snap.traderData != "":
            try:
                old_persist = json.loads(snap.traderData)
            except:
                old_persist = {}

        new_persist: Dict[str, JSON] = {}

        # 1) generic strategies
        for sm, st in self.strategies.items():
            if sm in old_persist:
                st.restore(old_persist[sm])
            if sm in snap.order_depths:
                st_orders, st_convs = st.process(snap)
                final_request[sm] = st_orders
                conv_sum += st_convs
            new_persist[sm] = st.preserve()

        # 2) special strategies (Rainforest, Kelp, Squid)
        for sm, st in self.special_strategies.items():
            if sm in snap.order_depths:
                st_orders, _ = st.process(snap)
                if st_orders:
                    final_request[sm] = st_orders
            # no persistent state for these now

        # 3) coupon manager
        coupon_out, coupon_convs, _ = self.coupon_machine.run(snap)
        for pr_s, pr_ordz in coupon_out.items():
            final_request.setdefault(pr_s, []).extend(pr_ordz)
        conv_sum += coupon_convs

        updated_data = json.dumps(new_persist, separators=(",", ":"))
        activity_logger.finalize(snap, final_request, conv_sum, updated_data)
        return final_request, conv_sum, updated_data
