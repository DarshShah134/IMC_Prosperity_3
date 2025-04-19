import json
import math
import statistics
from abc import abstractmethod
from collections import deque
from typing import Any, TypeAlias, List, Dict, Tuple

# External libs for state encoding and vector maths (needed for spread logic)
import jsonpickle
import numpy as np

# We still import the same data‑model classes, with the same names used in your environment.
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
# Type Definitions & Globals
############################################
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
DEFAULT_TRADE_SIZE = 10
ROLLING_WINDOW = 20  # generic rolling‑window constant for stats



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
# MarketBalancer (dynamic market‑maker)
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
        valuation = self.get_valuation(snapshot)

        if self.ticker not in snapshot.order_depths:
            return

        od = snapshot.order_depths[self.ticker]
        sorted_buys = sorted(od.buy_orders.items(), reverse=True)
        sorted_sells = sorted(od.sell_orders.items())

        pos = snapshot.position.get(self.ticker, 0)
        remain_buy = self.capacity - pos
        remain_sell = self.capacity + pos

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

        buy_ceiling = int(valuation * 0.999) if pos > self.capacity * 0.5 else valuation
        sell_floor = int(valuation * 1.001) if pos < self.capacity * -0.5 else valuation

        for p, volume in sorted_sells:
            if remain_buy > 0 and p <= buy_ceiling:
                can_take = min(remain_buy, -volume)
                self.buy(p, can_take)
                remain_buy -= can_take

        if remain_buy > 0 and hard_liquid:
            self.buy(buy_ceiling, remain_buy // 2)
            remain_buy = remain_buy // 2

        if remain_buy > 0 and soft_liquid:
            self.buy(buy_ceiling - 2, remain_buy // 2)
            remain_buy = remain_buy // 2

        if remain_buy > 0:
            if sorted_buys:
                top_bid = max(sorted_buys, key=lambda x: x[1])[0]
            else:
                top_bid = buy_ceiling
            final_buy = min(buy_ceiling, top_bid + 1)
            self.buy(final_buy, remain_buy)

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
# Example dynamic Market Maker strategies
############################################
class AmethystsFlow(MarketBalancer):
    def get_valuation(self, snapshot: TradingState) -> int:
        return 10_000


class StarfruitFlow(MarketBalancer):
    def get_valuation(self, snapshot: TradingState) -> int:
        if self.ticker not in snapshot.order_depths:
            return 5000
        depth_info = snapshot.order_depths[self.ticker]
        if not depth_info.buy_orders or not depth_info.sell_orders:
            return 5000
        b_arr = sorted(depth_info.buy_orders.items(), reverse=True)
        s_arr = sorted(depth_info.sell_orders.items())
        best_b = b_arr[0][0] if b_arr else 0
        best_s = s_arr[0][0] if s_arr else 0
        return int((best_b + best_s) / 2)


############################################
# Orchids Strategy => FlowerArrangementMethod
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
# HamperDealApproach (generic for JAMS & PICNIC_BASKET2)
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
# IgneousStoneCouponManager (Volcanic logic)
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
# Custom Logic for Resins, Kelp, and Squid
############################################
class RainforestResinLogic(ApproachBase):
    FV = 10000

    def _clear_position_order(self, fv: float, width: int, orders: List[Order], od: OrderDepth, position: int, buy_vol: int, sell_vol: int) -> Tuple[int, int]:
        pos_after = position + buy_vol - sell_vol
        fair_bid = round(fv - width)
        fair_ask = round(fv + width)
        buy_qty = self.capacity - (position + buy_vol)
        sell_qty = self.capacity + (position - sell_vol)
        if pos_after > 0:
            clear_qty = sum(vol for price, vol in od.buy_orders.items() if price >= fair_ask)
            clear_qty = min(clear_qty, pos_after)
            send_qty = min(sell_qty, clear_qty)
            if send_qty > 0:
                orders.append(Order(self.ticker, fair_ask, -abs(send_qty)))
                sell_vol += abs(send_qty)
        if pos_after < 0:
            clear_qty = sum(abs(vol) for price, vol in od.sell_orders.items() if price <= fair_bid)
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
            fv, 1, self.current_orders, od, pos, buy_vol, sell_vol
        )
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

    def _clear_position_order_kelp(self, od: OrderDepth, pos: int, limit: int, fv: float, buy_fill: int, sell_fill: int) -> Tuple[int, int]:
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
            if volume != 0 else mmmid
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
        buy_fill, sell_fill = self._clear_position_order_kelp(od, pos, limit, fv, buy_fill, sell_fill)
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
# Spread Arbitrage Manager replacing logic for
# PICNIC_BASKET1, CROISSANTS, DJEMBES
############################################
class BasketSpreadManager:
    """Implements the alternate spread / micro‑price arbitrage over
    PICNIC_BASKET1, CROISSANTS, DJEMBES (and JAMS as component)."""

    # Copy of constants directly from the alternate reference code
    class Product:
        DJEMBES = "DJEMBES"
        JAMS = "JAMS"
        CROISSANTS = "CROISSANTS"
        PICNIC_BASKET1 = "PICNIC_BASKET1"

    PARAMS = {
        "SPREAD1": {
            "default_spread_mean": 48.777856,
            "default_spread_std": 85.119723,
            "spread_window": 55,
            "zscore_threshold": 4,
            "target_position": 100,
        },
        Product.PICNIC_BASKET1: {
            "adverse_volume": 999999,
            "b2_adjustment_factor": 0.05,
        },
    }

    POSITION_LIMITS = {
        Product.DJEMBES: 60,
        Product.JAMS: 350,
        Product.CROISSANTS: 250,
        Product.PICNIC_BASKET1: 60,
    }

    def __init__(self, params=None):
        self.params = params or self.PARAMS
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helper market computations (exact copies from alternate code)
    # ------------------------------------------------------------------
    def _get_microprice(self, od: OrderDepth) -> float:
        if not od.buy_orders or not od.sell_orders:
            return 0.0
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bid_vol = abs(od.buy_orders[best_bid])
        ask_vol = abs(od.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (ask_vol + bid_vol)

    def _artificial_order_depth(self, order_depths: dict, picnic1: bool = True) -> OrderDepth:
        od = OrderDepth()
        if picnic1:
            DJ_PER, CR_PER, JA_PER = 1, 6, 3
        else:
            CR_PER, JA_PER = 4, 2
        cro_od = order_depths.get(self.Product.CROISSANTS)
        ja_od = order_depths.get(self.Product.JAMS)
        if not cro_od or not ja_od:
            return od
        cbid = max(cro_od.buy_orders) if cro_od.buy_orders else 0
        cask = min(cro_od.sell_orders) if cro_od.sell_orders else float("inf")
        jbid = max(ja_od.buy_orders) if ja_od.buy_orders else 0
        jask = min(ja_od.sell_orders) if ja_od.sell_orders else float("inf")
        if picnic1:
            dj_od = order_depths.get(self.Product.DJEMBES)
            if not dj_od:
                return od
            dbid = max(dj_od.buy_orders) if dj_od.buy_orders else 0
            dask = min(dj_od.sell_orders) if dj_od.sell_orders else float("inf")
            bid = dbid * DJ_PER + cbid * CR_PER + jbid * JA_PER
            ask = dask * DJ_PER + cask * CR_PER + jask * JA_PER
        else:
            bid = cbid * CR_PER + jbid * JA_PER
            ask = cask * CR_PER + jask * JA_PER
        if 0 < bid < float("inf"):
            od.buy_orders[bid] = 1
        if 0 < ask < float("inf"):
            od.sell_orders[ask] = -1
        return od

    def _convert_orders(self, artificial_orders: List[Order], order_depths: dict, picnic1: bool = True):
        if picnic1:
            out = {self.Product.DJEMBES: [], self.Product.CROISSANTS: [], self.Product.JAMS: []}
            DJ_PER, CR_PER, JA_PER = 1, 6, 3
        else:
            out = {self.Product.CROISSANTS: [], self.Product.JAMS: []}
            CR_PER, JA_PER = 4, 2
        art_od = self._artificial_order_depth(order_depths, picnic1)
        best_bid = max(art_od.buy_orders) if art_od.buy_orders else 0
        best_ask = min(art_od.sell_orders) if art_od.sell_orders else float("inf")
        for o in artificial_orders:
            price, qty = o.price, o.quantity
            if qty > 0 and price >= best_ask:
                cp = min(order_depths[self.Product.CROISSANTS].sell_orders)
                jp = min(order_depths[self.Product.JAMS].sell_orders)
                if picnic1:
                    dp = min(order_depths[self.Product.DJEMBES].sell_orders)
            elif qty < 0 and price <= best_bid:
                cp = max(order_depths[self.Product.CROISSANTS].buy_orders)
                jp = max(order_depths[self.Product.JAMS].buy_orders)
                if picnic1:
                    dp = max(order_depths[self.Product.DJEMBES].buy_orders)
            else:
                continue
            out[self.Product.CROISSANTS].append(Order(self.Product.CROISSANTS, cp, qty * CR_PER))
            out[self.Product.JAMS].append(Order(self.Product.JAMS, jp, qty * JA_PER))
            if picnic1:
                out[self.Product.DJEMBES].append(Order(self.Product.DJEMBES, dp, qty * DJ_PER))
        return out

    def _execute_spreads(self, target_pos: int, basket_pos: int, order_depths: dict, picnic1: bool = True):
        basket = self.Product.PICNIC_BASKET1
        if target_pos == basket_pos:
            return None
        diff = abs(target_pos - basket_pos)
        pic_od = order_depths[basket]
        art_od = self._artificial_order_depth(order_depths, picnic1)
        if target_pos > basket_pos:
            if not pic_od.sell_orders or not art_od.buy_orders:
                return None
            p_ask = min(pic_od.sell_orders)
            p_vol = abs(pic_od.sell_orders[p_ask])
            a_bid = max(art_od.buy_orders)
            a_vol = art_od.buy_orders[a_bid]
            vol = min(p_vol, abs(a_vol), diff)
            pic_orders = [Order(basket, p_ask, vol)]
            art_orders = [Order("ARTIFICAL1", a_bid, -vol)]
            agg = self._convert_orders(art_orders, order_depths, picnic1)
            agg[basket] = pic_orders
            return agg
        if not pic_od.buy_orders or not art_od.sell_orders:
            return None
        p_bid = max(pic_od.buy_orders)
        p_vol = pic_od.buy_orders[p_bid]
        a_ask = min(art_od.sell_orders)
        a_vol = abs(art_od.sell_orders[a_ask])
        vol = min(p_vol, a_vol, diff)
        pic_orders = [Order(basket, p_bid, -vol)]
        art_orders = [Order("ARTIFICAL1", a_ask, -vol)]
        agg = self._convert_orders(art_orders, order_depths, picnic1)
        agg[basket] = pic_orders
        return agg

    def _spread_orders(self, order_depths: dict, picnic_product: str, picnic_pos: int, spread_data: dict, spread_key: str, picnic1: bool = True):
        basket = self.Product.PICNIC_BASKET1
        if basket not in order_depths:
            return None
        pic_od = order_depths[basket]
        art_od = self._artificial_order_depth(order_depths, picnic1)
        spread = self._get_microprice(pic_od) - self._get_microprice(art_od)
        spread_data["spread_history"].append(spread)
        if len(spread_data["spread_history"]) < self.params[spread_key]["spread_window"]:
            return None
        if len(spread_data["spread_history"]) > self.params[spread_key]["spread_window"]:
            spread_data["spread_history"].pop(0)
        arr = np.array(spread_data["spread_history"])
        std = np.std(arr)
        if std == 0:
            return None
        z = (spread - self.params[spread_key]["default_spread_mean"]) / std
        if z >= self.params[spread_key]["zscore_threshold"] and picnic_pos != -self.params[spread_key]["target_position"]:
            return self._execute_spreads(-self.params[spread_key]["target_position"], picnic_pos, order_depths, picnic1)
        if z <= -self.params[spread_key]["zscore_threshold"] and picnic_pos != self.params[spread_key]["target_position"]:
            return self._execute_spreads(self.params[spread_key]["target_position"], picnic_pos, order_depths, picnic1)
        spread_data["prev_zscore"] = z
        return None

    # ------------------------------------------------------------------
    # Public interface expected by main Trader
    # ------------------------------------------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if "SPREAD1" not in self.state:
            self.state["SPREAD1"] = {"spread_history": [], "prev_zscore": 0}
        pos_basket = state.position.get(self.Product.PICNIC_BASKET1, 0)
        orders = self._spread_orders(state.order_depths, self.Product.PICNIC_BASKET1, pos_basket, self.state["SPREAD1"], spread_key="SPREAD1", picnic1=True)
        result: Dict[str, List[Order]] = {}
        if orders:
            for sym, lst in orders.items():
                if lst:
                    result[sym] = lst
        # No conversions are used here, return 0 for convs
        return result, 0, jsonpickle.encode(self.state)

# JAMS LOGIC –– **newly ported from alternate code**
############################################
class JamsLogic(ApproachBase):
    FV = 6500

    def __init__(self, ticker: str, capacity: int):
        super().__init__(ticker, capacity)
        self.price_hist: List[float] = []

    # ---------------- helper -----------------
    @staticmethod
    def _mid_price(od: OrderDepth) -> float | None:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2
        if od.buy_orders:
            return max(od.buy_orders)
        if od.sell_orders:
            return min(od.sell_orders)
        return None

    # ---------------- main behaviour ---------
    def behave(self, snapshot: TradingState):
        od = snapshot.order_depths.get(self.ticker)
        if not od:
            return
        pos = snapshot.position.get(self.ticker, 0)
        lim = self.capacity

        mid = self._mid_price(od)
        if mid is not None:
            self.price_hist.append(mid)
            if len(self.price_hist) > 2 * ROLLING_WINDOW:
                self.price_hist = self.price_hist[-2 * ROLLING_WINDOW :]

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        # Simple fair‑value reference strategy as in alternate code
        fv = self.FV
        if best_ask is not None and best_ask < fv:
            qty = min(lim - pos, -od.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
            if qty > 0:
                self.buy(best_ask, qty)
        if best_bid is not None and best_bid > fv:
            qty = min(lim + pos, od.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
            if qty > 0:
                self.sell(best_bid, qty)

    # persistence (optional)
    def preserve(self) -> JSON:
        return self.price_hist[-ROLLING_WINDOW:]

    def restore(self, data: JSON):
        if isinstance(data, list):
            self.price_hist = data

class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}

############################################
# Trader
############################################
import json, jsonpickle, math, numpy as np
from typing import Dict, List, Tuple, Any

# --------------- make sure these come from your existing code -------------
# from datamodel import Order, TradingState, Symbol
# from helpers      import MarketData, activity_logger, logger
# from strategies   import (RainforestResinLogic, KelpLogic, SquidInkLogic,
#                           JamsLogic, HamperDealApproach, BasketSpreadManager,
#                           IgneousStoneCouponManager, ApproachBase)

# ------------------------------------------------------------------------- #
class Trader:
    # ---------------- static, cross‑tick fields ------------------------- #
    limits: Dict[str, int] = {
        "KELP": 50,
        "RAINFOREST_RESIN": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
        "VOLCANIC_ROCK": 400,
        "MAGNIFICENT_MACARONS": 75,
    }

    def __init__(self) -> None:
        # ----------- single‑symbol bespoke strategies ------------------ #
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

        self.special_strategies: Dict[Symbol, ApproachBase] = {
            "RAINFOREST_RESIN": RainforestResinLogic("RAINFOREST_RESIN", caps["RAINFOREST_RESIN"]),
            "KELP": KelpLogic("KELP", caps["KELP"]),
            "SQUID_INK": SquidInkLogic("SQUID_INK", caps["SQUID_INK"]),
            "JAMS": JamsLogic("JAMS", caps["JAMS"]),
        }

        # ----------- generic Hamper / Flower etc ----------------------- #
        self.strategies: Dict[Symbol, ApproachBase] = {
            "PICNIC_BASKET2": HamperDealApproach("PICNIC_BASKET2", caps["PICNIC_BASKET2"]),
            # "ORCHIDS": FlowerArrangementMethod("ORCHIDS", caps["ORCHIDS"]),
        }

        # ----------- spread & coupon managers -------------------------- #
        self.basket_manager = BasketSpreadManager()
        self.coupon_machine = IgneousStoneCouponManager()

        # ----------- macaron‑specific rolling state -------------------- #
        self.recent_std: float = 0.0
        self.last_sunlight: float = 0.0
        self.sunlight_history: List[float] = []   # last 5 readings

    # ========================  helper: sunlight ROC  ===================== #
    def _sunlight_roc(self) -> float:
        if len(self.sunlight_history) < 5:
            return 0.0
        diffs = [self.sunlight_history[i] - self.sunlight_history[i - 1]
                 for i in range(1, len(self.sunlight_history))]
        return sum(diffs) / len(diffs)

    # ========================  macaron helpers  ========================== #
    # (verbatim from your previous class, but trimmed to use `self.limits`)
    def take_macaron(self, state, md) -> Tuple[List["Order"], int]:
        product = "MAGNIFICENT_MACARONS"
        odrs: List["Order"] = []
        convs = 0

        conv_obs = getattr(state.observations, "conversionObservations", {}).get(product)
        if conv_obs is None:                           # no quote this tick
            return odrs, convs

        # overseas effective prices
        over_ask = conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff
        over_bid = conv_obs.bidPrice - conv_obs.transportFees - conv_obs.exportTariff
        cur_sun  = conv_obs.sunlightIndex

        # direction of last change
        if cur_sun < self.last_sunlight:
            direction = -1
        elif cur_sun > self.last_sunlight:
            direction = 1
        else:
            direction = 0

        self.sunlight_history.append(cur_sun)
        if len(self.sunlight_history) > 5:
            self.sunlight_history.pop(0)
        self.last_sunlight = cur_sun

        # book aggregates
        tot_bids = sum(md.bid_volumes[product])
        tot_asks = -sum(md.ask_volumes[product])

        # mean‑reversion constants
        mean_p  = 640
        sigma   = 55
        z       = (md.fair[product] - mean_p) / sigma

        # ---------- strategy logic ------------------------------------ #
        if cur_sun < 50:
            if direction == -1 and md.buy_sum[product] > 0:
                amt = min(md.buy_sum[product], -tot_asks)
                for px, vol in zip(md.ask_prices[product], md.ask_volumes[product]):
                    fill = min(-vol, amt)
                    if fill:
                        odrs.append(Order(product, px, fill))
                        amt -= fill
                convs += 0  # (keep conversion logic separate if you have it)

            elif (direction == 1 and md.sell_sum[product] > 0 and
                  self._sunlight_roc() > 0.008):
                amt = min(md.sell_sum[product], tot_bids)
                for px, vol in zip(md.bid_prices[product], md.bid_volumes[product]):
                    fill = min(vol, amt)
                    if fill:
                        odrs.append(Order(product, px, -fill))
                        amt -= fill

            elif abs(cur_sun - 49) < 1 and md.end_pos[product] < 0:
                amt = min(md.buy_sum[product], -md.end_pos[product])
                for px, vol in zip(md.ask_prices[product], md.ask_volumes[product]):
                    fill = min(-vol, amt)
                    if fill:
                        odrs.append(Order(product, px, fill))
                        amt -= fill

        elif cur_sun > 50:
            if z < -1.2 and md.buy_sum[product] > 0:
                amt = min(md.buy_sum[product], -tot_asks)
                for px, vol in zip(md.ask_prices[product], md.ask_volumes[product]):
                    fill = min(-vol, amt)
                    if fill:
                        odrs.append(Order(product, px, fill))
                        amt -= fill
            elif z > 1.2 and md.sell_sum[product] > 0:
                amt = min(md.sell_sum[product], tot_bids)
                for px, vol in zip(md.bid_prices[product], md.bid_volumes[product]):
                    fill = min(vol, amt)
                    if fill:
                        odrs.append(Order(product, px, -fill))
                        amt -= fill
        return odrs, convs

    def make_macaron(self, state, md) -> List["Order"]:
        p = "MAGNIFICENT_MACARONS"
        odrs: List["Order"] = []
        fair = md.fair[p]
        pos  = md.end_pos[p]

        bid = math.floor(fair - 4)
        ask = math.ceil (fair + 4)
        slice_sz = 14

        buy_cap  = self.limits[p] - pos
        sell_cap = self.limits[p] + pos
        if buy_cap > 0:
            odrs.append(Order(p, bid, min(slice_sz, buy_cap)))
        if sell_cap > 0:
            odrs.append(Order(p, ask, -min(slice_sz, sell_cap)))
        return odrs

    def clear_macaron(self, state, md) -> List["Order"]:
        p = "MAGNIFICENT_MACARONS"
        odrs: List["Order"] = []
        fair = md.fair[p]
        pos  = md.end_pos[p]
        width = 3 if self.recent_std > 7 else 4
        if pos > 0:
            odrs.append(Order(p, round(fair + width), -pos))
        elif pos < 0:
            odrs.append(Order(p, round(fair - width), -pos))
        return odrs

    # ===========================  main entry  ============================ #
    def run(self, snap: "TradingState"):
        out: Dict[Symbol, List["Order"]] = {}
        convs = 0

        # ---------- restore persisted JSON -------------------------------- #
        try:
            persist_prev: Dict[str, Any] = (
                json.loads(snap.traderData) if snap.traderData else {}
            )
        except Exception:
            persist_prev = {}

        # macaron‑specific rolling lists
        prev_mac_prices: List[float] = persist_prev.get("prev_mac_prices", [])
        self.sunlight_history        = persist_prev.get("sun_hist",  [])
        self.last_sunlight           = persist_prev.get("last_sun", 0.0)

        # ---------- rebuild MarketData snapshot --------------------------- #
        md = MarketData()
        for prod in self.limits:
            pos  = snap.position.get(prod, 0)
            od   = snap.order_depths.get(prod)
            bids = od.buy_orders if od else {}
            asks = od.sell_orders if od else {}

            if bids:
                mm_bid = max(bids)
                mm_ask = min(asks) if asks else mm_bid
                fair   = (mm_ask + mm_bid) / 2
            elif asks:
                fair = min(asks)
            else:
                fair = 0

            md.end_pos[prod]   = pos
            md.buy_sum[prod]   = self.limits[prod] - pos
            md.sell_sum[prod]  = self.limits[prod] + pos
            md.bid_prices[prod] = list(bids.keys())
            md.bid_volumes[prod] = list(bids.values())
            md.ask_prices[prod] = list(asks.keys())
            md.ask_volumes[prod] = list(asks.values())
            md.fair[prod]      = fair

        # ---------- generic Hamper / Flower etc --------------------------- #
        # restore basket‑spread sub‑state first
        self.basket_manager.state = jsonpickle.decode(
            persist_prev.get("BASKET_SPREAD", "{}")
        ) if persist_prev.get("BASKET_SPREAD") else {}

        persist_next: Dict[str, Any] = {}

        for sym, strat in self.strategies.items():
            strat.restore(persist_prev.get(sym))
            if sym in snap.order_depths:
                odrs, cv = strat.process(snap)
                if odrs:
                    out[sym] = odrs
                convs += cv
            persist_next[sym] = strat.preserve()

        # ---------- bespoke single‑symbol strategies --------------------- #
        for sym, strat in self.special_strategies.items():
            strat.restore(persist_prev.get(sym))
            if sym in snap.order_depths:
                odrs, _ = strat.process(snap)
                if odrs:
                    out[sym] = odrs
            persist_next[sym] = strat.preserve()

        # ---------- basket spread manager -------------------------------- #
        b_orders, _, b_state = self.basket_manager.run(snap)
        for s, ol in b_orders.items():
            out.setdefault(s, []).extend(ol)
        persist_next["BASKET_SPREAD"] = b_state

        # ---------- coupon arbitrage ------------------------------------- #
        c_orders, c_convs, _ = self.coupon_machine.run(snap)
        for s, ol in c_orders.items():
            out.setdefault(s, []).extend(ol)
        convs += c_convs

        # ---------- MAGNIFICENT MACARONS logic --------------------------- #
        prev_mac_prices.append(md.fair["MAGNIFICENT_MACARONS"])
        if len(prev_mac_prices) > 13:
            prev_mac_prices.pop(0)
        self.recent_std = float(np.std(prev_mac_prices))

        mac_take, mac_convs = self.take_macaron(snap, md)
        mac_make = mac_clear = []
        if self.recent_std < 53:
            mac_make  = self.make_macaron(snap, md)
            mac_clear = self.clear_macaron(snap, md)

        mac_orders = mac_take + mac_make + mac_clear
        if mac_orders:
            out["MAGNIFICENT_MACARONS"] = mac_orders
        convs += mac_convs

        # ---------- persist everything for next tick --------------------- #
        persist_next.update(
            prev_mac_prices=prev_mac_prices,
            sun_hist=self.sunlight_history,
            last_sun=self.last_sunlight,
        )

        trader_data_out = json.dumps(persist_next, separators=(",", ":"))
        activity_logger.finalize(snap, out, convs, trader_data_out)
        return out, convs, trader_data_out