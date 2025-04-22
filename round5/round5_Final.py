from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Any, Dict, Tuple, TypeAlias, Optional
import json
import jsonpickle
import numpy as np
import math
import collections
import pandas as pd
import copy
from collections import deque
from math import sqrt, log, erf
import statistics
from copy import deepcopy

# =============================================================================
# Alternate Helpers – compact logger (unchanged)
# =============================================================================
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

DEFAULT_TRADE_SIZE = 10
ROLLING_WINDOW = 20

class ActivityRecorderAlt:
    def __init__(self) -> None:
        self.cached_logs = ""
        self.maximum_log_size = 3750

    def output(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        self.cached_logs += sep.join(map(str, args)) + end

    def finalize(self, st: TradingState, orders: Dict[Symbol, List[Order]], conv: int, tdata: str):
        prelim = len(self._json([self._brief(st, ""), self._minify(orders), conv, "", ""]))
        slice_len = (self.maximum_log_size - prelim) // 3
        print(self._json([
            self._brief(st, self._trim(st.traderData, slice_len)),
            self._minify(orders),
            conv,
            self._trim(tdata, slice_len),
            self._trim(self.cached_logs, slice_len),
        ]))
        self.cached_logs = ""

    def _trim(self, s: str, n: int) -> str:
        return s if len(s) <= n else s[: n - 3] + "..."

    def _json(self, v: Any) -> str:
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def _brief(self, st: TradingState, txt: str) -> List[Any]:
        return [
            st.timestamp,
            txt,
            [[l.symbol, l.product, l.denomination] for l in st.listings.values()],
            {s: [d.buy_orders, d.sell_orders] for s, d in st.order_depths.items()},
            [[t.symbol, t.price, t.quantity] for arr in st.own_trades.values() for t in arr],
            [[t.symbol, t.price, t.quantity] for arr in st.market_trades.values() for t in arr],
            st.position,
            [st.observations.plainValueObservations, {
                p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                for p, o in st.observations.conversionObservations.items()
            }],
        ]

    def _minify(self, odm: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for lst in odm.values() for o in lst]

activity_logger_alt = ActivityRecorderAlt()

# =============================================================================
# Core abstraction for baskets
# =============================================================================
class ApproachBase:
    def __init__(self, ticker: str, capacity: int):
        self.ticker = ticker
        self.capacity = capacity
        self.current_orders: List[Order] = []
        self.num_conversions = 0

    def buy(self, p: int, q: int):
        self.current_orders.append(Order(self.ticker, p, q))

    def sell(self, p: int, q: int):
        self.current_orders.append(Order(self.ticker, p, -q))

    def convert(self, n: int):
        self.num_conversions += n

    def preserve(self) -> JSON:
        return None

    def restore(self, data: JSON):
        pass

    def process(self, snap: TradingState) -> Tuple[List[Order], int]:
        self.current_orders, self.num_conversions = [], 0
        self.behave(snap)
        return self.current_orders, self.num_conversions

    def behave(self, snap: TradingState):
        raise NotImplementedError

# =============================================================================
# Basket 2 – HamperDealApproach (unchanged)
# =============================================================================
class HamperDealApproach(ApproachBase):
    def __init__(self, ticker: str, cap: int):
        super().__init__(ticker, cap)
        self.hist = deque(maxlen=50)

    def _mid(self, st: TradingState, sym: str) -> float:
        od = st.order_depths.get(sym)
        if not od or not od.buy_orders or not od.sell_orders:
            return 0.0
        return (max(od.buy_orders) + min(od.sell_orders)) / 2

    def _do_side(self, st: TradingState, is_buy: bool):
        od = st.order_depths[self.ticker]
        if not od:
            return
        pos = st.position.get(self.ticker, 0)
        qty = (self.capacity - pos) if is_buy else (self.capacity + pos)
        if qty <= 0:
            return
        price = min(od.sell_orders) if is_buy else max(od.buy_orders)
        if price is not None:
            (self.buy if is_buy else self.sell)(price, qty)

    def behave(self, st: TradingState):
        need = ["JAMS", "CROISSANTS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        if any(n not in st.order_depths for n in need):
            return
        c, s = self._mid(st, "JAMS"), self._mid(st, "CROISSANTS")
        b2 = self._mid(st, "PICNIC_BASKET2")
        offset = b2 - 4 * s - 2 * c
        self.hist.append(offset)
        avg = statistics.mean(self.hist) if len(self.hist) > 1 else offset
        std = statistics.pstdev(self.hist) if len(self.hist) > 1 else 1
        if offset < avg - std and offset < 50:
            self._do_side(st, True)
        elif offset > avg + std and offset > 100:
            self._do_side(st, False)

# =============================================================================
# Basket 1 – BasketSpreadManager (unchanged)
# =============================================================================
class BasketSpreadManager:
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

    def __init__(self):
        self.state: Dict[str, Any] = {}

    def _get_microprice(self, od: OrderDepth) -> float:
        if not od.buy_orders or not od.sell_orders:
            return 0.0
        bid, ask = max(od.buy_orders), min(od.sell_orders)
        bv, av = abs(od.buy_orders[bid]), abs(od.sell_orders[ask])
        return (bid * av + ask * bv) / (av + bv)

    def _artificial_order_depth(self, ods: dict, picnic1: bool = True) -> OrderDepth:
        od = OrderDepth()
        if picnic1:
            DJ_PER, CR_PER, JA_PER = 1, 6, 3
        else:
            CR_PER, JA_PER = 4, 2
        cro, jam = ods.get(self.Product.CROISSANTS), ods.get(self.Product.JAMS)
        if not cro or not jam:
            return od
        cbid = max(cro.buy_orders) if cro.buy_orders else 0
        cask = min(cro.sell_orders) if cro.sell_orders else float("inf")
        jbid = max(jam.buy_orders) if jam.buy_orders else 0
        jask = min(jam.sell_orders) if jam.sell_orders else float("inf")
        if picnic1:
            dj = ods.get(self.Product.DJEMBES)
            if not dj:
                return od
            dbid = max(dj.buy_orders) if dj.buy_orders else 0
            dask = min(dj.sell_orders) if dj.sell_orders else float("inf")
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

    def _execute_spreads(self, target: int, pos: int, ods: dict, picnic1: bool = True):
        basket = self.Product.PICNIC_BASKET1
        if target == pos:
            return None
        diff = abs(target - pos)
        pic_od = ods[basket]
        art_od = self._artificial_order_depth(ods, picnic1)
        if target > pos:
            if not pic_od.sell_orders or not art_od.buy_orders:
                return None
            p_ask = min(pic_od.sell_orders)
            p_vol = abs(pic_od.sell_orders[p_ask])
            a_bid = max(art_od.buy_orders)
            a_vol = art_od.buy_orders[a_bid]
            vol = min(p_vol, abs(a_vol), diff)
            return {basket: [Order(basket, p_ask, vol)]}
        if not pic_od.buy_orders or not art_od.sell_orders:
            return None
        p_bid = max(pic_od.buy_orders)
        p_vol = pic_od.buy_orders[p_bid]
        a_ask = min(art_od.sell_orders)
        a_vol = abs(art_od.sell_orders[a_ask])
        vol = min(p_vol, a_vol, diff)
        return {basket: [Order(basket, p_bid, -vol)]}

    def _spread_orders(self, ods: dict, pos: int):
        basket = self.Product.PICNIC_BASKET1
        if basket not in ods:
            return None
        pic_od = ods[basket]
        art_od = self._artificial_order_depth(ods, True)
        spread = self._get_microprice(pic_od) - self._get_microprice(art_od)
        sdata = self.state.setdefault("SPREAD1", {"hist": []})
        sdata["hist"].append(spread)
        if len(sdata["hist"]) > self.PARAMS["SPREAD1"]["spread_window"]:
            sdata["hist"].pop(0)
        if len(sdata["hist"]) < self.PARAMS["SPREAD1"]["spread_window"]:
            return None
        std = np.std(sdata["hist"]) or 1
        z = (spread - self.PARAMS["SPREAD1"]["default_spread_mean"]) / std
        tgt = self.PARAMS["SPREAD1"]["target_position"]
        if z >= self.PARAMS["SPREAD1"]["zscore_threshold"] and pos != -tgt:
            return self._execute_spreads(-tgt, pos, ods, True)
        if z <= -self.PARAMS["SPREAD1"]["zscore_threshold"] and pos != tgt:
            return self._execute_spreads(tgt, pos, ods, True)
        return None

    def run(self, st: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        pos = st.position.get(self.Product.PICNIC_BASKET1, 0)
        odict = self._spread_orders(st.order_depths, pos)
        return (odict or {}), 0, jsonpickle.encode(self.state)

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MACARONS = "MAGNIFICENT_MACARONS"
    options = [
        VOLCANIC_ROCK_VOUCHER_10000,
        VOLCANIC_ROCK_VOUCHER_9500,
        VOLCANIC_ROCK_VOUCHER_9750,
        VOLCANIC_ROCK_VOUCHER_10250,
        VOLCANIC_ROCK_VOUCHER_10500,
    ]

PARAMS = {
    Product.VOLCANIC_ROCK: {
        "ma_length": 100,
        "open_threshold": -1.6,
        "close_threshold": 1.6,
        "short_ma_length": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "ma_length": 150,
        "open_threshold": -1.1,
        "close_threshold": 1.1,
        "short_ma_length": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "ma_length": 80,
        "open_threshold": -1.6,
        "close_threshold": 1.6,
        "short_ma_length": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "ma_length": 100,
        "open_threshold": -1.6,
        "close_threshold": 1.6,
        "short_ma_length": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "ma_length": 80,
        "open_threshold": -1.6,
        "close_threshold": 1.6,
        "short_ma_length": 10,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "ma_length": 500,
        "open_threshold": -1.5,
        "close_threshold": 1.5,
        "short_ma_length": 5,
    },

}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate("", max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            "",
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class ParabolaFitIVStrategy:
    def __init__(self, voucher: str, strike: int, adaptive: bool = False, absolute: bool = False):
        self.voucher = voucher
        self.strike = strike
        self.adaptive = adaptive
        self.absolute = absolute
        self.expiry_day = 3
        self.ticks_per_day = 1000
        self.window = 500
        self.position_limit = 200
        self.start_ts = None
        self.history = deque(maxlen=self.window)
        self.iv_cache = {}
        self.a = self.b = self.c = None

    def norm_cdf(self, x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    def bs_price(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(S - K, 0)
        d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def implied_vol(self, S, K, T, price, tol=1e-4, max_iter=50):
        key = (round(S, 1), round(K, 1), round(T, 5), round(price, 1))
        if key in self.iv_cache:
            return self.iv_cache[key]
        low, high = 1e-6, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            val = self.bs_price(S, K, T, mid) - price
            if abs(val) < tol:
                self.iv_cache[key] = mid
                return mid
            if val > 0:
                high = mid
            else:
                low = mid
        return None

    def update_fit(self):
        m_vals = [m for m, v in self.history]
        v_vals = [v for m, v in self.history]
        self.a, self.b, self.c = np.polyfit(m_vals, v_vals, 2)

    def fitted_iv(self, m):
        return self.a * m ** 2 + self.b * m + self.c

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        ts = state.timestamp
        if self.start_ts is None:
            self.start_ts = ts

        depth = state.order_depths.get(self.voucher, OrderDepth())
        rock_depth = state.order_depths.get(Product.VOLCANIC_ROCK, OrderDepth())
        if not depth.sell_orders or not depth.buy_orders:
            return {}, 0, ""
        if not rock_depth.sell_orders or not rock_depth.buy_orders:
            return {}, 0, ""

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        rock_bid = max(rock_depth.buy_orders.keys())
        rock_ask = min(rock_depth.sell_orders.keys())
        spot_price = (rock_bid + rock_ask) / 2

        TTE = max(0.1, self.expiry_day - (ts - self.start_ts) / self.ticks_per_day)
        T = TTE / 365

        if self.absolute:
            iv_guess = 1.1
            fair_value = self.bs_price(spot_price, self.strike, T, iv_guess)
            mispricing = mid_price - fair_value

            position = state.position.get(self.voucher, 0)
            result = []

            if mispricing > 2 and position < self.position_limit:
                qty = min(20, self.position_limit - position)
                result.append(Order(self.voucher, best_ask, qty))
            elif mispricing < -2 and position > -self.position_limit:
                qty = min(20, self.position_limit + position)
                result.append(Order(self.voucher, best_bid, -qty))

            orders[self.voucher] = result
            return orders, 0, ""

        m_t = log(self.strike / spot_price) / sqrt(TTE)
        v_t = self.implied_vol(spot_price, self.strike, T, mid_price)
        if v_t is None or v_t < 0.5:
            return {}, 0, ""

        self.history.append((m_t, v_t))
        if len(self.history) < self.window:
            return {}, 0, ""

        self.update_fit()
        current_fit = self.fitted_iv(m_t)
        position = state.position.get(self.voucher, 0)
        result = []

        if v_t < current_fit - 0.019 and position < self.position_limit:
            qty = min(30, self.position_limit - position)
            result.append(Order(self.voucher, best_ask, qty))
        elif v_t > current_fit + 0.013 and position > -self.position_limit:
            qty = min(30, self.position_limit + position)
            result.append(Order(self.voucher, best_bid, -qty))

        orders[self.voucher] = result
        return orders, 0, ""



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






# --- BS Approximations (unchanged) ---
def approx_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def approx_norm_pdf(x: float) -> float:
    if abs(x) > 30: return 0.0
    try: return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x**2)
    except OverflowError: return 0.0

# --- BS Pricing and Delta (unchanged) ---
def black_scholes_call_price_approx(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Optional[float]:
    # Calculation remains the same...
    if T <= 1e-9 or sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return max(0.0, S - K * math.exp(-r * T))
    sqrtT = math.sqrt(T)
    if sigma * sqrtT < 1e-9: return max(0.0, S - K * math.exp(-r * T))
    try:
        if S / K <= 1e-9 : return max(0.0, S - K * math.exp(-r * T))
        log_term = math.log(S / K)
        d1_num = log_term + (r + 0.5 * sigma**2) * T
        d1_den = sigma * sqrtT
        if abs(d1_den) < 1e-9: return max(0.0, S - K * math.exp(-r*T))
        d1 = d1_num / d1_den
        d2 = d1 - sigma * sqrtT
        call_price = (S * approx_norm_cdf(d1) - K * math.exp(-r * T) * approx_norm_cdf(d2))
        return max(0.0, call_price)
    except (ValueError, OverflowError, ZeroDivisionError): return max(0.0, S - K * math.exp(-r * T))

def black_scholes_delta_call_approx(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Optional[float]:
    # Calculation remains the same...
    if T <= 1e-9 or sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    if sigma * sqrtT < 1e-9: return 1.0 if S > K else 0.0
    try:
        if S / K <= 1e-9: return 1.0 if S > K else 0.0
        log_term = math.log(S / K)
        d1_num = log_term + (r + 0.5 * sigma**2) * T
        d1_den = sigma * sqrtT
        if abs(d1_den) < 1e-9: return 1.0 if S > K else 0.0
        d1 = d1_num / d1_den
        delta = approx_norm_cdf(d1)
        return delta
    except (ValueError, OverflowError, ZeroDivisionError): return 1.0 if S > K else 0.0







class Trader:
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
    
    def __init__(self, params=None):
        self.params = params or PARAMS
        
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50, 
                      Product.PICNIC_BASKET1: 60, Product.CROISSANTS: 250, Product.JAMS: 350,
                      Product.PICNIC_BASKET2: 100,  Product.VOLCANIC_ROCK: 400,
                      Product.VOLCANIC_ROCK_VOUCHER_10000:200, Product.VOLCANIC_ROCK_VOUCHER_10250:200,
                        Product.VOLCANIC_ROCK_VOUCHER_10500:200, Product.VOLCANIC_ROCK_VOUCHER_9750:200,
                        Product.VOLCANIC_ROCK_VOUCHER_9500:200, Product.MACARONS: 65}
                      
        self.signal = {
            Product.RAINFOREST_RESIN: 0,
            Product.KELP: 0,
            Product.SQUID_INK: 0,
            Product.PICNIC_BASKET1: 0,
            Product.CROISSANTS: 0,
            Product.JAMS: 0,
            Product.DJEMBES: 0,
            Product.PICNIC_BASKET2: 0,
            Product.VOLCANIC_ROCK: 0,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 0,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 0,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 0,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 0,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 0,
        }


        self.voucher_strategies = [
            ParabolaFitIVStrategy("VOLCANIC_ROCK_VOUCHER_10000", 10000)
        ]

        self.b2 = HamperDealApproach(Product.PICNIC_BASKET2, self.LIMIT[Product.PICNIC_BASKET2])
        self.b1_mgr = BasketSpreadManager()

        self.special_strategies: Dict[Symbol, ApproachBase] = {
            "RAINFOREST_RESIN": RainforestResinLogic("RAINFOREST_RESIN", self.LIMIT[Product.RAINFOREST_RESIN]),
            "KELP": KelpLogic("KELP", self.LIMIT[Product.KELP]),
            "SQUID_INK": SquidInkLogic("SQUID_INK",self.LIMIT[Product.SQUID_INK]),
            "JAMS": JamsLogic("JAMS", self.LIMIT[Product.JAMS]),
        }
         # ----------- macaron‑specific rolling state -------------------- #
        self.recent_std: float = 0.0
        self.last_sunlight: float = 0.0
        self.sunlight_history: List[float] = []   # last 5 readings



        self.risk_free_rate = 0.0 # Keep as 0 for consistency

        # Voucher strategy parameters (UNCHANGED)
        self.voucher_trade_threshold_base_pct = 0.007 # Base threshold (0.7%) for BS mean reversion
        self.voucher_order_size = 7 # Base order size for BS mean reversion
        self.min_theoretical_iv = 0.01
        self.iv_param_a = 0.2131 # Volatility smile parameters
        self.iv_param_b = 0.0192
        self.iv_param_c = 0.1895
        self.voucher_aggressiveness = 1.5 # How much rock price dev affects thresholds/size in BS strategy
        self.iv_adjustment_sensitivity = 0.5 # How much rock price dev adjusts theoretical IV (0 to disable)

        # Volcanic Rock EMA parameters - OPTIMIZED FOR CONSISTENCY
        self.rock_ema_span = 75 # Increased from 50 for more stable price basis
        self.rock_volatility_window = 39 # Increased from 20 for more reliable volatility measurement
        self.rock_volatility_history = deque(maxlen=self.rock_volatility_window + 5) # Store recent mid-prices for volatility calc
        self.rock_price_history_ema = deque(maxlen=self.rock_ema_span + 10) # Separate deque for EMA calc
        self.current_rock_ema = None

        # Rock Market Making parameters - OPTIMIZED FOR CONSISTENCY
        self.rock_mm_spread_multiplier = 1.3  # Reduced from 1.5 for more balanced spreads
        self.rock_mm_min_spread = 3  # Increased from 2 for consistent protection
        self.rock_mm_base_order_size = 17  # Reduced from 12 for more conservative sizing
        self.rock_mm_inventory_skew_factor = 0.04  # Reduced from 0.05 for more balanced inventory management
        
        # Base IV tracking parameters - OPTIMIZED FOR CONSISTENCY
        self.base_iv_window = 30  # Increased from 20 for more stable IV signals
        self.base_iv_history = deque(maxlen=self.base_iv_window)  # Recent base IV values
        self.base_iv_ema = None  # EMA of base IV
        self.base_iv_ema_span = 20  # Increased from 10 for smoother trends
        self.iv_price_correlation_strength = 0.15  # Reduced from 0.4 for more conservative adjustments
        
        # Crash/spike protection parameters - OPTIMIZED FOR CONSISTENCY
        self.crash_protection_threshold = 2.0  # Increased from 2.5 for fewer false positives
        self.spike_protection_threshold = 3.0  # Increased from 2.5 for fewer false positives
        self.crash_protection_spread_multiplier = 1.7  # Reduced from 2.0 for more balanced protection
        self.crash_protection_size_reducer = 0.75  # Increased from 0.5 for less dramatic size reduction
        self.volatility_regime_window = 45  # Increased from 30 for more stable regime detection
        self.volatility_regime_threshold = 1.8  # Increased from 1.5 for more conservative regime switching
        
        # BS-based price adjustment parameters - OPTIMIZED FOR CONSISTENCY
        self.bs_adjustment_strength = 0.2  # Reduced from 0.3 for more conservative signals
        self.bs_moneyness_range = [-0.15, 0.0, 0.15]  # Narrower than [-0.2, 0.0, 0.2] for more reliable pricing
        
        # Mean reversion parameters - OPTIMIZED FOR CONSISTENCY
        self.rock_mr_std_dev_threshold = 2.25  # Increased from 1.5 for more reliable signals
        self.rock_mr_base_order_size = 12  # Reduced from 15 for more consistent sizing
        self.rock_mr_size_scaling_factor = 0.3  # Reduced from 0.5 for more consistent scaling

        # No persistent state needed in traderData


    # --- Helper Functions (unchanged) ---
    def _calculate_tte(self, timestamp: int) -> float:
        timesteps_per_day = 1_000_000
        days_total = 5
        days_passed = timestamp / float(timesteps_per_day)
        time_left_days = max(0.0, days_total - days_passed)
        return max(1e-9, time_left_days / 365.0)

    def _calculate_mt(self, strike: float, S: float, T: float) -> Optional[float]:
        if T <= 1e-9 or S <= 1e-9 or strike <= 1e-9: return None
        try:
            sqrtT = math.sqrt(T)
            if sqrtT < 1e-6: return None
            ratio = strike / S
            if ratio <= 1e-9: return None
            return math.log(ratio) / sqrtT
        except (ValueError, ZeroDivisionError, OverflowError): return None

    def _calculate_theoretical_iv(self, moneyness_m: float) -> float:
        theoretical_iv = (self.iv_param_a * (moneyness_m**2) +
                          self.iv_param_b * moneyness_m +
                          self.iv_param_c)
        return max(self.min_theoretical_iv, theoretical_iv)

    def _get_best_bid(self, symbol: Symbol, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders.keys()) if order_depth and order_depth.buy_orders else None
    def _get_best_ask(self, symbol: Symbol, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders.keys()) if order_depth and order_depth.sell_orders else None
    def _get_mid_price(self, symbol: Symbol, order_depth: OrderDepth) -> Optional[float]:
        best_bid = self._get_best_bid(symbol, order_depth)
        best_ask = self._get_best_ask(symbol, order_depth)
        if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
        if best_bid is not None: return float(best_bid)
        if best_ask is not None: return float(best_ask)
        return None

    def _calculate_rock_ema(self, current_rock_price: Optional[float]):
        """Updates EMA state"""
        if current_rock_price is not None:
            self.rock_price_history_ema.append(current_rock_price)
        if len(self.rock_price_history_ema) < self.rock_ema_span // 2:
            self.current_rock_ema = None
        else:
            prices = list(self.rock_price_history_ema)
            if self.current_rock_ema is None: self.current_rock_ema = np.mean(prices)
            else:
                 alpha = 2.0 / (self.rock_ema_span + 1.0)
                 price_to_use = current_rock_price if current_rock_price is not None else self.current_rock_ema
                 self.current_rock_ema = (price_to_use * alpha) + (self.current_rock_ema * (1.0 - alpha))
        return self.current_rock_ema

    def _calculate_rock_volatility(self, current_rock_price: Optional[float]) -> float:
        """Calculates recent price standard deviation as a volatility measure."""
        if current_rock_price is not None:
             self.rock_volatility_history.append(current_rock_price)
        if len(self.rock_volatility_history) < self.rock_volatility_window // 2:
             return 0.0
        recent_prices = list(self.rock_volatility_history)
        std_dev = np.std(recent_prices)
        return std_dev
    
    # --- Base IV and BS-related calculations (unchanged logic) ---
    def _estimate_base_iv(self) -> float:
        """Calculate base IV (at-the-money) from the volatility smile formula."""
        # When moneyness (m_t) = 0, the formula simplifies to just the constant term
        base_iv = self.iv_param_c
        return base_iv
    
    def _update_base_iv_history(self, current_base_iv: float):
        """Updates base IV history and EMA"""
        self.base_iv_history.append(current_base_iv)
        
        if len(self.base_iv_history) < self.base_iv_ema_span // 2:
            self.base_iv_ema = current_base_iv
        else:
            if self.base_iv_ema is None:
                self.base_iv_ema = np.mean(list(self.base_iv_history))
            else:
                alpha = 2.0 / (self.base_iv_ema_span + 1.0)
                self.base_iv_ema = (current_base_iv * alpha) + (self.base_iv_ema * (1.0 - alpha))
        
        return self.base_iv_ema
    
    def _detect_volatility_regime(self, recent_vol: float) -> bool:
        """Detects if we're in a high volatility regime"""
        if len(self.rock_volatility_history) < self.volatility_regime_window // 2:
            return False
            
        vol_history = list(self.rock_volatility_history)
        vol_mean = np.mean(vol_history)
        vol_std = np.std(vol_history)
        
        if vol_std > 1e-6:
            vol_z_score = (recent_vol - vol_mean) / vol_std
            return vol_z_score > self.volatility_regime_threshold
        return False
    
    def _detect_crash_or_spike(self, St: float, current_rock_ema: float, recent_vol: float) -> Tuple[bool, bool]:
        """Detects potential crash or spike conditions"""
        if recent_vol <= 1e-6 or St is None or current_rock_ema is None:
            return False, False
            
        # Calculate z-score of current price
        price_deviation = St - current_rock_ema
        price_z_score = price_deviation / recent_vol
        
        # Detect crash (price significantly below EMA)
        crash_detected = price_z_score < -self.crash_protection_threshold
        
        # Detect spike (price significantly above EMA)
        spike_detected = price_z_score > self.spike_protection_threshold
        
        return crash_detected, spike_detected
        
    def _calculate_bs_based_adjustment(self, St: float, TTE: float) -> float:
        """
        Uses BS pricing across different moneyness levels to estimate price trend
        Returns an adjustment factor (-1 to +1) to apply to fair value
        """
        if St is None or St <= 0 or TTE <= 1e-9:
            return 0.0
            
        # Calculate BS prices at different moneyness levels
        adjustment = 0.0
        valid_points = 0
        
        for m in self.bs_moneyness_range:
            # Convert moneyness to strike price
            K = St * math.exp(m * math.sqrt(TTE))
            
            # Calculate IV using volatility smile
            sigma = self._calculate_theoretical_iv(m)
            
            # Calculate BS price
            bs_price = black_scholes_call_price_approx(St, K, TTE, sigma)
            
            if bs_price is not None:
                # Calculate delta to understand directional exposure
                delta = black_scholes_delta_call_approx(St, K, TTE, sigma)
                
                if delta is not None:
                    # Delta ranges from 0 to 1, remap to -1 to +1 for symmetric impact
                    delta_centered = (delta - 0.5) * 2
                    
                    # Weight contribution by position in moneyness_range
                    # Center points count more
                    weight = 1.0 - abs(m) / 0.5  # Higher weight for central points
                    
                    adjustment += delta_centered * weight
                    valid_points += 1
        
        if valid_points > 0:
            adjustment /= valid_points
            
        # Limit adjustment to -1.0 to 1.0 range
        return max(-1.0, min(1.0, adjustment)) * self.bs_adjustment_strength

    # --- Order Generation Functions ---
    # UNCHANGED: Voucher Orders Function (as per instructions)
    def _generate_voucher_orders(self, state: TradingState, St: float, TTE: float, rock_price_deviation: Optional[float]) -> Tuple[Dict[Symbol, List[Order]], float]:
        """
        Generates orders for vouchers using mean reversion around the theoretical
        Black-Scholes price, incorporating volatility smile and adjustments.
        """
        voucher_orders: Dict[Symbol, List[Order]] = {}
        net_delta_exposure = 0.0

        if St is None or St <= 0 or TTE <= 1e-9:
            return voucher_orders, net_delta_exposure

        # --- Calculate Adjustment Factors for BS strategy ---
        buy_threshold_adjust = 1.0
        sell_threshold_adjust = 1.0
        size_adjustment_factor = 1.0
        iv_adjustment_factor = 1.0 # Factor to adjust theoretical IV

        if rock_price_deviation is not None:
            # Adjust thresholds, size, and IV based on underlying deviation
            threshold_impact = rock_price_deviation * self.voucher_aggressiveness * 0.5
            buy_threshold_adjust = max(0.5, min(1.5, 1.0 + threshold_impact))
            sell_threshold_adjust = max(0.5, min(1.5, 1.0 - threshold_impact))
            size_adjustment_factor = max(0.5, min(1.5, 1.0 - rock_price_deviation * self.voucher_aggressiveness))
            iv_adjustment = 1.0 - rock_price_deviation * self.iv_adjustment_sensitivity
            iv_adjustment_factor = max(0.7, min(1.3, iv_adjustment))

        for symbol in self.voucher_symbols:
            orders_for_symbol: List[Order] = []
            K = self.voucher_strikes[symbol]
            moneyness_m = self._calculate_mt(K, St, TTE)
            if moneyness_m is None:
                continue

            # --- Black-Scholes Calculation (This is the 'Mean' for reversion) ---
            sigma_theo = self._calculate_theoretical_iv(moneyness_m) # Uses vol smile
            sigma_adjusted = sigma_theo * iv_adjustment_factor # Apply adjustment
            bs_theoretical_price = black_scholes_call_price_approx(St, K, TTE, sigma_adjusted, self.risk_free_rate)
            if bs_theoretical_price is None:
                continue

            # --- Get Market Data ---
            order_depth = state.order_depths.get(symbol, OrderDepth())
            best_bid = self._get_best_bid(symbol, order_depth)
            best_ask = self._get_best_ask(symbol, order_depth)
            current_position = state.position.get(symbol, 0)
            position_limit = self.position_limits[symbol]

            # --- Calculate Effective BS Price Thresholds (Entry points for mean reversion) ---
            effective_buy_threshold_pct = self.voucher_trade_threshold_base_pct * buy_threshold_adjust
            effective_sell_threshold_pct = self.voucher_trade_threshold_base_pct * sell_threshold_adjust
            # Buy if market ask is below this price (BS Theo - % Threshold)
            bs_buy_entry_price = bs_theoretical_price * (1.0 - effective_buy_threshold_pct)
            # Sell if market bid is above this price (BS Theo + % Threshold)
            bs_sell_entry_price = bs_theoretical_price * (1.0 + effective_sell_threshold_pct)

            log_entry = {
                 "event": "VOUCHER_EVAL", "symbol": symbol, "St": round(St), "K": K, "TTE": round(TTE, 5),
                 "m": round(moneyness_m, 3), "sigma_adj": round(sigma_adjusted, 4),
                 "bs_theo_px": round(bs_theoretical_price, 2), # This is the mean
                 "bid": best_bid, "ask": best_ask,
                 "bs_buy_entry": round(bs_buy_entry_price, 2), # Buy threshold
                 "bs_sell_entry": round(bs_sell_entry_price, 2), # Sell threshold
                 "current_pos": current_position
            }
            # logger.print(log_entry) # Optional: Reduce logging verbosity

            # --- Trading Logic ---
            # BS Mean Reversion Buy Logic
            if best_ask is not None and best_ask < bs_buy_entry_price:
                mispricing = bs_theoretical_price - best_ask # How far below the mean is the ask?
                # Scale size based on magnitude of mispricing relative to threshold width
                threshold_width = bs_theoretical_price * effective_buy_threshold_pct + 1e-9
                size_factor = max(0.1, min(2.0, (mispricing / threshold_width)))
                buy_size_modifier = size_adjustment_factor # Adjust based on rock price deviation
                quantity = math.floor(self.voucher_order_size * buy_size_modifier * size_factor)
                quantity = max(1, quantity)
                available_buy_limit = position_limit - current_position
                order_qty = min(quantity, available_buy_limit)
                vol_at_ask = abs(order_depth.sell_orders.get(best_ask, 0))
                order_qty = min(order_qty, vol_at_ask)
                if order_qty > 0:
                    orders_for_symbol.append(Order(symbol, best_ask, order_qty))
                    logger.print({"event":"VOUCHER_ORDER_GEN", "reason": "BS_MR_BUY", "symbol":symbol, "price": best_ask, "qty": order_qty, "bs_theo": round(bs_theoretical_price,2), "misprice": round(mispricing,2)})

            # BS Mean Reversion Sell Logic
            elif best_bid is not None and best_bid > bs_sell_entry_price: # Use elif to avoid placing buy and sell in same iteration
                mispricing = best_bid - bs_theoretical_price # How far above the mean is the bid?
                # Scale size based on magnitude of mispricing relative to threshold width
                threshold_width = bs_theoretical_price * effective_sell_threshold_pct + 1e-9
                size_factor = max(0.1, min(2.0, (mispricing / threshold_width)))
                sell_size_modifier = (1.0 / size_adjustment_factor) if size_adjustment_factor > 1e-6 else 1.0 # Inverse adjustment for selling
                quantity = math.floor(self.voucher_order_size * sell_size_modifier * size_factor)
                quantity = max(1, quantity)
                available_sell_limit = position_limit + current_position
                order_qty = min(quantity, available_sell_limit)
                vol_at_bid = abs(order_depth.buy_orders.get(best_bid, 0))
                order_qty = min(order_qty, vol_at_bid)
                if order_qty > 0:
                   orders_for_symbol.append(Order(symbol, best_bid, -order_qty))
                   logger.print({"event":"VOUCHER_ORDER_GEN", "reason": "BS_MR_SELL", "symbol":symbol, "price": best_bid, "qty": -order_qty, "bs_theo": round(bs_theoretical_price,2), "misprice": round(mispricing,2)})

            if orders_for_symbol:
                voucher_orders[symbol] = orders_for_symbol
                # Delta calculation remains the same, based on final position and adjusted sigma
                final_position = current_position
                for order in orders_for_symbol: final_position += order.quantity
                delta = black_scholes_delta_call_approx(St, K, TTE, sigma_adjusted, self.risk_free_rate)
                if delta is not None:
                    position_delta = final_position * delta
                    net_delta_exposure += position_delta

        return voucher_orders, net_delta_exposure

    # Rock Market Making (unchanged logic, only parameters optimized)
    def _generate_rock_market_making_orders(self, state: TradingState, St: float, current_rock_ema: Optional[float], TTE: float) -> List[Order]:
        """Generates market making orders for VOLCANIC_ROCK with enhanced strategies."""
        mm_orders: List[Order] = []
        symbol = Product.VOLCANIC_ROCK

        if St is None:
            if current_rock_ema is None:
                logger.print({"event":"ROCK_MM_SKIP", "reason":"Missing St and EMA"})
                return mm_orders
            else:
                fair_value = current_rock_ema
                logger.print({"event":"ROCK_MM_WARN", "reason":"Using EMA as fair value"})
        else:
            # Use EMA as the fair value anchor for MM
            fair_value = current_rock_ema if current_rock_ema is not None else St
        
        # Original fair value before adjustments
        original_fair_value = fair_value
        
        # Calculate volatility for dynamic spread
        recent_vol = self._calculate_rock_volatility(St)
        
        # --- Base IV Analysis ---
        # Get current base IV and update history
        current_base_iv = self._estimate_base_iv()
        base_iv_ema = self._update_base_iv_history(current_base_iv)
        
        # Detect IV trend (is IV decreasing, which is often bullish for price?)
        iv_adjustment = 0.0
        if base_iv_ema is not None and len(self.base_iv_history) >= self.base_iv_window // 2:
            iv_deviation = current_base_iv - base_iv_ema
            if abs(base_iv_ema) > 1e-6:
                iv_pct_change = iv_deviation / base_iv_ema
                # Negative correlation: IV down → Price up and vice versa
                iv_signal = -iv_pct_change
                iv_adjustment = fair_value * iv_signal * self.iv_price_correlation_strength
                
                logger.print({
                    "event": "IV_ANALYSIS", 
                    "base_iv": round(current_base_iv, 4), 
                    "iv_ema": round(base_iv_ema, 4), 
                    "iv_pct_chg": round(iv_pct_change, 4),
                    "iv_signal": round(iv_signal, 4),
                    "adjustment": round(iv_adjustment, 2)
                })
                
                # Apply adjustment to fair value
                fair_value += iv_adjustment
        
        # --- BS-based Price Adjustment ---
        bs_adjustment = self._calculate_bs_based_adjustment(St, TTE)
        bs_adjustment_amount = fair_value * bs_adjustment * 0.01  # Max 1% adjustment
        fair_value += bs_adjustment_amount
        
        logger.print({
            "event": "BS_ADJUSTMENT",
            "signal": round(bs_adjustment, 4),
            "adjustment": round(bs_adjustment_amount, 2)
        })
        
        # --- Crash/Spike Detection ---
        crash_detected, spike_detected = False, False
        if St is not None and current_rock_ema is not None and recent_vol > 1e-6:
            crash_detected, spike_detected = self._detect_crash_or_spike(St, current_rock_ema, recent_vol)
            
        # --- Volatility Regime Detection ---
        high_vol_regime = self._detect_volatility_regime(recent_vol)
        
        # --- Dynamic Spread and Size Calculation ---
        # Base spread calculation
        spread = max(self.rock_mm_min_spread, recent_vol * self.rock_mm_spread_multiplier)
        
        # Adjust spread for crash/spike protection
        if crash_detected or spike_detected:
            spread *= self.crash_protection_spread_multiplier
            logger.print({
                "event": "CRASH_SPIKE_DETECTED",
                "crash": crash_detected,
                "spike": spike_detected,
                "spread_multiplier": self.crash_protection_spread_multiplier
            })
        
        # Adjust spread for volatility regime
        if high_vol_regime:
            spread *= 1.3  # Increase spread in high volatility regimes
            logger.print({
                "event": "HIGH_VOL_REGIME",
                "spread_multiplier": 1.3
            })
        
        # Calculate bid and ask prices
        our_bid = math.floor(fair_value - spread / 2.0)
        our_ask = math.ceil(fair_value + spread / 2.0)
        
        # Get market data
        order_depth = state.order_depths.get(symbol, OrderDepth())
        current_position = state.position.get(symbol, 0)
        position_limit = self.LIMIT[symbol]
        
        # --- Inventory Management ---
        max_pos = position_limit
        min_pos = -position_limit
        inventory_pct = (current_position - min_pos) / (max_pos - min_pos) if (max_pos - min_pos) > 0 else 0.5
        
        # Skew prices based on inventory (higher bid if inventory low, higher ask if inventory high)
        inventory_skew = (0.5 - inventory_pct) * spread * 0.2
        our_bid += math.floor(inventory_skew)
        our_ask += math.floor(inventory_skew)
        
        # Base order size calculation
        base_size = self.rock_mm_base_order_size
        
        # Reduce size during crash/spike
        if crash_detected or spike_detected:
            base_size *= self.crash_protection_size_reducer
        
        # Reduce size in high volatility regimes
        if high_vol_regime:
            base_size *= 0.8
        
        # Skew size based on inventory
        buy_size_skew = max(0.5, min(1.5, 1.0 - (inventory_pct - 0.5) * 2 * self.rock_mm_inventory_skew_factor))
        sell_size_skew = max(0.5, min(1.5, 1.0 + (inventory_pct - 0.5) * 2 * self.rock_mm_inventory_skew_factor))
        
        # Special case handling for crash/spike
        if crash_detected:
            # During crash: Reduce buy size, increase sell size (to capture recovery)
            buy_size_skew *= 0.7
            # Only buy below EMA minus threshold
            if St is not None and current_rock_ema is not None:
                crash_buy_threshold = current_rock_ema - recent_vol * 1.0
                our_bid = min(our_bid, math.floor(crash_buy_threshold))
            
        if spike_detected:
            # During spike: Reduce sell size, increase buy size (to capture reversion)
            sell_size_skew *= 0.7
            # Only sell above EMA plus threshold
            if St is not None and current_rock_ema is not None:
                spike_sell_threshold = current_rock_ema + recent_vol * 1.0
                our_ask = max(our_ask, math.ceil(spike_sell_threshold))
        
        # Final size calculations
        buy_order_size = max(1, int(round(base_size * buy_size_skew)))
        sell_order_size = max(1, int(round(base_size * sell_size_skew)))
        
        # --- Place Orders ---
        # Buy Order
        available_buy_limit = position_limit - current_position
        final_buy_qty = min(buy_order_size, available_buy_limit)
        if final_buy_qty > 0:
            mm_orders.append(Order(symbol, our_bid, final_buy_qty))
        
        # Sell Order
        available_sell_limit = position_limit + current_position
        final_sell_qty = min(sell_order_size, available_sell_limit)
        if final_sell_qty > 0:
            mm_orders.append(Order(symbol, our_ask, -final_sell_qty))
        
        if mm_orders:
            logger.print({
                "event": "ROCK_MM_ORDERS",
                "orig_fv": round(original_fair_value, 1),
                "adj_fv": round(fair_value, 1),
                "iv_adj": round(iv_adjustment, 2),
                "bs_adj": round(bs_adjustment_amount, 2),
                "bid": our_bid,
                "ask": our_ask,
                "spread": round(spread, 1),
                "buy_sz": final_buy_qty,
                "sell_sz": -final_sell_qty,
                "inventory": current_position,
                "inv_pct": round(inventory_pct, 2),
                "crash": crash_detected,
                "spike": spike_detected,
                "high_vol": high_vol_regime
            })
        
        return mm_orders







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

    def VOLCANIC_ROCK_price(self, state):
        depth = state.order_depths["VOLCANIC_ROCK"]
        if not depth.sell_orders or not depth.buy_orders:
            return 0
        buy = max(list(depth.buy_orders.keys()))
        sell = min(list(depth.sell_orders.keys()))
        if (buy == 0 or sell == 0):
            return 0
        return (buy + sell) / 2
    
    def update_signal(self, state: TradingState, traderObject, product) -> None:
        
        if not state.order_depths[product].sell_orders or not state.order_depths[product].buy_orders:
            return None
        
        order_depth = state.order_depths[product]
        sell_vol = sum(abs(qty) for qty in order_depth.sell_orders.values())
        buy_vol = sum(abs(qty) for qty in order_depth.buy_orders.values())
        sell_money = sum(price * abs(qty) for price, qty in order_depth.sell_orders.items())
        buy_money = sum(price * abs(qty) for price, qty in order_depth.buy_orders.items())
        if sell_vol == 0 or buy_vol == 0:
            return None
        fair_value = (sell_money + buy_money) / (sell_vol + buy_vol)

        vwap = fair_value
        last_prices = traderObject.get(f"{product}_last_prices", [])
        last_prices.append(vwap)
        
        if len(last_prices) > self.params[product]["ma_length"]:
            last_prices.pop(0)
        
        traderObject[f"{product}_last_prices"] = last_prices

        if len(last_prices) < self.params[product]["ma_length"]:
            return None
        
        long_ma = np.mean(last_prices)
        sd = np.std(last_prices)
        if sd == 0:
            return None
        zscore = (vwap - long_ma) / sd
        sma_short = last_prices[-self.params[product]["short_ma_length"] :]
        sma_diffed = np.diff(sma_short, n=1)

        buy_signal = zscore < self.params[product]["open_threshold"] and sma_diffed[-1] > 0 and sma_diffed[-2] > 0 and sma_short[-1] > sma_short[-2] and sma_diffed[-1] > sma_diffed[-2]
        sell_signal = zscore > self.params[product]["close_threshold"] and sma_diffed[-1] < 0 and sma_diffed[-2] > 0 and sma_short[-1] < sma_short[-2] and sma_diffed[-1] < sma_diffed[-2]

        extreme_buy_signal = zscore < -4 or fair_value < 20
        buy_signal |= extreme_buy_signal
        extreme_sell_signal = zscore > 4
        sell_signal |= extreme_sell_signal

        neutral_signal = abs(zscore) < 0

        if buy_signal:
            self.signal[product] = 1
        elif sell_signal:
            self.signal[product] = -1


        if extreme_sell_signal:
            self.signal[product] = -2
        if extreme_buy_signal:
            self.signal[product] = 2

    
        
    def spam_orders(self, state : TradingState, product, signal_product):

        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders

        if not buy_orders or not sell_orders:
            return []
        
        
        orders = []
        pos = state.position.get(product, 0)

        if self.signal[signal_product] == 2:
            # take all sell orders
            orderdepth = state.order_depths[product]
            for price, qty in orderdepth.sell_orders.items():
                if pos + abs(qty) > self.LIMIT[product]:
                    break
                orders.append(Order(product, price, abs(qty)))
                pos += abs(qty)
            rem_buy = self.LIMIT[product] - pos
            best_buy = max(orderdepth.buy_orders.keys())
            orders.append(Order(product, best_buy + 1, rem_buy))
            return orders
        
        elif self.signal[signal_product] == -2:
            # take all buy orders
            orderdepth = state.order_depths[product]
            for price, qty in orderdepth.buy_orders.items():
                if pos - abs(qty) < -self.LIMIT[product]:
                    break
                orders.append(Order(product, price, -abs(qty)))
                pos -= abs(qty)
            rem_sell = self.LIMIT[product] + pos
            best_sell = min(orderdepth.sell_orders.keys())
            orders.append(Order(product, best_sell - 1, -rem_sell))
            return orders


        if self.signal[signal_product] > 0:
            rem_buy = self.LIMIT[product] - pos
            orderdepth = state.order_depths[product]
            # add our own buy order at best_buy + 1
            best_buy = max(orderdepth.buy_orders.keys())
            orders.append(Order(product, best_buy + 1, rem_buy))
        
        elif self.signal[signal_product] < 0:
            rem_sell = self.LIMIT[product] + pos
            orderdepth = state.order_depths[product]
            # add our own sell order at best_sell - 1
            best_sell = min(orderdepth.sell_orders.keys())
            orders.append(Order(product, best_sell - 1, -rem_sell))
        
        elif self.signal[signal_product] == 0:
            best_buy = max(state.order_depths[product].buy_orders.keys())
            best_sell = min(state.order_depths[product].sell_orders.keys())

            if pos > 0:
                # close buy position
                orders.append(Order(product, best_buy + 1, -pos))
            elif pos < 0:
                # close sell position
                orders.append(Order(product, best_sell - 1, -pos))
        
        return orders
    
    def mean_reversion_orders(
        self, product: str, order_depth: OrderDepth, position: int, trader_object: dict
    ) -> List[Order]:
        orders = []
        mid_price = self.get_swmid(order_depth)
        if mid_price is None:
            return orders
        history = trader_object.setdefault("mid_history", {}).setdefault(product, [])
        history.append(mid_price)

        if len(history) > 100:
            history.pop(0)
        if len(history) > 40:
            mean = np.mean(history)
            std = np.std(history)
            zscore = (mid_price - mean) / std if std > 0 else (mid_price - mean)

        if len(history) <= 40: zscore = 0

        POSITION_LIMIT = 400
        amount = 400
        if zscore > z_score_threshold and position > -POSITION_LIMIT:
            best_bid = max(order_depth.buy_orders.keys())
            quantity = -min(amount, POSITION_LIMIT + position)
            orders.append(Order(product, best_bid, quantity))

        elif zscore < z_score_threshold and position < POSITION_LIMIT:
            best_ask = min(order_depth.sell_orders.keys())
            quantity = min(amount, POSITION_LIMIT - position)
            orders.append(Order(product, best_ask, quantity))

        if abs(zscore) < z_score_take_profit and position != 0:
            if position > 0:
                best_bid = max(order_depth.buy_orders.keys())
                orders.append(Order(product, best_bid, -position))
            elif position < 0:
                best_ask = min(order_depth.sell_orders.keys())
                orders.append(Order(product, best_ask, -position))
        return orders


    def go_fully_long(
        self,
        product: str,
        order_depth: OrderDepth,
        limit_price: int | float,
        position: int,
        position_limit: int,
    ) -> tuple[list[Order], int, int]:
        to_buy = position_limit - position
        market_sell_orders = sorted(order_depth.sell_orders.items())

        own_orders = []
        buy_order_volume = 0

        max_buy_price = limit_price

        for price, volume in market_sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                own_orders.append(Order(product, price, quantity))
                to_buy -= quantity
                order_depth.sell_orders[price] += quantity
                if order_depth.sell_orders[price] == 0:
                    order_depth.sell_orders.pop(price)
                buy_order_volume += quantity

        # make a market order to attempt to fill the rest
        if to_buy > 0:
            own_orders.append(Order(product, price, to_buy))

        return own_orders

    def go_fully_short(
        self,
        product: str,
        order_depth: OrderDepth,
        limit_price: int | float,
        position: int,
        position_limit: int,
    ) -> tuple[list[Order], int, int]:
        to_sell = position - -position_limit

        market_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        own_orders = []
        sell_order_volume = 0

        min_sell_price = limit_price

        for price, volume in market_buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                own_orders.append(Order(product, price, -quantity))
                to_sell -= quantity
                order_depth.buy_orders[price] -= quantity
                if order_depth.buy_orders[price] == 0:
                    order_depth.buy_orders.pop(price)
                sell_order_volume += quantity

        if to_sell > 0:
            own_orders.append(Order(product, price, -to_sell))

        return own_orders
    def olivia_trades_goated(self, state: TradingState, result: dict, trader_data: dict):
        products = [Product.CROISSANTS]

        for product in products:
            if product not in state.order_depths:
                continue

            od = deepcopy(state.order_depths[product])
            prod_orders = []
            pb1_orders = []
            pb2_orders = []

            pos = state.position.get(product, 0)
            # print("Product:", product)
            # print("Position:", pos)

            sell_time_diff = 0 
            buy_time_diff = 0
            last_buy_price = 0
            last_sell_price = 0

            if product == Product.CROISSANTS:
                sell_time_diff = state.timestamp - trader_data.get("last_olivia_croissant_sell_trade", -1000)
                buy_time_diff = state.timestamp -  trader_data.get("last_olivia_croissant_buy_trade", -1000)
                last_buy_price = trader_data.get("last_olivia_croissant_buy_price", 0)
                last_sell_price = trader_data.get("last_olivia_croissant_sell_price", 0)

                last_buy_price_pb2 = trader_data.get("last_olivia_pb2_buy_price", 0)
                last_sell_price_pb2 = trader_data.get("last_olivia_pb2_sell_price", 0)

                if Product.PICNIC_BASKET2 in state.order_depths:
                    od_pb2 = deepcopy(state.order_depths[Product.PICNIC_BASKET2])
                    pos_pb2 = state.position.get(Product.PICNIC_BASKET2, 0)

                    if buy_time_diff < 1000:
                        price_pb2 = last_buy_price_pb2 * 1.5
                        pb2_orders += self.go_fully_long(Product.PICNIC_BASKET2, od_pb2, price_pb2, pos_pb2, self.LIMIT[Product.PICNIC_BASKET2])
                    elif sell_time_diff < 1000:
                        price_pb2 = last_sell_price_pb2 * 0.5
                        pb2_orders += self.go_fully_short(Product.PICNIC_BASKET2, od_pb2, price_pb2, pos_pb2, self.LIMIT[Product.PICNIC_BASKET2])
                    
                    # result[Product.PICNIC_BASKET2] = pb2_orders
            
            # if product == Product.SQUID_INK:
            #     sell_time_diff = state.timestamp - trader_data.get("last_olivia_ink_sell_trade", -1000)
            #     buy_time_diff = state.timestamp -  trader_data.get("last_olivia_ink_buy_trade", -1000)
            #     last_buy_price = trader_data.get("last_olivia_ink_buy_price", 0)
            #     last_sell_price = trader_data.get("last_olivia_ink_sell_price", 0)


            if buy_time_diff < 500:
                logger.print("HI")
                price = last_buy_price * 1.1
                prod_orders += self.go_fully_long(product, od, price, pos, self.LIMIT[product])
                # limit buy
            
            elif sell_time_diff < 500:
                logger.print("HEY")
                price = last_sell_price * 0.9
                prod_orders += self.go_fully_short(product, od, price, pos, self.LIMIT[product])
                
            result[product] = result.get(product, []) + prod_orders

    


    def run(self, state: TradingState):
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {}

        convs = 0
        
        # macaron‑specific rolling lists
        prev_mac_prices: List[float] = traderObject.get("prev_mac_prices", [])
        self.sunlight_history        = traderObject.get("sun_hist",  [])
        self.last_sunlight           = traderObject.get("last_sun", 0.0)

        # ---------- rebuild MarketData snapshot --------------------------- #
        md = MarketData()
        for prod in self.limits:
            pos  = state.position.get(prod, 0)
            od   = state.order_depths.get(prod)
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

        self.time = state.timestamp
        conversions = 0
        product = Product.VOLCANIC_ROCK

        self.update_signal(state, traderObject, product)

        delta = 0
        
        price = self.VOLCANIC_ROCK_price(state)
        
        if price == 0:
            return {}, 0, ""

        if price >= 10500 - delta:
            for option in Product.options:
                result[option] = self.spam_orders(state, option, product)
        elif price >= 10250 - delta:
            for option in [Product.VOLCANIC_ROCK_VOUCHER_10250, 
                Product.VOLCANIC_ROCK_VOUCHER_9750, Product.VOLCANIC_ROCK_VOUCHER_9500]:
                result[option] = self.spam_orders(state, option, product)
            for option in [Product.VOLCANIC_ROCK_VOUCHER_10500]:
                self.update_signal(state, traderObject, option)
                result[option] = self.spam_orders(state, option, option)
        elif price >= 10000 - delta:
            for option in [Product.VOLCANIC_ROCK_VOUCHER_9750, Product.VOLCANIC_ROCK_VOUCHER_9500]:
                result[option] = self.spam_orders(state, option, product)
            for option in [Product.VOLCANIC_ROCK_VOUCHER_10250, Product.VOLCANIC_ROCK_VOUCHER_10500]:
                self.update_signal(state, traderObject, option)
                result[option] = self.spam_orders(state, option, option)
        elif price >= 9750 - delta:
            for option in [Product.VOLCANIC_ROCK_VOUCHER_9750, Product.VOLCANIC_ROCK_VOUCHER_9500]:
                result[option] = self.spam_orders(state, option, product)
            for option in [Product.VOLCANIC_ROCK_VOUCHER_10250, Product.VOLCANIC_ROCK_VOUCHER_10500]:
                self.update_signal(state, traderObject, option)
                result[option] = self.spam_orders(state, option, option)
        elif price >= 9500 - delta:
            for option in [Product.VOLCANIC_ROCK_VOUCHER_9500]:
                result[option] = self.spam_orders(state, option, product)
            for option in [Product.VOLCANIC_ROCK_VOUCHER_10250, Product.VOLCANIC_ROCK_VOUCHER_10500, Product.VOLCANIC_ROCK_VOUCHER_9750]:
                self.update_signal(state, traderObject, option)
                result[option] = self.spam_orders(state, option, option)
        else:
            for option in [Product.VOLCANIC_ROCK_VOUCHER_10250, Product.VOLCANIC_ROCK_VOUCHER_10500, Product.VOLCANIC_ROCK_VOUCHER_9750, Product.VOLCANIC_ROCK_VOUCHER_9500]:
                self.update_signal(state, traderObject, option)
                result[option] = self.spam_orders(state, option, option)

        for strategy in self.voucher_strategies:
            orders, _, _ = strategy.run(state)
            for symbol, order_list in orders.items():
                result[symbol] = order_list

        if Product.PICNIC_BASKET2 in state.order_depths:
            o2, _ = self.b2.process(state)
            if o2:
                result[Product.PICNIC_BASKET2] = o2

        # integrate Basket 1
        b1_orders, _, _ = self.b1_mgr.run(state)
        for sym, lst in b1_orders.items():
            result.setdefault(sym, []).extend(lst)

         # ---------- bespoke single‑symbol strategies --------------------- #
        for sym, strat in self.special_strategies.items():
            strat.restore(traderObject.get(sym))
            if sym in state.order_depths:
                odrs, _ = strat.process(state)
                if odrs:
                    result[sym] = odrs
            traderObject[sym] = strat.preserve()

        # if Product.VOLCANIC_ROCK in state.order_depths:
        #     vr_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        #     vr_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
        #     vr_orders = self.mean_reversion_orders(
        #         Product.VOLCANIC_ROCK,
        #         vr_order_depth,
        #         vr_position,
        #         traderObject
        #     )
        #     result[Product.VOLCANIC_ROCK] = vr_orders

        prev_mac_prices.append(md.fair["MAGNIFICENT_MACARONS"])
        if len(prev_mac_prices) > 13:
            prev_mac_prices.pop(0)
        self.recent_std = float(np.std(prev_mac_prices))

        mac_take, mac_convs = self.take_macaron(state, md)
        mac_make = mac_clear = []
        if self.recent_std < 53:
            mac_make  = self.make_macaron(state, md)
            mac_clear = self.clear_macaron(state, md)

        mac_orders = mac_take + mac_make + mac_clear
        if mac_orders:
            result["MAGNIFICENT_MACARONS"] = mac_orders
        convs += mac_convs



        trades = state.market_trades.get(Product.CROISSANTS, [])
        for t in trades:
            if t.buyer == "Olivia":
                traderObject["last_olivia_croissant_buy_trade"] = t.timestamp
                traderObject["last_olivia_croissant_buy_price"] = t.price

                od = deepcopy(state.order_depths[Product.PICNIC_BASKET1])
                if od:
                    best_bid = max(od.buy_orders.keys())
                    best_ask = min(od.sell_orders.keys())
                    mid = (best_bid + best_ask) / 2
                    traderObject["last_olivia_pb2_buy_price"] = mid

            elif t.seller == "Olivia":
                traderObject["last_olivia_croissant_sell_trade"] = t.timestamp
                traderObject["last_olivia_croissant_sell_price"] = t.price

                od = deepcopy(state.order_depths[Product.PICNIC_BASKET2])
                if od:
                    best_bid = max(od.buy_orders.keys())
                    best_ask = min(od.sell_orders.keys())
                    mid = (best_bid + best_ask) / 2
                    traderObject["last_olivia_pb2_sell_price"] = mid
 
        self.olivia_trades_goated(state, result, traderObject)



        # --- Calculations ---
        timestamp = state.timestamp
        TTE = self._calculate_tte(timestamp)
        rock_order_depth = state.order_depths.get(Product.VOLCANIC_ROCK, OrderDepth())
        St = self._get_mid_price(Product.VOLCANIC_ROCK, rock_order_depth)
        current_rock_ema = self._calculate_rock_ema(St) # Also updates self.rock_price_history_ema

        # --- Determine Rock Price Deviation (for voucher BS strategy adjustments) ---
        rock_price_deviation_pct = None # Relative deviation
        if St is not None and self.current_rock_ema is not None and abs(self.current_rock_ema) > 1e-6:
            rock_price_deviation_pct = (St - self.current_rock_ema) / self.current_rock_ema

        # --- Generate Voucher Orders (UNCHANGED BS Mean Reversion Strategy) ---
        #voucher_orders_dict, _net_delta_exposure = self._generate_voucher_orders(state, St, TTE, rock_price_deviation_pct)
        #result.update(voucher_orders_dict)

        # --- Generate Volcanic Rock Orders (NEW Enhanced Market Making Strategy) ---
        rock_orders_list = self._generate_rock_market_making_orders(state, St, current_rock_ema, TTE)
        if rock_orders_list:
            result[Product.VOLCANIC_ROCK] = rock_orders_list




        # ---------- persist everything for next tick --------------------- #
        traderObject.update(
            prev_mac_prices=prev_mac_prices,
            sun_hist=self.sunlight_history,
            last_sun=self.last_sunlight,
        )


        traderData = jsonpickle.encode(traderObject)


        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData