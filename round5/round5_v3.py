from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Any, Dict, Tuple, TypeAlias
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

# =============================================================================
# Alternate Helpers – compact logger (unchanged)
# =============================================================================
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

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



class Trader:
    
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
    


    def run(self, state: TradingState):
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {}

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

        traderData = jsonpickle.encode(traderObject)


        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData