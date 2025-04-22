import json, math, statistics
from collections import deque
from typing import Dict, List, Tuple, Any, TypeAlias

import jsonpickle
import numpy as np

# ------- PLATFORM IMPORTS --------------------------------------------------
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

# =============================================================================
#  Helpers – compact logger (unchanged from original codebase)
# =============================================================================
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class ActivityRecorder:
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

    # ---------- utils ----------------------------------------------------
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

activity_logger = ActivityRecorder()

# =============================================================================
#  Core abstraction
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
#  HamperDealApproach – Basket 2 (unchanged)
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
#  BasketSpreadManager – original implementation (Basket 1)
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

    # ------------ helpers --------------------------------------------------
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

    def _convert_orders(self, art_orders: List[Order], ods: dict, picnic1: bool = True):
        # We intentionally drop component legs to honour "baskets only" requirement
        return {}

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
            pic_orders = [Order(basket, p_ask, vol)]
            return {basket: pic_orders}
        if not pic_od.buy_orders or not art_od.sell_orders:
            return None
        p_bid = max(pic_od.buy_orders)
        p_vol = pic_od.buy_orders[p_bid]
        a_ask = min(art_od.sell_orders)
        a_vol = abs(art_od.sell_orders[a_ask])
        vol = min(p_vol, a_vol, diff)
        pic_orders = [Order(basket, p_bid, -vol)]
        return {basket: pic_orders}

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

    # ------------------------------------------------------------------
    def run(self, st: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        pos = st.position.get(self.Product.PICNIC_BASKET1, 0)
        odict = self._spread_orders(st.order_depths, pos)
        return (odict or {}), 0, jsonpickle.encode(self.state)

# =============================================================================
#  Trader – only two baskets reach the exchange
# =============================================================================
class Trader:
    limits = {"PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100}

    def __init__(self):
        self.b2 = HamperDealApproach("PICNIC_BASKET2", self.limits["PICNIC_BASKET2"])
        self.b1_mgr = BasketSpreadManager()

    def run(self, snap: TradingState):
        out: Dict[str, List[Order]] = {}

        # Basket 2
        if "PICNIC_BASKET2" in snap.order_depths:
            self.b2.restore(None)
            o2, _ = self.b2.process(snap)
            if o2:
                out["PICNIC_BASKET2"] = o2

        # Basket 1
        b_orders, _, _ = self.b1_mgr.run(snap)
        for s, lst in b_orders.items():
            out.setdefault(s, []).extend(lst)

        # final filter to baskets only
        out = {s: lst for s, lst in out.items() if s in ("PICNIC_BASKET1", "PICNIC_BASKET2") and lst}

        activity_logger.finalize(snap, out, 0, "{}")
        return out, 0, "{}"
