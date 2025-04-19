###############################################################################
# macaron_trader.py  –  standalone bot for MAGNIFICENT_MACARONS only
###############################################################################
from collections import deque
from typing import Dict, List, Tuple, Any
import json
import jsonpickle
import math

from datamodel import (          # provided by the IMC‑Prosperity framework
    Order,
    OrderDepth,
    TradingState,
    ConversionObservation,
    Symbol,
)

# --------------------------------------------------------------------------- #
#                           0.  Product constants                             #
# --------------------------------------------------------------------------- #
PRODUCT: Symbol = "MAGNIFICENT_MACARONS"

POS_LIMIT        = 75
CONVERSION_CL    = 10          # max units per conversion request

STORAGE_COST     = 0.1
TRANSPORT_FEES   = 3
IMPORT_TARIFF    = 1
EXPORT_TARIFF    = 1

CSI              = 40          # Critical Sunlight Index
BASE_FAIR        = 200
PREMIUM_SLOPE    = 0.6         # fundamental premium per index point < CSI

# execution parameters
EDGE_TAKE        = 0.35        # required edge to lift/hit
MIN_REST_QTY     = 5           # resting order size baseline


###############################################################################
#                       1.  Adaptive single‑symbol logic                      #
###############################################################################
class MacaronLogic:
    """Adaptive, volatility‑aware model for MAGNIFICENT_MACARONS."""

    HISTORY   = 30
    EMA_ALPHA = 0.25

    def __init__(self, product: Symbol, limit: int) -> None:
        self.product = product
        self.limit   = limit
        self.mid_hist: deque[float] = deque(maxlen=self.HISTORY)
        self.ema_mid: float | None  = None

    # --------------------- helpers --------------------------------------- #
    @staticmethod
    def _csi_premium(sun_idx: float) -> float:
        if sun_idx < CSI:
            gap = CSI - sun_idx
            # slight non‑linear kick as sunlight collapses
            return PREMIUM_SLOPE * gap * (1 + 0.02 * gap)
        return 0.0

    def _update_mid(self, mid: float) -> None:
        self.mid_hist.append(mid)
        if self.ema_mid is None:
            self.ema_mid = mid
        else:
            self.ema_mid = (
                self.EMA_ALPHA * mid + (1 - self.EMA_ALPHA) * self.ema_mid
            )

    # --------------------- main entry ------------------------------------ #
        ########################################################################
    #                    MacaronLogic.process  (revised)                   #
    ########################################################################
    def process(self, snap: TradingState) -> Tuple[List[Order], int]:
        """Return (orders, convert_qty).  NOTE: orders will always be empty
        because MACARONS are only tradable via conversion."""
        orders: List[Order] = []           # kept for interface symmetry
        convert_qty: int = 0

        # ---------- 1.  Pull sunlight + update fair value  -------------- #
        obs = snap.observations
        if hasattr(obs, "sunlight"):
            sun_idx = obs.sunlight
        elif isinstance(obs, dict):
            sun_idx = obs.get("sunlight", 50)
        else:
            sun_idx = 50                                   # neutral default

        csi_prem  = self._csi_premium(sun_idx)
        fair      = BASE_FAIR + csi_prem                   # no EMA needed
        sigma     = 2.0                                    # rough proxy; tune

        # ---------- 2.  See if conversion quotes are available ---------- #
        conv_map = None
        if hasattr(snap, "conversion_observations"):
            conv_map = snap.conversion_observations
        elif hasattr(snap, "conversionObservations"):
            conv_map = snap.conversionObservations

        if not (conv_map and PRODUCT in conv_map):
            # Local back‑tester → conversions not supported.
            return orders, convert_qty                     # do nothing

        conv: ConversionObservation = conv_map[PRODUCT]
        pos  = snap.position.get(PRODUCT, 0)

        eff_ask = conv.askPrice + TRANSPORT_FEES + IMPORT_TARIFF
        eff_bid = conv.bidPrice - TRANSPORT_FEES - EXPORT_TARIFF

        # ---------- 3.  Decide whether to convert ----------------------- #
        buy_cap  = POS_LIMIT - pos
        sell_cap = POS_LIMIT + pos

        if (
            eff_ask + STORAGE_COST < fair - sigma
            and buy_cap > 0
        ):
            convert_qty = min(CONVERSION_CL, buy_cap)

        elif (
            eff_bid - STORAGE_COST > fair + sigma
            and sell_cap > 0
        ):
            convert_qty = -min(CONVERSION_CL, sell_cap)

        return orders, convert_qty

    # --------------------- persistence ---------------------------------- #
    def preserve(self) -> Dict[str, Any]:
        return {
            "hist": list(self.mid_hist),
            "ema": self.ema_mid,
        }

    def restore(self, blob: Dict[str, Any]) -> None:
        if not isinstance(blob, dict):
            return
        self.mid_hist = deque(blob.get("hist", []), maxlen=self.HISTORY)
        self.ema_mid  = blob.get("ema")


###############################################################################
#                               2.  Trader class                              #
###############################################################################
class Trader:
    """
    Exposed entry‑point for the IMC‑Prosperity engine.
    Returns (orders_dict, convertQty_int, traderData_str) each tick.
    """

    def __init__(self) -> None:
        self.strategies: Dict[Symbol, MacaronLogic] = {
            PRODUCT: MacaronLogic(PRODUCT, POS_LIMIT)
        }

    # --------------------------------------------------------------------- #
    def run(self, snap: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        final_orders: Dict[Symbol, List[Order]] = {}
        convert_sum: int = 0

        # 1.  Restore persisted state
        old_state: Dict[str, Any] = {}
        if snap.traderData:
            try:
                old_state = json.loads(snap.traderData)
            except Exception:
                pass

        for sym, strat in self.strategies.items():
            if sym in old_state:
                try:
                    strat.restore(jsonpickle.decode(old_state[sym]))
                except Exception:
                    pass

            if sym not in snap.order_depths:
                continue

            orders, conv = strat.process(snap)
            if orders:
                final_orders[sym] = orders
            convert_sum += conv

        # 2.  Preserve state for next tick
        new_state = {
            sym: jsonpickle.encode(strat.preserve(), unpicklable=False)
            for sym, strat in self.strategies.items()
        }
        trader_data = json.dumps(new_state, separators=(",", ":"))

        return final_orders, convert_sum, trader_data