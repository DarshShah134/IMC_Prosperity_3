# Updated statistical arbitrage strategy with dynamic premium detection,
# smarter hedging, position-aware sizing, and PB2-focused optimization.

from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import json

POSITION_LIMITS = {
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
}

HEDGE_RATIOS = {
    "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
    "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
}


class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        def get_best_bid_ask(od: OrderDepth):
            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_bid_qty = od.buy_orders[best_bid] if best_bid is not None else 0
            best_ask = min(od.sell_orders) if od.sell_orders else None
            best_ask_qty = od.sell_orders[best_ask] if best_ask is not None else 0
            return best_bid, best_bid_qty, best_ask, best_ask_qty

        def get_fair_value(basket: str):
            ratio = HEDGE_RATIOS[basket]
            fair = 0
            for prod, qty in ratio.items():
                od = state.order_depths.get(prod)
                if not od:
                    return None
                bid, _, ask, _ = get_best_bid_ask(od)
                if bid is None or ask is None:
                    return None
                mid = 0.5 * (bid + ask)
                fair += mid * qty
            return fair

        def can_trade(prod: str, qty: int):
            pos = state.position.get(prod, 0)
            limit = POSITION_LIMITS[prod]
            return abs(pos + qty) <= limit

        def place_order(symbol, price, qty):
            if symbol not in result:
                result[symbol] = []
            result[symbol].append(Order(symbol, price, qty))

        # Main loop: Evaluate each basket
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            od_basket = state.order_depths.get(basket)
            if not od_basket:
                continue

            best_bid, bid_qty, best_ask, ask_qty = get_best_bid_ask(od_basket)
            if best_bid is None or best_ask is None:
                continue

            fair = get_fair_value(basket)
            if fair is None:
                continue

            threshold = 0.0015 * fair  # dynamic premium threshold ~0.15%
            pos_basket = state.position.get(basket, 0)
            ratio = HEDGE_RATIOS[basket]

            # --- Opportunity: Buy basket, sell components ---
            if best_ask + threshold < fair and pos_basket < POSITION_LIMITS[basket]:
                max_lots = min(-ask_qty, POSITION_LIMITS[basket] - pos_basket, 3)
                for i in range(max_lots):
                    if not can_trade(basket, +1):
                        break
                    place_order(basket, best_ask, +1)
                    for prod, qty in ratio.items():
                        od = state.order_depths.get(prod)
                        bid, bid_qty, _, _ = get_best_bid_ask(od)
                        if can_trade(prod, -qty):
                            place_order(prod, bid, -qty)

            # --- Opportunity: Sell basket, buy components ---
            if best_bid - threshold > fair and pos_basket > -POSITION_LIMITS[basket]:
                max_lots = min(bid_qty, POSITION_LIMITS[basket] + pos_basket, 3)
                for i in range(max_lots):
                    if not can_trade(basket, -1):
                        break
                    place_order(basket, best_bid, -1)
                    for prod, qty in ratio.items():
                        od = state.order_depths.get(prod)
                        _, _, ask, ask_qty = get_best_bid_ask(od)
                        if can_trade(prod, +qty):
                            place_order(prod, ask, +qty)

        return result, conversions, json.dumps({})
