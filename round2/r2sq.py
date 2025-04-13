# squid_ink_logic.py
from typing import List
from datamodel import Order, TradingState
import jsonpickle

# We define the product constant here.
SQUID_INK = "SQUID_INK"

def rsi_squid_ink_logic(state: TradingState, trader_state: dict, params: dict, limit: dict) -> List[Order]:
    """
    RSI-based SQUID_INK logic.
    - Maintains a rolling price history (up to 20 data points) in trader_state.
    - Computes RSI over up to 14 periods.
    - If best ask < 1900 and RSI < 30 → buy; if best bid > 2100 and RSI > 70 → sell.
    """
    orders: List[Order] = []
    product = SQUID_INK
    order_depth = state.order_depths[product]
    position = state.position.get(product, 0)
    pos_limit = limit[product]

    # Evaluate mid price
    mid_price = None
    if order_depth.buy_orders and order_depth.sell_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0
    elif order_depth.buy_orders:
        mid_price = max(order_depth.buy_orders.keys())
    elif order_depth.sell_orders:
        mid_price = min(order_depth.sell_orders.keys())

    if mid_price is not None:
        # Update rolling price history
        if "prices" not in trader_state:
            trader_state["prices"] = []
        trader_state["prices"].append(mid_price)
        if len(trader_state["prices"]) > 20:
            trader_state["prices"] = trader_state["prices"][-20:]
        
        prices = trader_state["prices"]
        period = min(14, len(prices))
        if period >= 2:
            deltas = [prices[i] - prices[i - 1] for i in range(-period + 1, 0)]
            gains = [d for d in deltas if d > 0]
            losses = [-d for d in deltas if d < 0]
            avg_gain = sum(gains) / period if gains else 0
            avg_loss = sum(losses) / period if losses else 0
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50

        print(f"[SQUID_INK] mid={mid_price:.2f}, RSI={rsi:.2f}")

        # If ask < 1900 and RSI < 30, then buy
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            ask_qty = order_depth.sell_orders[best_ask]  # note: negative volume
            can_buy = pos_limit - position
            if best_ask < 1900 and rsi < 30 and can_buy > 0:
                quantity = min(5, can_buy, -ask_qty)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))

        # If bid > 2100 and RSI > 70, then sell
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            bid_qty = order_depth.buy_orders[best_bid]
            can_sell = pos_limit + position
            if best_bid > 2100 and rsi > 70 and can_sell > 0:
                quantity = min(5, can_sell, bid_qty)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))

    return orders

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = {
                SQUID_INK: {
                    "fair_value": 1850,
                }
            }
        self.params = params
        self.LIMIT = {
            SQUID_INK: 50
        }

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = ""

        # Initialize or load trader state
        try:
            trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}
        except:
            trader_state = {}

        if SQUID_INK in state.order_depths:
            result[SQUID_INK] = rsi_squid_ink_logic(state, trader_state, self.params, self.LIMIT)

        # Save trader state
        traderData = jsonpickle.encode(trader_state)
        return result, conversions, traderData