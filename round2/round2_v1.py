# import jsonpickle
# import math
# from typing import List
# from datamodel import Order, TradingState, OrderDepth

# # Round 2 position limits for each product
# POSITION_LIMITS = {
#     "RAINFOREST_RESIN": 50,
#     "KELP": 50,
#     "SQUID_INK": 50,
#     "CROISSANTS": 250,
#     "JAMS": 350,
#     "DJEMBES": 60,
#     "PICNIC_BASKET1": 60,
#     "PICNIC_BASKET2": 100,
# }

# # Approximate fair values from analysis
# FAIR_VALUES = {
#     "RAINFOREST_RESIN": 10000,
#     "KELP": 2032,
#     "SQUID_INK": 2000,
#     "CROISSANTS": 4300,
#     "JAMS": 6500,
#     "DJEMBES": 13450,
#     "PICNIC_BASKET1": 59000,
#     "PICNIC_BASKET2": 30500,
# }

# # Basic strategy parameters
# RSI_PERIOD = 14        # for Squid Ink RSI
# SQUID_RSI_BUY = 30
# SQUID_RSI_SELL = 70
# SQUID_TRADE_SIZE = 5
# DEFAULT_TRADE_SIZE = 10
# ROLLING_WINDOW = 20     # how many mid-prices to keep for general reference

# class Trader:
#     def run(self, state: TradingState):
#         # Restore persistent data
#         try:
#             trader_state = jsonpickle.decode(state.traderData)
#             if not isinstance(trader_state, dict):
#                 trader_state = {}
#         except:
#             trader_state = {}

#         # Ensure we have a price list for each product
#         for prod in POSITION_LIMITS:
#             if prod not in trader_state:
#                 trader_state[prod] = {"prices": []}

#         result = {}

#         def update_price_history(prod: str, price: float):
#             trader_state[prod]["prices"].append(price)
#             if len(trader_state[prod]["prices"]) > 2 * ROLLING_WINDOW:
#                 trader_state[prod]["prices"] = trader_state[prod]["prices"][-2 * ROLLING_WINDOW:]

#         def compute_rsi(price_list: List[float], period:int=14):
#             if len(price_list) < period+1:
#                 return 50.0
#             recent = price_list[-(period+1):]
#             deltas = [recent[i+1]-recent[i] for i in range(len(recent)-1)]
#             gains = sum(d for d in deltas if d>0) / period
#             losses = -sum(d for d in deltas if d<0) / period
#             if losses==0:
#                 return 100.0
#             rs = gains/losses
#             return 100 - (100/(1+ rs))

#         for product, order_depth in state.order_depths.items():
#             orders: List[Order] = []
#             position = state.position.get(product, 0)
#             limit = POSITION_LIMITS[product]

#             # find best bid/ask => mid price
#             best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
#             best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
#             mid_price = None
#             if best_bid is not None and best_ask is not None:
#                 mid_price = (best_bid + best_ask) / 2
#             elif best_bid is not None:
#                 mid_price = best_bid
#             elif best_ask is not None:
#                 mid_price = best_ask

#             if mid_price is not None:
#                 update_price_history(product, mid_price)

#             # RAINFOREST_RESIN => stable at ~10k
#             if product == "RAINFOREST_RESIN":
#                 fv = FAIR_VALUES[product]
#                 if best_ask is not None and best_ask < fv:
#                     qty = min(limit - position, -order_depth.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_ask, qty))
#                 if best_bid is not None and best_bid> fv:
#                     qty = min(limit + position, order_depth.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_bid, -qty))

#             # KELP => slight up => anchored ~2032
#             elif product == "KELP":
#                 fv = FAIR_VALUES[product]
#                 if best_ask is not None and best_ask< fv:
#                     qty = min(limit - position, -order_depth.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_ask, qty))
#                 if best_bid is not None and best_bid> fv:
#                     qty = min(limit + position, order_depth.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_bid, -qty))

#             # SQUID_INK => RSI approach
#             elif product == "SQUID_INK":
#                 rsi_val = compute_rsi(trader_state[product]["prices"], RSI_PERIOD)
#                 if best_ask is not None and rsi_val< SQUID_RSI_BUY:
#                     qty = min(limit - position, -order_depth.sell_orders[best_ask], SQUID_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_ask, qty))
#                 if best_bid is not None and rsi_val> SQUID_RSI_SELL:
#                     qty = min(limit + position, order_depth.buy_orders[best_bid], SQUID_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_bid, -qty))

#             # For the newly introduced Round2 items => short reversion around approx FV
#             else:
#                 fv = FAIR_VALUES.get(product, mid_price if mid_price else 0)
#                 if fv == 0 and mid_price is not None:
#                     fv = mid_price
#                 # Buy if ask < fv
#                 if best_ask is not None and best_ask< fv:
#                     qty = min(limit - position, -order_depth.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_ask, qty))
#                 # Sell if bid> fv
#                 if best_bid is not None and best_bid> fv:
#                     qty = min(limit + position, order_depth.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
#                     if qty>0:
#                         orders.append(Order(product, best_bid, -qty))

#             if orders:
#                 print(f"[Round2OptimalTrader] {product} => {orders}")
#             result[product] = orders

#         traderData = jsonpickle.encode(trader_state)
#         conversions = 0
#         return result, conversions, traderData

import jsonpickle
import math
from typing import List
from datamodel import Order, TradingState, OrderDepth

# Round 2 position limits for each product
POSITION_LIMITS = {
    "DJEMBES": 60,
    "JAMS": 350,
}

# Approximate fair values from analysis
FAIR_VALUES = {
    "DJEMBES": 13450,
    "JAMS": 6500,
}

DEFAULT_TRADE_SIZE = 10
ROLLING_WINDOW = 20

class Trader:
    def run(self, state: TradingState):
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except:
            trader_state = {}

        for prod in POSITION_LIMITS:
            if prod not in trader_state:
                trader_state[prod] = {"prices": []}

        result = {}

        def update_price_history(prod: str, price: float):
            trader_state[prod]["prices"].append(price)
            if len(trader_state[prod]["prices"]) > 2 * ROLLING_WINDOW:
                trader_state[prod]["prices"] = trader_state[prod]["prices"][-2 * ROLLING_WINDOW:]

        for product, order_depth in state.order_depths.items():
            if product not in POSITION_LIMITS:
                continue  # Only trade DJEMBES and JAMS

            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            mid_price = None
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid_price = best_bid
            elif best_ask is not None:
                mid_price = best_ask

            if mid_price is not None:
                update_price_history(product, mid_price)

            fv = FAIR_VALUES.get(product, mid_price if mid_price else 0)
            if fv == 0 and mid_price is not None:
                fv = mid_price

            if best_ask is not None and best_ask < fv:
                qty = min(limit - position, -order_depth.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))

            if best_bid is not None and best_bid > fv:
                qty = min(limit + position, order_depth.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            if orders:
                print(f"[DjembeJamTrader] {product} => {orders}")
            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        return result, conversions, traderData
