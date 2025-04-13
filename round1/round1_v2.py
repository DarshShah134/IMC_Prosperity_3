import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias, List
import jsonpickle
import math

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

###############################################################################
# EXACT Partner code for RAINFOREST_RESIN & KELP. NO CHANGES MADE.
###############################################################################

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 50,
    },
    Product.KELP: {
        "fair_value": 2026,
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 10,
        "reversion_beta": 0.0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 25,
        "reversion_beta": -0.3,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

# Fixed trade size cap (used by our arbitrage logic for SQUID_INK as well)
trade_size_limit = 10

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if (not prevent_adverse) or (abs(best_ask_amount) <= adverse_volume):
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if (not prevent_adverse) or (abs(best_bid_amount) <= adverse_volume):
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: float,
        ask: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    ###############################################################################
    # NEW SQUID_INK ARBITRAGE LOGIC
    ###############################################################################
    def arbitrage_squid_ink_logic(self, state: TradingState, trader_state: dict) -> List[Order]:
        """
        Arbitrage strategy for SQUID_INK:
         - Determine the best bid and ask from the order book.
         - Maintain a rolling history of the observed spread (best_bid - best_ask).
         - Compute the average spread over the last 20 observations.
         - If the current spread exceeds the average spread by a fixed margin,
           then place simultaneous BUY (at best ask) and SELL (at best bid) orders
           to capture the spread.
        """
        orders: List[Order] = []
        product = Product.SQUID_INK
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMIT[product]

        # Get best bid and best ask.
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
        else:
            return orders

        current_spread = best_bid - best_ask

        # Maintain a rolling history of spreads in trader_state.
        key = "spread_history"
        if key not in trader_state:
            trader_state[key] = []
        trader_state[key].append(current_spread)
        if len(trader_state[key]) > 20:
            trader_state[key] = trader_state[key][-20:]
        avg_spread = sum(trader_state[key]) / len(trader_state[key])

        # Define a margin threshold – arbitrage only when current spread is significantly higher.
        margin_threshold = 2.0  # You can adjust this as needed.

        print(f"[SQUID_INK Arbitrage] best_bid={best_bid}, best_ask={best_ask}, spread={current_spread:.2f}, avg_spread={avg_spread:.2f}")

        # Available capacity.
        available_buy = max(0, limit - position)
        available_sell = max(0, limit + position)

        if current_spread > avg_spread + margin_threshold:
            # Place a BUY order at best_ask and a SELL order at best_bid.
            quantity = min(trade_size_limit, available_buy, available_sell)
            if quantity > 0:
                print(f"[SQUID_INK Arbitrage] Placing BUY for {product}: {quantity} @ {best_ask}")
                orders.append(Order(product, best_ask, quantity))
                print(f"[SQUID_INK Arbitrage] Placing SELL for {product}: {quantity} @ {best_bid}")
                orders.append(Order(product, best_bid, -quantity))
        return orders

    ###############################################################################
    # MAIN RUN METHOD
    ###############################################################################
    def run(self, state: TradingState):
        # Retrieve persistent dict from state
        try:
            trader_dict = jsonpickle.decode(state.traderData)
            if not isinstance(trader_dict, dict):
                trader_dict = {}
        except:
            trader_dict = {}

        result = {}

        # Process RAINFOREST_RESIN EXACT PARTNER LOGIC
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            product = Product.RAINFOREST_RESIN
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            buy_vol = 0
            sell_vol = 0
            fair_value = self.params[product]["fair_value"]
            take_width = self.params[product]["take_width"]
            clear_width = self.params[product]["clear_width"]
            disregard_edge = self.params[product]["disregard_edge"]
            join_edge = self.params[product]["join_edge"]
            default_edge = self.params[product]["default_edge"]
            soft_position_limit = self.params[product]["soft_position_limit"]
            prevent_adverse = self.params[product].get("prevent_adverse", False)
            adverse_volume = self.params[product].get("adverse_volume", 0)

            resin_orders, buy_vol, sell_vol = self.take_orders(
                product, od, fair_value, take_width, position, prevent_adverse, adverse_volume
            )
            resin_clear_orders, buy_vol, sell_vol = self.clear_orders(
                product, od, fair_value, clear_width, position, buy_vol, sell_vol
            )
            resin_make_orders, _, _ = self.make_orders(
                product, od, fair_value, position,
                buy_vol, sell_vol,
                disregard_edge, join_edge, default_edge,
                True, soft_position_limit
            )
            result[product] = resin_orders + resin_clear_orders + resin_make_orders

        # Process KELP EXACT PARTNER LOGIC
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            product = Product.KELP
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            buy_vol = 0
            sell_vol = 0
            fair_value = self.params[product]["fair_value"]
            take_width = self.params[product]["take_width"]
            clear_width = self.params[product]["clear_width"]
            disregard_edge = self.params[product]["disregard_edge"]
            join_edge = self.params[product]["join_edge"]
            default_edge = self.params[product]["default_edge"]
            soft_position_limit = self.params[product].get("soft_position_limit", 0)
            prevent_adverse = self.params[product].get("prevent_adverse", False)
            adverse_volume = self.params[product].get("adverse_volume", 0)

            kelp_orders, buy_vol, sell_vol = self.take_orders(
                product, od, fair_value, take_width, position,
                prevent_adverse, adverse_volume
            )
            kelp_clear_orders, buy_vol, sell_vol = self.clear_orders(
                product, od, fair_value, clear_width, position, buy_vol, sell_vol
            )
            kelp_make_orders, _, _ = self.make_orders(
                product, od, fair_value, position,
                buy_vol, sell_vol,
                disregard_edge, join_edge, default_edge,
                False, soft_position_limit
            )
            result[product] = kelp_orders + kelp_clear_orders + kelp_make_orders

        # Use our new arbitrage-based logic for SQUID_INK.
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_orders = self.arbitrage_squid_ink_logic(state, trader_dict)
            result[Product.SQUID_INK] = squid_orders

        conversions = 1  # EXACT PARTNER LOGIC
        traderData = jsonpickle.encode(trader_dict)
        return result, conversions, traderData