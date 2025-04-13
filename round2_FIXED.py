from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import math

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
    # Product.SQUID_INK: {
    #     "take_width": 1,
    #     "clear_width": 0,
    #     "prevent_adverse": True,
    #     "adverse_volume": 25,
    #     "reversion_beta": -0.3,
    #     "disregard_edge": 1,
    #     "join_edge": 0,
    #     "default_edge": 1,
    # },
    Product.SQUID_INK: {
        "fair_value": 1850,
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
                        orders.append(Order(product, best_bid, -1 * quantity))
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
    # YOUR RSI LOGIC FOR SQUID INK REPLACES PARTNER LOGIC
    ###############################################################################

    

    def rsi_squid_ink_logic(self, state, trader_state) -> List[Order]:
        """Your RSI approach for SQUID_INK: 
        - Keep a rolling history of mid prices (up to 20).
        - Compute RSI over up to 14 periods.
        - If ask <1900 and RSI<30 => buy. If bid>2100 and RSI>70 => sell.
        """
        orders: List[Order] = []
        product = Product.SQUID_INK
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMIT[product]

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
            # Add to persistent price list
            if "prices" not in trader_state:
                trader_state["prices"] = []
            # We'll store it under trader_state["prices_SQUID"] if you prefer
            # but let's just keep it direct
            trader_state["prices"].append(mid_price)
            if len(trader_state["prices"])>20:
                trader_state["prices"] = trader_state["prices"][-20:]
            
            prices = trader_state["prices"]
            period = min(14, len(prices))
            if period>=2:
                deltas = [prices[i]- prices[i-1] for i in range(-period+1,0)]
                gains = [d for d in deltas if d>0]
                losses = [-d for d in deltas if d<0]
                avg_gain = sum(gains)/period if gains else 0
                avg_loss = sum(losses)/period if losses else 0
                if avg_loss==0:
                    rsi = 100
                else:
                    rs = avg_gain/ avg_loss
                    rsi = 100 - (100/(1+ rs))
            else:
                rsi=50
            
            print(f"[SQUID_INK] mid={mid_price:.2f}, RSI={rsi:.2f}")
            
            # If ask <1900 and RSI<30 => buy
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_qty = order_depth.sell_orders[best_ask]  # negative
                can_buy = limit - position
                if best_ask<1900 and rsi<30 and can_buy>0:
                    quantity = min(5, can_buy, -ask_qty)
                    if quantity>0:
                        orders.append(Order(product, best_ask, quantity))
            
            # If bid>2100 and RSI>70 => sell
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_qty = order_depth.buy_orders[best_bid]
                can_sell = limit + position
                if best_bid>2100 and rsi>70 and can_sell>0:
                    quantity = min(5, can_sell, bid_qty)
                    if quantity>0:
                        orders.append(Order(product, best_bid, -quantity))
        
        return orders
    
    def squid_ink_logic_algo2(self, state) -> List[Order]:
        product = Product.SQUID_INK
        od = state.order_depths[product]
        fair_value = self.params[product]["fair_value"]
        limit = self.LIMIT[product]
        position = state.position.get(product, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_bid_vol = od.buy_orders[best_bid] if best_bid else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_ask_vol = od.sell_orders[best_ask] if best_ask else 0

        orders: List[Order] = []

        if best_ask is not None and best_ask < fair_value:
            can_buy = limit - position
            if can_buy > 0:
                ask_qty = -best_ask_vol
                quantity = min(can_buy, ask_qty)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))

        if best_bid is not None and best_bid > fair_value:
            if position >= 0:
                can_sell = limit + position
            else:
                can_sell = limit - abs(position)
            if can_sell > 0:
                bid_qty = best_bid_vol
                quantity = min(can_sell, bid_qty)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))

        return orders
    ###############################################################################
    # MAIN RUN METHOD
    ###############################################################################
    def run(self, state):
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

            # Exactly the partner’s flow: take -> clear -> make
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

        # REPLACE PARTNER’S SQUID_INK LOGIC WITH YOUR RSI LOGIC
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = self.squid_ink_logic_algo2(state)

        conversions = 1  # Partner code uses 1
        traderData = jsonpickle.encode(trader_dict)
        return result, conversions, traderData
