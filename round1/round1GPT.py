import jsonpickle
import math
from typing import List
from datamodel import Order, TradingState

class Trader:
    def run(self, state: TradingState):
        # Print current traderData and observations (for debug/logging)
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Initialize persistent trader state from traderData.
        # We'll store a history of prices for each product to help calculate moving averages.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception as e:
            trader_state = {}
        
        # For our three products, ensure a structure exists for historical prices.
        for prod in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if prod not in trader_state:
                trader_state[prod] = {"prices": []}

        # Prepare the result orders dictionary.
        result = {}

        # Define per-product position limits.
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        
        # Set fixed fair values for stable products based on broadcast info.
        fair_values = {"RAINFOREST_RESIN": 10000, "KELP": 2000}
        
        # Use a fixed trade size cap to avoid huge orders.
        trade_size_limit = 10

        # Process each product in the order depth.
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            # Current position for product (default to 0 if not tracked).
            current_position = state.position.get(product, 0)
            # For buy orders, available capacity: cannot exceed +limit.
            available_buy = max(0, position_limits[product] - current_position)
            # For sell orders, available capacity: cannot exceed -limit.
            available_sell = max(0, position_limits[product] + current_position)

            # For products with fixed fair value (Rainforest Resin and Kelp):
            if product in ["RAINFOREST_RESIN", "KELP"]:
                fair_value = fair_values[product]
                # Check sell orders (bots' ask prices) to see if there is an opportunity to buy
                if order_depth.sell_orders:
                    # Find the best (lowest) ask price.
                    best_ask = min(order_depth.sell_orders.keys())
                    ask_qty = order_depth.sell_orders[best_ask]  # sell orders quantities are negative.
                    if best_ask < fair_value and available_buy > 0:
                        # Quantity to buy: limited by trade_size_limit, available buy capacity, and the quantity offered.
                        quantity = min(trade_size_limit, available_buy, -ask_qty)
                        if quantity > 0:
                            print(f"Placing BUY for {product}: {quantity} @ {best_ask} (fair_value: {fair_value})")
                            orders.append(Order(product, best_ask, quantity))
                
                # Check buy orders (bots' bid prices) to see if there is an opportunity to sell.
                if order_depth.buy_orders:
                    # Find the best (highest) bid price.
                    best_bid = max(order_depth.buy_orders.keys())
                    bid_qty = order_depth.buy_orders[best_bid]
                    if best_bid > fair_value and available_sell > 0:
                        quantity = min(trade_size_limit, available_sell, bid_qty)
                        if quantity > 0:
                            print(f"Placing SELL for {product}: {quantity} @ {best_bid} (fair_value: {fair_value})")
                            # Sell orders are represented with a negative quantity.
                            orders.append(Order(product, best_bid, -quantity))

            # For Squid Ink, we use a dynamic moving average strategy.
            elif product == "SQUID_INK":
                mid_price = None
                # Calculate a mid price from the order book if possible.
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2.0
                elif order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    mid_price = best_bid
                elif order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = best_ask

                if mid_price is not None:
                    # Update historical prices for SQUID_INK.
                    trader_state[product]["prices"].append(mid_price)
                    # Limit history to the 20 most recent values.
                    if len(trader_state[product]["prices"]) > 20:
                        trader_state[product]["prices"] = trader_state[product]["prices"][-20:]
                    
                    prices = trader_state[product]["prices"]
                    avg_price = sum(prices) / len(prices)
                    if len(prices) > 1:
                        std_dev = math.sqrt(sum((p - avg_price) ** 2 for p in prices) / len(prices))
                    else:
                        std_dev = 0
                    threshold = max(1, 0.5 * std_dev)
                    
                    print(f"{product} - mid_price: {mid_price}, avg: {avg_price:.2f}, std: {std_dev:.2f}, threshold: {threshold:.2f}")
                    
                    # If the market is undervalued, we consider buying:
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        ask_qty = order_depth.sell_orders[best_ask]
                        if best_ask < (avg_price - threshold) and available_buy > 0:
                            quantity = min(trade_size_limit, available_buy, -ask_qty)
                            if quantity > 0:
                                print(f"Placing BUY for {product}: {quantity} @ {best_ask} (avg: {avg_price:.2f} - threshold: {threshold:.2f})")
                                orders.append(Order(product, best_ask, quantity))
                    
                    # If the market is overvalued, we consider selling:
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        bid_qty = order_depth.buy_orders[best_bid]
                        if best_bid > (avg_price + threshold) and available_sell > 0:
                            quantity = min(trade_size_limit, available_sell, bid_qty)
                            if quantity > 0:
                                print(f"Placing SELL for {product}: {quantity} @ {best_bid} (avg: {avg_price:.2f} + threshold: {threshold:.2f})")
                                orders.append(Order(product, best_bid, -quantity))
            
            # Assign any generated orders for this product.
            result[product] = orders

        # In Round 1 we are not requesting any conversions.
        conversions = 0

        # Save updated persistent state back to traderData.
        traderData = jsonpickle.encode(trader_state)
        
        # Include our submission identifier for reference.
        submission_id = "59f81e67-f6c6-4254-b61e-39661eac6141"
        print("Submission ID:", submission_id)
        
        return result, conversions, traderData
