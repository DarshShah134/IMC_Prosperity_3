import jsonpickle
import math
from typing import List
from datamodel import Order, TradingState

class Trader:
    def run(self, state: TradingState):
        # Print current persistent state and observations (for debugging)
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        # Recover persistent state from traderData.
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except Exception:
            trader_state = {}
        
        # Ensure persistent state for each product exists.
        for prod in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if prod not in trader_state:
                trader_state[prod] = {"prices": []}
        
        # Prepare orders dictionary.
        result = {}
        
        # Per-product position limits and fixed fair values for stable assets.
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        fair_values = {"RAINFOREST_RESIN": 10000, "KELP": 2000, "SQUID_INK": 2000}
        # Trade size limits: Squid Ink uses a lower size.
        trade_size_limit = {"RAINFOREST_RESIN": 10, "KELP": 10, "SQUID_INK": 5}
        
        # Process each product.
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            available_buy = max(0, position_limits[product] - current_position)
            available_sell = max(0, position_limits[product] + current_position)
            
            if product in ["RAINFOREST_RESIN", "KELP"]:
                # Use fixed fair value strategy for stable products.
                fair_value = fair_values[product]
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    ask_qty = order_depth.sell_orders[best_ask]  # Sell quantities are negative.
                    if best_ask < fair_value and available_buy > 0:
                        quantity = min(trade_size_limit[product], available_buy, -ask_qty)
                        if quantity > 0:
                            print(f"Placing BUY for {product}: {quantity} @ {best_ask} (fair: {fair_value})")
                            orders.append(Order(product, best_ask, quantity))
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    bid_qty = order_depth.buy_orders[best_bid]
                    if best_bid > fair_value and available_sell > 0:
                        quantity = min(trade_size_limit[product], available_sell, bid_qty)
                        if quantity > 0:
                            print(f"Placing SELL for {product}: {quantity} @ {best_bid} (fair: {fair_value})")
                            orders.append(Order(product, best_bid, -quantity))
            
            elif product == "SQUID_INK":
                # -------------------------------
                # New strategy for Squid Ink using fixed anchor and RSI filter.
                # -------------------------------
                # Compute mid price from available order book.
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
                    # Update Squid Ink mid price history (up to 20 entries).
                    trader_state[product]["prices"].append(mid_price)
                    if len(trader_state[product]["prices"]) > 20:
                        trader_state[product]["prices"] = trader_state[product]["prices"][-20:]
                    
                    prices = trader_state[product]["prices"]
                    # Compute RSI over a 14-period window if available; otherwise, over available prices.
                    period = min(14, len(prices))
                    if period >= 2:
                        # Compute gains and losses.
                        deltas = [prices[i] - prices[i-1] for i in range(-period+1, 0)]
                        gains = [delta for delta in deltas if delta > 0]
                        losses = [-delta for delta in deltas if delta < 0]
                        avg_gain = sum(gains) / period if gains else 0
                        avg_loss = sum(losses) / period if losses else 0
                        if avg_loss == 0:
                            rsi = 100
                        else:
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 50  # Neutral if not enough data.
                    
                    print(f"{product} - mid: {mid_price:.2f}, RSI: {rsi:.2f}")
                    
                    # Use fixed thresholds anchored around the fair value (2000):
                    buy_price_threshold = 1900  # If price is below 1900, consider as oversold.
                    sell_price_threshold = 2100 # If price is above 2100, consider as overbought.
                    
                    # Combine with RSI: Only buy if RSI < 30 (strongly oversold) AND price is below threshold.
                    if order_depth.sell_orders and available_buy > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        ask_qty = order_depth.sell_orders[best_ask]
                        if best_ask < buy_price_threshold and rsi < 30:
                            quantity = min(trade_size_limit[product], available_buy, -ask_qty)
                            if quantity > 0:
                                print(f"Placing BUY for {product}: {quantity} @ {best_ask} (price < {buy_price_threshold} and RSI {rsi:.2f} < 30)")
                                orders.append(Order(product, best_ask, quantity))
                    
                    # Only sell if RSI > 70 (strongly overbought) AND price exceeds threshold.
                    if order_depth.buy_orders and available_sell > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        bid_qty = order_depth.buy_orders[best_bid]
                        if best_bid > sell_price_threshold and rsi > 70:
                            quantity = min(trade_size_limit[product], available_sell, bid_qty)
                            if quantity > 0:
                                print(f"Placing SELL for {product}: {quantity} @ {best_bid} (price > {sell_price_threshold} and RSI {rsi:.2f} > 70)")
                                orders.append(Order(product, best_bid, -quantity))
                # End Squid Ink processing.
            
            # Save orders for this product.
            result[product] = orders
        
        # No conversion requests in round 1.
        conversions = 0
        
        # Update persistent state.
        traderData = jsonpickle.encode(trader_state)
        
        # Print the submission identifier for tracking.
        submission_id = "59f81e67-f6c6-4254-b61e-39661eac6141"
        print("Submission ID:", submission_id)
        
        return result, conversions, traderData
