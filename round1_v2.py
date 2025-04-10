import pandas as pd
import numpy as np
import math
import jsonpickle
import json
from statistics import mean, stdev
from typing import List, Dict, Tuple

# -------------------------
# Data loading & processing functions

def load_price_data(filename: str) -> pd.DataFrame:
    """Load price data CSV (semicolon-separated)."""
    df = pd.read_csv(filename, sep=";")
    df["timestamp"] = df["timestamp"].astype(int)
    # Compute mid_price if not provided
    if "mid_price" not in df.columns or df["mid_price"].isnull().all():
        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features and labels per product.
    
    For each product:
      - Compute a rolling mean and rolling standard deviation over a 5-tick window.
      - Compute a z-score: (mid_price - rolling_mean) / (rolling_std + epsilon).
      - Define a "future" mid_price 10 ticks ahead to compute the delta.
      - Create a label: -1 if delta < -0.5, 0 if between -0.5 and 0.5, 1 if delta > 0.5.
    """
    df = df.sort_values("timestamp")
    feature_dfs = []
    for product, group in df.groupby("product"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group["rolling_mean"] = group["mid_price"].rolling(window=5, min_periods=1).mean()
        group["rolling_std"] = group["mid_price"].rolling(window=5, min_periods=1).std().fillna(0.0)
        group["z_score"] = (group["mid_price"] - group["rolling_mean"]) / (group["rolling_std"] + 1e-6)
        horizon = 10  # ticks ahead
        group["future_mid"] = group["mid_price"].shift(-horizon)
        group["delta"] = group["future_mid"] - group["mid_price"]
        group["label"] = group["delta"].apply(lambda x: -1 if x < -0.5 else (1 if x > 0.5 else 0))
        feature_dfs.append(group)
    full_df = pd.concat(feature_dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["future_mid"])
    return full_df

def dummy_train_model(df: pd.DataFrame):
    """
    Dummy model training that returns a function acting as a classifier.
    
    The "model" here is a simple heuristic:
      - If z_score > 0.3, predict 1 (price will go up).
      - If z_score < -0.3, predict -1 (price will go down).
      - Otherwise, predict 0.
    """
    def model_predict(features: np.ndarray) -> int:
        # Expect features: [mid_price, rolling_mean, rolling_std, z_score]
        z = features[0, 3]
        if z > 0.3:
            return 1
        elif z < -0.3:
            return -1
        else:
            return 0
    # For demonstration, we print training set average label
    avg_label = mean(df["label"].tolist())
    print("Dummy model trained. Average label value: {:.2f}".format(avg_label))
    return model_predict

# -------------------------
# Trader class definition

class TradingState:
    """
    A simple TradingState object.
    
    Attributes:
      order_depths: dict of product -> {"buy_orders": {price: volume}, "sell_orders": {price: volume}}
      position: dict of product -> current net position
      timestamp: current tick timestamp (int)
      traderData: string for persistent state (JSON-encoded)
    """
    def __init__(self, order_depths: Dict, position: Dict, timestamp: int, traderData: str = ""):
        self.order_depths = order_depths
        self.position = position
        self.timestamp = timestamp
        self.traderData = traderData

class Order:
    """
    Order object with:
      - product: string
      - price: int or float
      - quantity: int (positive = buy, negative = sell)
    """
    def __init__(self, product: str, price: float, quantity: int):
        self.product = product
        self.price = price
        self.quantity = quantity
        
    def __repr__(self):
        return f"({self.product}, {self.price}, {self.quantity})"

class Trader:
    """
    Trader class that implements the run() method which is called with a TradingState.
    
    The persistent state is stored using jsonpickle, and the trader computes
    a simple EMA and volatility per product to update a z-score.
    """
    def __init__(self, model_predict):
        self.model_predict = model_predict  # our dummy model function
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # Parse persistent state via jsonpickle
        if state.traderData.strip():
            try:
                td = jsonpickle.decode(state.traderData)
            except Exception:
                td = {}
        else:
            td = {}
        
        result = {}
        for product, order_depth in state.order_depths.items():
            orders = []
            # Compute mid_price from best bid & ask
            if order_depth["buy_orders"] and order_depth["sell_orders"]:
                best_bid = max(order_depth["buy_orders"].keys())
                best_ask = min(order_depth["sell_orders"].keys())
                mid_price = (best_bid + best_ask) / 2
            else:
                mid_price = 0.0

            # Retrieve or initialize product-specific state
            if product in td:
                prod_state = td[product]
            else:
                prod_state = {"ema": mid_price, "vol": 1.0}
            
            w = 0.2  # EMA weight
            ema = w * mid_price + (1 - w) * prod_state["ema"]
            vol = w * abs(mid_price - ema) + (1 - w) * prod_state["vol"]
            z = (mid_price - ema) / (vol + 1e-6)

            # Update persistent state for product
            td[product] = {"ema": ema, "vol": vol}
            
            # Build feature vector: [mid_price, ema, vol, z]
            feat = np.array([[mid_price, ema, vol, z]])
            prediction = self.model_predict(feat)
            
            pos = state.position.get(product, 0)
            # Decision logic: for "SQUID_INK", use the model signal;
            # for "RAINFOREST_RESIN" and "KELP", post symmetric quotes.
            if product == "SQUID_INK":
                if prediction == 1 and pos < 50:
                    best_ask = min(order_depth["sell_orders"].keys())
                    qty = min(10, 50 - pos)
                    orders.append(Order(product, best_ask, qty))
                elif prediction == -1 and pos > -50:
                    best_bid = max(order_depth["buy_orders"].keys())
                    qty = min(10, pos + 50)
                    orders.append(Order(product, best_bid, -qty))
            else:
                # For simpler products, post quotes a fixed spread away from the mid_price.
                spread = 1
                orders.append(Order(product, mid_price - spread, 5))
                orders.append(Order(product, mid_price + spread, -5))
            
            result[product] = orders
        
        traderData_out = jsonpickle.encode(td)
        conversions = 0
        return result, conversions, traderData_out