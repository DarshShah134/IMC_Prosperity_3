from typing import Dict, List, Any
import math
import statistics
import json
import jsonpickle
import numpy as np  # Added import since np is used in squid_ink_orders
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# --------------------
# Logger Class (unchanged)
# --------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


# --------------------
# Trader Class
# --------------------
class Trader:
    def __init__(self):
        # KELP tracking
        self.kelp_prices = []
        self.kelp_vwap = []
        
        # SQUID_INK tracking
        self.squid_ink_prices = []       # Store recent mid-prices
        self.squid_ink_vwap = []         # Not used in the new strategy but kept for record
        self.squid_ink_volatility = []   # For any future use
        self.squid_ink_last_signal = None
        self.squid_ink_entry_timestamp = None  # Timestamp when SQUID_INK position was entered
        self.current_timestamp = None          # Current market timestamp
        
        # RAINFOREST_RESIN values
        self.rainforest_resin_fair_value = 10000

    # --------------------
    # RAINFOREST_RESIN Strategy (unchanged)
    # --------------------
    def clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int, 
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def rainforest_resin_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        product = "RAINFOREST_RESIN"

        baaf = min(
            [p for p in order_depth.sell_orders if p > self.rainforest_resin_fair_value + 1],
            default=self.rainforest_resin_fair_value + 2,
        )
        bbbf = max(
            [p for p in order_depth.buy_orders if p < self.rainforest_resin_fair_value - 1],
            default=self.rainforest_resin_fair_value - 2,
        )

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < self.rainforest_resin_fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > self.rainforest_resin_fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, product,
            buy_order_volume, sell_order_volume, self.rainforest_resin_fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, baaf - 1, -sell_quantity))

        return orders

    # --------------------
    # KELP Strategy (unchanged)
    # --------------------
    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif method == "mid_price_with_vol_filter":
            filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= min_vol]
            filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= min_vol]
            
            best_ask = min(filtered_ask) if filtered_ask else min(order_depth.sell_orders.keys())
            best_bid = max(filtered_bid) if filtered_bid else max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float,
                    take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            # Filter for orders with volume >= 15
            filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
            filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]

            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            self.kelp_prices.append(mmmid_price)

            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-order_depth.sell_orders[best_ask]) + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})

            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)

            fair_value = sum(x["vwap"] * x["vol"] for x in self.kelp_vwap) / sum(x["vol"] for x in self.kelp_vwap)
            fair_value = mmmid_price  # Override with mm mid price

            # Take best bid/ask if profitable
            if best_ask <= fair_value - take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", int(best_ask), quantity))
                        buy_order_volume += quantity

            if best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", int(best_bid), -quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP",
                buy_order_volume, sell_order_volume, fair_value, 2
            )

            aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
            bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", int(bbbf + 1), buy_quantity))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", int(baaf - 1), -sell_quantity))

        return orders

    def squid_ink_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if not best_ask or not best_bid:
            return orders

        # 核心参数配置
        window_size = 8        # 更敏感的短期窗口
        entry_z = 1.8          # 降低入场阈值
        exit_z = 0.5           # 提前止盈
        max_risk = 0.02        # 最大仓位风险比例
        volatility_window = 20 # 波动率计算窗口

        # 计算市场深度流动性
        ask_liq = sum(abs(v) for v in order_depth.sell_orders.values())
        bid_liq = sum(abs(v) for v in order_depth.buy_orders.values())

        # 生成价格序列
        current_mid = (best_ask + best_bid) / 2
        self.squid_ink_prices.append(current_mid)

        # 保持价格序列长度
        if len(self.squid_ink_prices) > volatility_window:
            self.squid_ink_prices.pop(0)

        # 计算波动率过滤条件
        returns = [
            self.squid_ink_prices[i] / self.squid_ink_prices[i - 1] - 1
            for i in range(1, len(self.squid_ink_prices))
        ]
        volatility = np.std(returns) if returns else 0

        # 仅在有足够数据时交易
        if len(self.squid_ink_prices) < window_size + 2:
            return orders

        # 计算双重移动平均
        fast_ma = np.mean(self.squid_ink_prices[-window_size:])
        slow_ma = np.mean(self.squid_ink_prices[-2 * window_size:])
        ma_diff = fast_ma - slow_ma

        # 计算价格通道
        high_channel = max(self.squid_ink_prices[-window_size:])
        low_channel = min(self.squid_ink_prices[-window_size:])

        # 生成交易信号
        long_condition = (
            (current_mid < low_channel) and  # 价格突破通道下轨
            (ma_diff > 0) and                 # 短期趋势向上
            (volatility > 0.002) and          # 波动率过滤
            (bid_liq > ask_liq)         # 买单流动性充足
        )

        short_condition = (
            (current_mid > high_channel) and  # 价格突破通道上轨
            (ma_diff < 0) and                 # 短期趋势向下
            (volatility > 0.002) and
            (ask_liq > bid_liq)
        )

        # 动态仓位管理
        max_position = int(position_limit * max_risk * (1 - volatility * 100))
        max_position = max(max_position, 5)  # 保持最小交易单位

        # 止盈止损逻辑
        if position != 0:
            # Use hasattr to check for a previously set squid_ink_entry_index
            entry_price = (
                self.squid_ink_prices[self.squid_ink_entry_index]
                if hasattr(self, "squid_ink_entry_index") and self.squid_ink_entry_index is not None
                else current_mid
            )
            unrealized_pnl = (current_mid - entry_price) * position

            # 动态止盈
            profit_target = entry_price * (1 + 0.002 * np.sign(position))
            if (position > 0 and current_mid >= profit_target) or (position < 0 and current_mid <= profit_target):
                orders.append(Order("SQUID_INK", int(profit_target), -position))
                return orders

            # 波动止损
            stop_loss = entry_price * (1 - 0.005 * np.sign(position))
            if (position > 0 and current_mid <= stop_loss) or (position < 0 and current_mid >= stop_loss):
                orders.append(Order("SQUID_INK", best_bid if position > 0 else best_ask, -position))
                return orders

        # 生成交易指令
        if long_condition and position <= max_position:
            available_qty = min(position_limit - position, max_position)
            bid_price = best_bid + 1  # 激进挂单
            orders.append(Order("SQUID_INK", bid_price, available_qty))
            self.squid_ink_entry_index = len(self.squid_ink_prices) - 1

        elif short_condition and position >= -max_position:
            available_qty = min(position_limit + position, max_position)
            ask_price = best_ask - 1  # 激进挂单
            orders.append(Order("SQUID_INK", ask_price, -available_qty))
            self.squid_ink_entry_index = len(self.squid_ink_prices) - 1

        # 均值回归平仓
        elif abs(position) > 0 and abs(fast_ma - current_mid) < exit_z:
            orders.append(Order("SQUID_INK", best_bid if position > 0 else best_ask, -position))

        return orders

    # --------------------
    # Run Method: Executes Strategies for All Instruments
    # --------------------
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # Configuration parameters (adjust as needed)
        rainforest_resin_position_limit = 50
        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10
        squid_ink_position_limit = 50

        # Update the current timestamp for SQUID_INK strategy.
        self.current_timestamp = state.timestamp

        # RAINFOREST_RESIN Trading
        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_resin_position = state.position.get("RAINFOREST_RESIN", 0)
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"],
                rainforest_resin_position,
                rainforest_resin_position_limit
            )
            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        # KELP Trading
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                kelp_timespan,
                kelp_make_width,
                kelp_take_width,
                kelp_position,
                kelp_position_limit
            )
            result["KELP"] = kelp_orders

        # SQUID_INK Trading (using updated swing-based strategy)
        if "SQUID_INK" in state.order_depths:
            squid_ink_position = state.position.get("SQUID_INK", 0)
            squid_ink_orders = self.squid_ink_orders(
                state.order_depths["SQUID_INK"],
                squid_ink_position,
                squid_ink_position_limit
            )
            result["SQUID_INK"] = squid_ink_orders

        trader_data = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "squid_ink_prices": self.squid_ink_prices,
            "squid_ink_vwap": self.squid_ink_vwap,
            "squid_ink_volatility": self.squid_ink_volatility,
            "squid_ink_last_signal": self.squid_ink_last_signal,
            "squid_ink_entry_timestamp": self.squid_ink_entry_timestamp,
            "current_timestamp": self.current_timestamp
        })

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
