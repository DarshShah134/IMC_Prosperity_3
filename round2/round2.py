from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import math
import json
import statistics  # for stdev in the new basket1 snippet

###############################################################################
#                          MERGED CODE STARTS HERE                            #
###############################################################################

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"

    # Picnic logic references these three:
    CROISSANTS = "CROISSANTS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# Weights for Basket1 (from snippet)
PB1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS:       3,
    Product.DJEMBES:    1,
}

# Extra params for Basket1 spread logic (from snippet)
BASKET1_SPREAD_PARAMS = {
    "default_spread_mean": 100,
    "spread_std_window":   100,
    "zscore_threshold":    3,
    "target_position":     70,
    "confirmation_ticks":  2,
    "stop_loss":           10,
}

# Original params for other products plus minimal baseline for KELP
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
        # The snippet's KELP logic relies on certain fields,
        # but we keep "fair_value" for fallback if needed
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
    Product.DJEMBES: {
        "fair_value": 13450
    },
    Product.JAMS: {
        "fair_value": 6500
    }
}

# Integrated position limits, including baskets + CROISSANTS
POSITION_LIMITS = {
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUID_INK: 50,
    Product.DJEMBES: 60,
    Product.JAMS: 350,
    Product.CROISSANTS: 250,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
}

# Picnic-basket hedge sets from snippet #2 (we keep these for basket2)
HEDGE_SETS = {
    "A": {
        Product.PICNIC_BASKET1: {Product.CROISSANTS: 6,  Product.JAMS: 3,  Product.DJEMBES: 1},
        Product.PICNIC_BASKET2: {Product.CROISSANTS: 4,  Product.JAMS: 2},
    },
    "B": {
        Product.PICNIC_BASKET1: {Product.CROISSANTS: 6,  Product.JAMS: 3,  Product.DJEMBES: 1},
        Product.PICNIC_BASKET2: {Product.CROISSANTS: 4,  Product.JAMS: 2},
    },
}

DEFAULT_TRADE_SIZE = 10
ROLLING_WINDOW = 20

###############################################################################
#                                LOGGER CLASS                                 #
###############################################################################
from datamodel import Listing, Observation, Trade, ProsperityEncoder, Symbol

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
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

        # We truncate state.traderData, trader_data, and self.logs to fit the log limit
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

    def compress_state(self, state: TradingState, trader_data: str):
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

    def compress_listings(self, listings: dict[Symbol, Listing]):
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]):
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]):
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation):
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]):
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

###############################################################################
#                           TRADER CLASS (MAIN)                               #
###############################################################################

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = POSITION_LIMITS
        self.kelp_prices = []  # Initialize list for storing KELP prices
        self.kelp_vwap = []    # Initialize list for storing KELP VWAP data

        # Add the snippet Basket1 spread parameters here (no changes to the main PARAMS):
        self.basket1_params = BASKET1_SPREAD_PARAMS

    ############################################################################
    #                            EXISTING LOGIC                                #
    ############################################################################

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
    ):
        position_limit = self.LIMIT[product]

        # Take from the best ask
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            # If not 'prevent_adverse' or the ask volume is small enough
            if (not prevent_adverse) or (abs(best_ask_amount) <= adverse_volume):
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Take from the best bid
        if order_depth.buy_orders:
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
    ):
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
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # Clear if net positive
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # Clear if net negative
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
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
    ):
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
    ):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume
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

    ############################################################################
    #                   NEW KELP LOGIC (from your snippet)                     #
    ############################################################################

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
            if best_ask <= fair_value - take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(best_ask), quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(best_bid), -quantity))
                        sell_order_volume += quantity
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, Product.KELP,
                buy_order_volume, sell_order_volume, fair_value, 2
            )
            aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
            bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(Product.KELP, int(bbbf + 1), buy_quantity))
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(Product.KELP, int(baaf - 1), -sell_quantity))
        return orders
    
    def clear_position_order(self,
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
        product = Product.RAINFOREST_RESIN
        baaf = min(
            [p for p in order_depth.sell_orders if p > self.params[Product.RAINFOREST_RESIN]["fair_value"] + 1],
            default=self.params[Product.RAINFOREST_RESIN]["fair_value"] + 2,
        )
        bbbf = max(
            [p for p in order_depth.buy_orders if p < self.params[Product.RAINFOREST_RESIN]["fair_value"] - 1],
            default=self.params[Product.RAINFOREST_RESIN]["fair_value"] - 2,
        )
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < self.params[Product.RAINFOREST_RESIN]["fair_value"]:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > self.params[Product.RAINFOREST_RESIN]["fair_value"]:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, product,
            buy_order_volume, sell_order_volume, self.params[Product.RAINFOREST_RESIN]["fair_value"], 1
        )
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bbbf + 1, buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, baaf - 1, -sell_quantity))
        return orders

    ############################################################################
    #                            SQUID INK LOGIC                               #
    ############################################################################

    def rsi_squid_ink_logic(self, state, trader_state) -> List[Order]:
        """
        RSI approach for SQUID_INK, left intact from your code.
        """
        orders: List[Order] = []
        product = Product.SQUID_INK
        order_depth = state.order_depths.get(product, None)
        if not order_depth:
            return orders

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

            # If ask <1900 and RSI<30 => buy
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_qty = order_depth.sell_orders[best_ask]  # negative
                can_buy = limit - position
                if best_ask < 1900 and rsi < 30 and can_buy > 0:
                    quantity = min(5, can_buy, -ask_qty)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))

            # If bid >2100 and RSI>70 => sell
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_qty = order_depth.buy_orders[best_bid]
                can_sell = limit + position
                if best_bid > 2100 and rsi > 70 and can_sell > 0:
                    quantity = min(5, can_sell, bid_qty)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))

        return orders

    def squid_ink_logic_algo2(self, state) -> List[Order]:
        """
        Alternate SQUID_INK logic from your snippet (unchanged).
        """
        product = Product.SQUID_INK
        od = state.order_depths.get(product, None)
        orders: List[Order] = []
        if not od:
            return orders

        fair_value = self.params[product]["fair_value"]
        limit = self.LIMIT[product]
        position = state.position.get(product, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_bid_vol = od.buy_orders[best_bid] if best_bid else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_ask_vol = od.sell_orders[best_ask] if best_ask else 0

        # If best ask < fair_value => buy
        if best_ask is not None and best_ask < fair_value:
            can_buy = limit - position
            if can_buy > 0:
                ask_qty = -best_ask_vol
                quantity = min(can_buy, ask_qty)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))

        # If best bid > fair_value => sell
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

    ############################################################################
    #                 PICNIC BASKET LOGIC (Basket2 + old snippet)             #
    ############################################################################
    def run_picnic_basket_logic(self, state: TradingState, trader_state: dict, result: dict):
        """
        We keep the original snippet #2 logic for PICNIC_BASKET2,
        but skip PICNIC_BASKET1 so we can replace it with the new logic below.
        """
        hist = trader_state.setdefault("picnic_hist", {})

        for hedge_key in HEDGE_SETS:
            for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
                hist.setdefault(f"pnl_{hedge_key}_{basket}", [])

        def get_best_bid_ask(od: OrderDepth):
            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_bid_qty = od.buy_orders[best_bid] if best_bid is not None else 0
            best_ask = min(od.sell_orders) if od.sell_orders else None
            best_ask_qty = od.sell_orders[best_ask] if best_ask is not None else 0
            return best_bid, best_bid_qty, best_ask, best_ask_qty

        def get_fair_value(basket: str, ratio: dict):
            total = 0
            for prod, qty in ratio.items():
                od_c = state.order_depths.get(prod)
                if not od_c:
                    return None
                c_bid, _, c_ask, _ = get_best_bid_ask(od_c)
                if c_bid is None or c_ask is None:
                    return None
                mid = 0.5 * (c_bid + c_ask)
                total += mid * qty
            return total

        def can_trade(prod: str, q: int):
            pos_now = state.position.get(prod, 0)
            lim = self.LIMIT[prod]
            return abs(pos_now + q) <= lim

        def place_order(symbol: str, price: float, qty: int):
            if symbol not in result:
                result[symbol] = []
            result[symbol].append(Order(symbol, price, qty))

        def best_hedge_key(basket: str):
            scores = {}
            for k in HEDGE_SETS:
                arr = hist[f"pnl_{k}_{basket}"]
                recent = arr[-ROLLING_WINDOW:]
                scores[k] = sum(recent)
            best_k = max(scores, key=scores.get)
            return best_k

        # We skip the logic if basket == PICNIC_BASKET1
        for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if basket == Product.PICNIC_BASKET1:
                # *** Skip old snippet logic for Basket1, replaced below ***
                continue

            od_basket = state.order_depths.get(basket)
            if not od_basket:
                continue
            best_bid, bid_qty, best_ask, ask_qty = get_best_bid_ask(od_basket)
            if best_bid is None or best_ask is None:
                continue

            # 1) pick best hedge set
            hedge_k = best_hedge_key(basket)
            ratio = HEDGE_SETS[hedge_k][basket]

            # 2) compute fair
            fair = get_fair_value(basket, ratio)
            if fair is None:
                continue

            # 3) threshold
            threshold = 0.0015 * fair

            # 4) basket position
            pos_basket = state.position.get(basket, 0)

            # Buy basket, short components
            if best_ask + threshold < fair and pos_basket < self.LIMIT[basket]:
                max_lots = min(-ask_qty, self.LIMIT[basket] - pos_basket, 3)
                for _ in range(int(max_lots)):
                    if not can_trade(basket, +1):
                        break
                    place_order(basket, best_ask, +1)
                    trade_pnl = fair - best_ask
                    hist[f"pnl_{hedge_k}_{basket}"].append(trade_pnl)
                    if len(hist[f"pnl_{hedge_k}_{basket}"]) > ROLLING_WINDOW:
                        hist[f"pnl_{hedge_k}_{basket}"].pop(0)
                    # Short components
                    for comp, qcomp in ratio.items():
                        if can_trade(comp, -qcomp):
                            od_c = state.order_depths.get(comp)
                            c_bid, c_bid_qty, _, _ = get_best_bid_ask(od_c)
                            if c_bid is not None:
                                place_order(comp, c_bid, -qcomp)

            # Sell basket, buy components
            if best_bid - threshold > fair and pos_basket > -self.LIMIT[basket]:
                max_lots = min(bid_qty, self.LIMIT[basket] + pos_basket, 3)
                for _ in range(int(max_lots)):
                    if not can_trade(basket, -1):
                        break
                    place_order(basket, best_bid, -1)
                    trade_pnl = best_bid - fair
                    hist[f"pnl_{hedge_k}_{basket}"].append(trade_pnl)
                    if len(hist[f"pnl_{hedge_k}_{basket}"]) > ROLLING_WINDOW:
                        hist[f"pnl_{hedge_k}_{basket}"].pop(0)
                    # Buy components
                    for comp, qcomp in ratio.items():
                        if can_trade(comp, +qcomp):
                            od_c = state.order_depths.get(comp)
                            _, _, c_ask, c_ask_qty = get_best_bid_ask(od_c)
                            if c_ask is not None:
                                place_order(comp, c_ask, +qcomp)

    ############################################################################
    #                   NEW BASKET1 LOGIC (from your snippet)                  #
    ############################################################################

    def get_swmid(self, od: OrderDepth) -> float:
        """Volume-weighted mid from the snippet."""
        if not od.buy_orders or not od.sell_orders:
            return 0
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        vol_bid = abs(od.buy_orders[best_bid])
        vol_ask = abs(od.sell_orders[best_ask])
        total_vol = vol_bid + vol_ask
        if total_vol == 0:
            return 0
        return (best_bid * vol_ask + best_ask * vol_bid) / total_vol

    def get_synthetic_basket_order_depth_PB1(self, order_depths):
        """
        Creates a synthetic 'order depth' for Basket1,
        using best bids/asks of components * their PB1_WEIGHTS.
        """
        synthetic_od = OrderDepth()

        # CROISSANTS
        best_bid_c = (max(order_depths[Product.CROISSANTS].buy_orders.keys())
                      if order_depths[Product.CROISSANTS].buy_orders else 0)
        best_ask_c = (min(order_depths[Product.CROISSANTS].sell_orders.keys())
                      if order_depths[Product.CROISSANTS].sell_orders else float('inf'))

        # JAMS
        best_bid_j = (max(order_depths[Product.JAMS].buy_orders.keys())
                      if order_depths[Product.JAMS].buy_orders else 0)
        best_ask_j = (min(order_depths[Product.JAMS].sell_orders.keys())
                      if order_depths[Product.JAMS].sell_orders else float('inf'))

        # DJEMBES
        best_bid_d = (max(order_depths[Product.DJEMBES].buy_orders.keys())
                      if order_depths[Product.DJEMBES].buy_orders else 0)
        best_ask_d = (min(order_depths[Product.DJEMBES].sell_orders.keys())
                      if order_depths[Product.DJEMBES].sell_orders else float('inf'))

        implied_bid = (best_bid_c * PB1_WEIGHTS[Product.CROISSANTS] +
                       best_bid_j * PB1_WEIGHTS[Product.JAMS] +
                       best_bid_d * PB1_WEIGHTS[Product.DJEMBES])
        implied_ask = (best_ask_c * PB1_WEIGHTS[Product.CROISSANTS] +
                       best_ask_j * PB1_WEIGHTS[Product.JAMS] +
                       best_ask_d * PB1_WEIGHTS[Product.DJEMBES])

        # Just store a quantity of 1 for each synthetic price level
        synthetic_od.buy_orders[implied_bid] = 1
        synthetic_od.sell_orders[implied_ask] = -1
        return synthetic_od

    def execute_spread_orders_pb1(self, target_position: int, basket_position: int, order_depths):
        """Snippet logic for sending an order to get basket_position => target_position."""
        if target_position == basket_position:
            return {}

        basket_od = order_depths[Product.PICNIC_BASKET1]
        orders: List[Order] = []
        target_quantity = abs(target_position - basket_position)
        if target_position > basket_position:
            # We want to buy the basket
            if not basket_od.sell_orders:
                return {}
            best_ask = min(basket_od.sell_orders.keys())
            available_volume = abs(basket_od.sell_orders[best_ask])
            execute_volume = min(available_volume, target_quantity)
            if execute_volume > 0:
                orders.append(Order(Product.PICNIC_BASKET1, best_ask, execute_volume))
        else:
            # We want to sell the basket
            if not basket_od.buy_orders:
                return {}
            best_bid = max(basket_od.buy_orders.keys())
            available_volume = basket_od.buy_orders[best_bid]
            execute_volume = min(available_volume, target_quantity)
            if execute_volume > 0:
                orders.append(Order(Product.PICNIC_BASKET1, best_bid, -execute_volume))

        return {Product.PICNIC_BASKET1: orders}

    def check_stop_loss_pb1(self, basket_position: int, entry_price: float, current_od: OrderDepth):
        """Snippet's stop-loss logic for Basket1."""
        stop_loss = self.basket1_params["stop_loss"]
        if basket_position > 0:
            if current_od.buy_orders:
                best_bid = max(current_od.buy_orders.keys())
                if best_bid <= entry_price - stop_loss:
                    return {Product.PICNIC_BASKET1: [Order(Product.PICNIC_BASKET1, best_bid, -basket_position)]}
        elif basket_position < 0:
            if current_od.sell_orders:
                best_ask = min(current_od.sell_orders.keys())
                if best_ask >= entry_price + stop_loss:
                    return {Product.PICNIC_BASKET1: [Order(Product.PICNIC_BASKET1, best_ask, -basket_position)]}
        return {}

    def spread_orders_pb1(self, order_depths, basket_position: int, basket_state: dict):
        """Core snippet logic for deciding trades in Basket1 based on spread vs synthetic."""
        if Product.PICNIC_BASKET1 not in order_depths:
            return {}

        basket_od = order_depths[Product.PICNIC_BASKET1]
        synthetic_od = self.get_synthetic_basket_order_depth_PB1(order_depths)

        basket_swmid = self.get_swmid(basket_od)
        synthetic_swmid = self.get_swmid(synthetic_od)
        spread = basket_swmid - synthetic_swmid

        basket_state["spread_history"].append(spread)
        window = self.basket1_params["spread_std_window"]
        if len(basket_state["spread_history"]) < window:
            return {}
        elif len(basket_state["spread_history"]) > window:
            basket_state["spread_history"].pop(0)

        try:
            spread_std = statistics.stdev(basket_state["spread_history"])
        except statistics.StatisticsError:
            return {}

        if spread_std == 0:
            return {}

        default_mean = self.basket1_params["default_spread_mean"]
        zscore = (spread - default_mean) / spread_std

        threshold = self.basket1_params["zscore_threshold"]
        if abs(zscore) >= threshold:
            basket_state["confirmation_count"] += 1
        else:
            basket_state["confirmation_count"] = 0

        if basket_state["confirmation_count"] < self.basket1_params["confirmation_ticks"]:
            return {}

        target_pos = self.basket1_params["target_position"]
        if zscore >= threshold:
            # Overvalued => short basket => negative position
            if basket_position != -target_pos:
                return self.execute_spread_orders_pb1(-target_pos, basket_position, order_depths)
        elif zscore <= -threshold:
            # Undervalued => buy basket => positive position
            if basket_position != target_pos:
                return self.execute_spread_orders_pb1(target_pos, basket_position, order_depths)

        return {}

    ############################################################################
    #                           MAIN run(...) METHOD                           #
    ############################################################################

    def run(self, state: TradingState):
        try:
            trader_state = jsonpickle.decode(state.traderData)
            if not isinstance(trader_state, dict):
                trader_state = {}
        except:
            trader_state = {}

        # Ensure a "prices" dict for each product
        for prod in POSITION_LIMITS:
            if prod not in trader_state:
                trader_state[prod] = {"prices": []}

        result = {}

        # --- 1) DJEMBES and JAMS logic (unchanged) ---
        for product in [Product.DJEMBES, Product.JAMS]:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.LIMIT[product]
            best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

            mid_price = None
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid_price = best_bid
            elif best_ask is not None:
                mid_price = best_ask

            if mid_price is not None:
                trader_state[product]["prices"].append(mid_price)
                if len(trader_state[product]["prices"]) > 2 * ROLLING_WINDOW:
                    trader_state[product]["prices"] = trader_state[product]["prices"][-2*ROLLING_WINDOW:]

            fv = self.params.get(product, {}).get("fair_value", mid_price if mid_price else 0)

            # Simple "if ask < FV => buy, if bid > FV => sell"
            if best_ask is not None and best_ask < fv:
                qty = min(limit - position, -od.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            if best_bid is not None and best_bid > fv:
                qty = min(limit + position, od.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            if orders:
                print(f"[DjembeJamTrader] {product} => {orders}")
            result[product] = orders

        # --- 2) RAINFOREST_RESIN logic (unchanged) ---
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            position = state.position.get(Product.RAINFOREST_RESIN, 0)
            result[Product.RAINFOREST_RESIN] = self.rainforest_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN], position, self.LIMIT[Product.RAINFOREST_RESIN]
            )


        # --- 3) KELP logic: replaced with your snippet approach ---
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            position = state.position.get(Product.KELP, 0)
            result[Product.KELP] = self.kelp_orders(
                state.order_depths[Product.KELP],
                timespan=10,
                width=3.5,
                take_width=1,
                position=position,
                position_limit=self.LIMIT[Product.KELP]
            )


        # --- 4) SQUID_INK logic (using your "algo2" version) ---
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = self.squid_ink_logic_algo2(state)

        # --- 5) PICNIC BASKETS logic (Basket2 from the old snippet) ---
        self.run_picnic_basket_logic(state, trader_state, result)

        # --- 6) NEW BASKET1 Logic from your snippet ---
        # We store BASKET1 data similarly to the snippet:
        if "BASKET1" not in trader_state:
            trader_state["BASKET1"] = {
                "entry_price": None,
                "spread_history": [],
                "confirmation_count": 0
            }

        basket1_position = state.position.get(Product.PICNIC_BASKET1, 0)
        if basket1_position == 0:
            trader_state["BASKET1"]["entry_price"] = None

        if Product.PICNIC_BASKET1 in state.order_depths:
            # If we have a nonzero basket position, check stop-loss:
            if basket1_position != 0 and trader_state["BASKET1"]["entry_price"] is not None:
                stop_orders = self.check_stop_loss_pb1(
                    basket1_position,
                    trader_state["BASKET1"]["entry_price"],
                    state.order_depths[Product.PICNIC_BASKET1]
                )
                if stop_orders:
                    # If stop triggered, place that order and reset:
                    if Product.PICNIC_BASKET1 not in result:
                        result[Product.PICNIC_BASKET1] = []
                    result[Product.PICNIC_BASKET1].extend(stop_orders[Product.PICNIC_BASKET1])
                    trader_state["BASKET1"]["confirmation_count"] = 0
                    trader_state["BASKET1"]["entry_price"] = None
                else:
                    # Otherwise, run the main spread logic
                    orders_pb1 = self.spread_orders_pb1(state.order_depths, basket1_position, trader_state["BASKET1"])
                    for symb, ords in orders_pb1.items():
                        if symb not in result:
                            result[symb] = []
                        result[symb].extend(ords)
                    # If we flatten position => update entry_price
                    new_pos = state.position.get(Product.PICNIC_BASKET1, 0)
                    if new_pos == 0 and Product.PICNIC_BASKET1 in orders_pb1:
                        mid_now = self.get_swmid(state.order_depths[Product.PICNIC_BASKET1])
                        trader_state["BASKET1"]["entry_price"] = mid_now
            else:
                # If basket pos = 0 or no stored entry, run spread logic
                orders_pb1 = self.spread_orders_pb1(state.order_depths, basket1_position, trader_state["BASKET1"])
                for symb, ords in orders_pb1.items():
                    if symb not in result:
                        result[symb] = []
                    result[symb].extend(ords)
                # If we opened a new position, store entry_price
                new_pos = state.position.get(Product.PICNIC_BASKET1, 0)
                if new_pos != 0 and Product.PICNIC_BASKET1 in orders_pb1:
                    mid_now = self.get_swmid(state.order_depths[Product.PICNIC_BASKET1])
                    trader_state["BASKET1"]["entry_price"] = mid_now

        # We finalize
        conversions = 1
        traderData = jsonpickle.encode(trader_state)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
