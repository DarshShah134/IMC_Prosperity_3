from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import math
import json

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
        # The snippet’s KELP logic relies on certain fields,
        # but we keep “fair_value” for fallback if needed
        "fair_value": 2026,
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 10,
        "reversion_beta": 0.0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        #
        # The new snippet also expects:
        #   "KELP_min_edge"
        #   "take_width"
        #   "reversion_beta"
        #   "prevent_adverse" / "adverse_volume"
        # We'll override them in the new snippet logic below if needed
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

# Picnic‐basket hedge sets from snippet #2
HEDGE_SETS = {
    "A": {
        Product.PICNIC_BASKET1: {Product.CROISSANTS: 6,  Product.JAMS: 3,  Product.DJEMBES: 1},
        Product.PICNIC_BASKET2: {Product.CROISSANTS: 4,  Product.JAMS: 2},
    },
    "B": {
        Product.PICNIC_BASKET1: {Product.CROISSANTS: -4, Product.JAMS: -5, Product.DJEMBES: -4},
        Product.PICNIC_BASKET2: {Product.CROISSANTS: -4, Product.JAMS: -3},
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

    ############################################################################
    #                   NEW KELP LOGIC (from your snippet)                     #
    ############################################################################

    def kelp_take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ):
        position_limit = self.LIMIT[product]

        # Best ask
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            # Only take if volume <= adverse_volume
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Best bid
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def kelp_take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ):
        position_limit = self.LIMIT[product]

        # Best ask
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # Best bid
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def kelp_take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool,
        adverse_volume: int,
    ):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.kelp_take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.kelp_take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def kelp_clear_position_order(
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

        if position_after_take > 0:
            clear_quantity = sum(
                vol for pr, vol in order_depth.buy_orders.items() if pr >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(vol) for pr, vol in order_depth.sell_orders.items() if pr <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_clear_orders(
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
        buy_order_volume, sell_order_volume = self.kelp_clear_position_order(
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

    def KELP_fair_value(self, order_depth: OrderDepth, traderData: dict) -> float:
        """
        KELP_fair_value approach from your snippet:
          - Looks for best_ask/bid with enough volume
          - Reverts last price by reversion_beta
          - Stores in traderData["KELP_last_price"]
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # filter by adverse volume
        adverse_vol = self.params[Product.KELP].get("adverse_volume", 0)
        filtered_ask = [
            price for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= adverse_vol
        ]
        filtered_bid = [
            price for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= adverse_vol
        ]
        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            # fallback mid
            mmmid_price = traderData.get("KELP_last_price", (best_ask + best_bid) / 2)
        else:
            mmmid_price = (mm_ask + mm_bid) / 2

        if "KELP_last_price" in traderData:
            last_price = traderData["KELP_last_price"]
            last_returns = (mmmid_price - last_price) / max(last_price, 1e-9)
            pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
            fair = mmmid_price + (mmmid_price * pred_returns)
        else:
            fair = mmmid_price

        traderData["KELP_last_price"] = mmmid_price
        return fair

    def make_KELP_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ):
        """
        Final 'make' step:
          - We place a market-making spread around fair_value (± min_edge).
        """
        orders: List[Order] = []

        # We'll identify any orders in the book that are far beyond our min_edge
        # but typically we just place a simple symmetrical spread
        # If there's a genuine best ask above (fair + min_edge),
        # we can place an ask near it. If not, we pick (fair + min_edge) - 1, etc.

        aaf = [pr for pr in order_depth.sell_orders if pr >= round(fair_value + min_edge)]
        bbf = [pr for pr in order_depth.buy_orders if pr <= round(fair_value - min_edge)]
        best_ask_above = min(aaf) if aaf else round(fair_value + min_edge)
        best_bid_below = max(bbf) if bbf else round(fair_value - min_edge)

        # Then place two orders around these prices
        final_bid = best_bid_below + 1
        final_ask = best_ask_above - 1

        buy_order_volume, sell_order_volume = self.kelp_market_make(
            Product.KELP,
            orders,
            final_bid,
            final_ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def kelp_market_make(
        self,
        product: str,
        orders: List[Order],
        bid: float,
        ask: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ):
        # Very similar to your normal market_make, but we keep it separate to preserve the snippet logic
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

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
    #                 PICNIC BASKET LOGIC INSERTED FROM SNIPPET #2            #
    ############################################################################

    def run_picnic_basket_logic(self, state: TradingState, trader_state: dict, result: dict):
        """
        The snippet #2 picnic basket logic, stored in trader_state["picnic_hist"].
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

        for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
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

            # Simple “if ask < FV => buy, if bid > FV => sell”
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

            # partner flow: take → clear → make
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

        # --- 3) KELP logic: replaced with your snippet approach ---
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            product = Product.KELP
            od = state.order_depths[product]
            position = state.position.get(product, 0)

            # 1) Compute KELP fair value (reversion)
            kelp_fv = self.KELP_fair_value(od, trader_state.get(product, {}))
            if kelp_fv is None:
                # fallback to simple mid if no data
                if od.buy_orders and od.sell_orders:
                    best_bid = max(od.buy_orders.keys())
                    best_ask = min(od.sell_orders.keys())
                    kelp_fv = (best_bid + best_ask) / 2
                else:
                    kelp_fv = self.params[product]["fair_value"]

            # 2) Take
            tw = self.params[product].get("take_width", 1)
            pa = self.params[product].get("prevent_adverse", False)
            adv = self.params[product].get("adverse_volume", 10)
            kelp_take_orders, buy_vol, sell_vol = self.kelp_take_orders(
                product,
                od,
                kelp_fv,
                tw,
                position,
                pa,
                adv,
            )
            # 3) Clear
            cw = self.params[product].get("clear_width", 0)
            kelp_clear_orders, buy_vol, sell_vol = self.kelp_clear_orders(
                product,
                od,
                kelp_fv,
                cw,
                position,
                buy_vol,
                sell_vol,
            )
            # 4) Make
            min_edge = self.params[product].get("KELP_min_edge", 2)
            kelp_make_orders, _, _ = self.make_KELP_orders(
                od,
                kelp_fv,
                min_edge,
                position,
                buy_vol,
                sell_vol,
            )

            result[product] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # --- 4) SQUID_INK logic (using your “algo2” version) ---
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = self.squid_ink_logic_algo2(state)

        # --- 5) PICNIC BASKETS logic ---
        self.run_picnic_basket_logic(state, trader_state, result)

        # We finalize
        conversions = 1
        traderData = jsonpickle.encode(trader_state)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
