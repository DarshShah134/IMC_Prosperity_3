from datamodel import OrderDepth, TradingState, Order, Listing, Observation, Trade, ProsperityEncoder, Symbol
from typing import List
import jsonpickle
import math
import json
import statistics  # for stdev in the new basket1 snippet

###############################################################################
# CONSTANTS
###############################################################################
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    # We keep these for basket2 references:
    CROISSANTS = "CROISSANTS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

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
    },
    "SPREAD1": {
        "default_spread_mean": 48.777856,
        "default_spread_std": 85.119723,
        "spread_window": 55,
        "zscore_threshold": 4,
        "target_position": 100
    },
    Product.PICNIC_BASKET1: {
        "adverse_volume": 999999,
        "b2_adjustment_factor": 0.05,
    },
}

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

HEDGE_SETS = {
    "A": {
        Product.PICNIC_BASKET1: {Product.CROISSANTS: -4, Product.JAMS: -5, Product.DJEMBES: -4},
        Product.PICNIC_BASKET2: {Product.CROISSANTS: -4, Product.JAMS: -3},
    },
    "B": {
        Product.PICNIC_BASKET1: {Product.CROISSANTS: -4, Product.JAMS: -5, Product.DJEMBES: -4},
        Product.PICNIC_BASKET2: {Product.CROISSANTS: -4, Product.JAMS: -3},
    },
}

DEFAULT_TRADE_SIZE = 10
ROLLING_WINDOW = 20

###############################################################################
# LOGGER CLASS
###############################################################################
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
# TRADER CLASS
###############################################################################
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = POSITION_LIMITS
        # Retain any state you need for Kelp, Squid, etc. from the main algo:
        self.kelp_prices = []
        self.kelp_vwap = []

        ################################################################
        # (NEW) Create an instance of the VolcanicRock manager here.
        ################################################################
        self.volcanic_manager = IgneousStoneCouponManager()

    ############################################################################
    #                     SHARED / LOW-LEVEL UTILITIES (unchanged)
    ############################################################################

    def _take_best_orders(
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

    def _market_make(
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

    def _clear_position_order(
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


    ############################################################################
    #                   PRODUCT: JAMS (unchanged from main)
    ############################################################################
    def get_orders_jams(self, state: TradingState, trader_state: dict) -> List[Order]:
        product = Product.JAMS
        if product not in state.order_depths:
            return []
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

        # If best ask < FV => buy
        if best_ask is not None and best_ask < fv:
            qty = min(limit - position, -od.sell_orders[best_ask], DEFAULT_TRADE_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))

        # If best bid > FV => sell
        if best_bid is not None and best_bid > fv:
            qty = min(limit + position, od.buy_orders[best_bid], DEFAULT_TRADE_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        return orders

    ############################################################################
    #                PRODUCT: RAINFOREST_RESIN  (unchanged)
    ############################################################################
    def get_orders_rainforest_resin(self, state: TradingState) -> List[Order]:
        product = Product.RAINFOREST_RESIN
        if product not in state.order_depths:
            return []
        od = state.order_depths[product]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        position_limit = self.LIMIT[product]
        fv = self.params[product]["fair_value"]

        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None

        buy_order_volume = 0
        sell_order_volume = 0

        if best_ask is not None and best_ask < fv:
            best_ask_amount = -od.sell_orders[best_ask]
            qty = min(best_ask_amount, position_limit - position)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_order_volume += qty

        if best_bid is not None and best_bid > fv:
            best_bid_amount = od.buy_orders[best_bid]
            qty = min(best_bid_amount, position_limit + position)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_order_volume += qty

        buy_order_volume, sell_order_volume = self._clear_position_order(
            product,
            fv,
            1,
            orders,
            od,
            position,
            buy_order_volume,
            sell_order_volume
        )

        # Place final edges
        baaf = min(
            [p for p in od.sell_orders if p > fv + 1],
            default=fv + 2,
        )
        bbbf = max(
            [p for p in od.buy_orders if p < fv - 1],
            default=fv - 2,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, baaf - 1, -sell_quantity))

        return orders

    ############################################################################
    #                   PRODUCT: KELP  (unchanged)
    ############################################################################
    def _clear_position_order_kelp(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        fair_value: float,
        buy_order_volume: int,
        sell_order_volume: int,
        width: int
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # If net positive => flatten at around fair_for_ask
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(Product.KELP, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        # If net negative => flatten at around fair_for_bid
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(Product.KELP, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def get_orders_kelp(self, state: TradingState) -> List[Order]:
        product = Product.KELP
        if product not in state.order_depths:
            return []

        od = state.order_depths[product]
        position = state.position.get(product, 0)
        position_limit = self.LIMIT[product]

        if not od.sell_orders or not od.buy_orders:
            return []

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())

        filtered_ask = [p for p in od.sell_orders if abs(od.sell_orders[p]) >= 15]
        filtered_bid = [p for p in od.buy_orders if abs(od.buy_orders[p]) >= 15]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        mmmid_price = (mm_ask + mm_bid) / 2

        self.kelp_prices.append(mmmid_price)

        volume = -od.sell_orders[best_ask] + od.buy_orders[best_bid]  # total at best inside
        if volume != 0:
            vwap = ((best_bid * (-od.sell_orders[best_ask])) +
                    (best_ask * od.buy_orders[best_bid])) / volume
        else:
            vwap = mmmid_price

        self.kelp_vwap.append({"vol": volume, "vwap": vwap})
        if len(self.kelp_vwap) > 10:
            self.kelp_vwap.pop(0)
        if len(self.kelp_prices) > 10:
            self.kelp_prices.pop(0)

        fair_value = mmmid_price

        take_width = 1
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if best_ask <= fair_value - take_width:
            ask_amount = -od.sell_orders[best_ask]
            if ask_amount <= 20:
                qty = min(ask_amount, position_limit - position)
                if qty > 0:
                    orders.append(Order(product, int(best_ask), qty))
                    buy_order_volume += qty

        if best_bid >= fair_value + take_width:
            bid_amount = od.buy_orders[best_bid]
            if bid_amount <= 20:
                qty = min(bid_amount, position_limit + position)
                if qty > 0:
                    orders.append(Order(product, int(best_bid), -qty))
                    sell_order_volume += qty

        buy_order_volume, sell_order_volume = self._clear_position_order_kelp(
            orders,
            od,
            position,
            position_limit,
            fair_value,
            buy_order_volume,
            sell_order_volume,
            width=2
        )

        aaf = [p for p in od.sell_orders if p > fair_value + 1]
        bbf = [p for p in od.buy_orders if p < fair_value - 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbbf = max(bbf) if bbf else fair_value - 2

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(bbbf + 1), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(baaf - 1), -sell_quantity))

        return orders

    ############################################################################
    #                   PRODUCT: SQUID_INK  (unchanged)
    ############################################################################
    def get_orders_squid_ink(self, state: TradingState, trader_state: dict) -> List[Order]:
        product = Product.SQUID_INK
        if product not in state.order_depths:
            return []
        od = state.order_depths[product]
        orders: List[Order] = []

        fv = self.params[product]["fair_value"]
        limit = self.LIMIT[product]
        position = state.position.get(product, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_bid_vol = od.buy_orders[best_bid] if best_bid else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_ask_vol = od.sell_orders[best_ask] if best_ask else 0

        if best_ask is not None and best_ask < fv:
            can_buy = limit - position
            if can_buy > 0:
                ask_qty = -best_ask_vol
                quantity = min(can_buy, ask_qty)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))

        if best_bid is not None and best_bid > fv:
            can_sell = limit + position if position >= 0 else limit - abs(position)
            if can_sell > 0:
                bid_qty = best_bid_vol
                quantity = min(can_sell, bid_qty)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))

        return orders

    ############################################################################
    #                   PICNIC_BASKET2 logic (unchanged)
    ############################################################################
    def get_orders_picnic_basket2(self, state: TradingState, trader_state: dict, result: dict):
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
            if basket == Product.PICNIC_BASKET1:
                # SKIP the old snippet logic for Basket1
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
                    for comp, qcomp in ratio.items():
                        if can_trade(comp, +qcomp):
                            od_c = state.order_depths.get(comp)
                            _, _, c_ask, c_ask_qty = get_best_bid_ask(od_c)
                            if c_ask is not None:
                                place_order(comp, c_ask, +qcomp)

    ############################################################################
    # *** NEW: EXACT “DJEMBES / CROISSANTS / BASKET1” from second algo snippet
    ############################################################################
    def get_microprice(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def artifical_order_depth(self, order_depths: dict, picnic1: bool = True) -> OrderDepth:
        od = OrderDepth()
        if picnic1:
            DJEMBES_PER_PICNIC = 1
            CROISSANTS_PER_PICNIC = 6
            JAMS_PER_PICNIC = 3
        else:
            CROISSANTS_PER_PICNIC = 4
            JAMS_PER_PICNIC = 2

        croissant_od = order_depths.get(Product.CROISSANTS, None)
        jams_od = order_depths.get(Product.JAMS, None)
        if not croissant_od or not jams_od:
            return od

        croissant_best_bid = max(croissant_od.buy_orders) if croissant_od.buy_orders else 0
        croissant_best_ask = min(croissant_od.sell_orders) if croissant_od.sell_orders else float("inf")

        jams_best_bid = max(jams_od.buy_orders) if jams_od.buy_orders else 0
        jams_best_ask = min(jams_od.sell_orders) if jams_od.sell_orders else float("inf")

        if picnic1:
            dj_od = order_depths.get(Product.DJEMBES, None)
            if not dj_od:
                return od
            djembes_best_bid = max(dj_od.buy_orders) if dj_od.buy_orders else 0
            djembes_best_ask = min(dj_od.sell_orders) if dj_od.sell_orders else float("inf")

            art_bid = (djembes_best_bid * DJEMBES_PER_PICNIC +
                       croissant_best_bid * CROISSANTS_PER_PICNIC +
                       jams_best_bid * JAMS_PER_PICNIC)
            art_ask = (djembes_best_ask * DJEMBES_PER_PICNIC +
                       croissant_best_ask * CROISSANTS_PER_PICNIC +
                       jams_best_ask * JAMS_PER_PICNIC)
        else:
            art_bid = (croissant_best_bid * CROISSANTS_PER_PICNIC +
                       jams_best_bid * JAMS_PER_PICNIC)
            art_ask = (croissant_best_ask * CROISSANTS_PER_PICNIC +
                       jams_best_ask * JAMS_PER_PICNIC)

        if art_bid > 0 and art_bid < float("inf"):
            od.buy_orders[art_bid] = 1
        if art_ask < float("inf") and art_ask > 0:
            od.sell_orders[art_ask] = -1

        return od

    def convert_orders(self, artifical_orders: List[Order], order_depths: dict, picnic1: bool = True):
        if picnic1:
            component_orders = {
                Product.DJEMBES: [],
                Product.CROISSANTS: [],
                Product.JAMS: []
            }
            DJEMBES_PER_PICNIC = 1
            CROISSANTS_PER_PICNIC = 6
            JAMS_PER_PICNIC = 3
        else:
            component_orders = {
                Product.CROISSANTS: [],
                Product.JAMS: []
            }
            CROISSANTS_PER_PICNIC = 4
            JAMS_PER_PICNIC = 2

        artificial_od = self.artifical_order_depth(order_depths, picnic1)
        best_bid = max(artificial_od.buy_orders) if artificial_od.buy_orders else 0
        best_ask = min(artificial_od.sell_orders) if artificial_od.sell_orders else float("inf")

        for order in artifical_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                # “Buy at best ask scenario”
                croissant_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                if picnic1:
                    djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # “Sell at best bid scenario”
                croissant_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                if picnic1:
                    djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue

            croissants_ord = Order(
                Product.CROISSANTS,
                croissant_price,
                quantity * CROISSANTS_PER_PICNIC
            )
            jams_ord = Order(
                Product.JAMS,
                jams_price,
                quantity * JAMS_PER_PICNIC
            )
            if picnic1:
                djembes_ord = Order(
                    Product.DJEMBES,
                    djembes_price,
                    quantity * DJEMBES_PER_PICNIC
                )
                component_orders[Product.DJEMBES].append(djembes_ord)

            component_orders[Product.CROISSANTS].append(croissants_ord)
            component_orders[Product.JAMS].append(jams_ord)

        return component_orders

    def execute_spreads(self, target_position: int, basket_position: int,
                        order_depths: dict, picnic1: bool = True):
        if picnic1:
            basket = Product.PICNIC_BASKET1
        else:
            basket = Product.PICNIC_BASKET2

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        picnic_od = order_depths[basket]
        artificial_od = self.artifical_order_depth(order_depths, picnic1)

        if target_position > basket_position:
            # BUY more of the basket
            if not picnic_od.sell_orders or not artificial_od.buy_orders:
                return None
            picnic_ask_price = min(picnic_od.sell_orders.keys())
            picnic_ask_vol = abs(picnic_od.sell_orders[picnic_ask_price])

            art_bid_price = max(artificial_od.buy_orders.keys())
            art_bid_vol = artificial_od.buy_orders[art_bid_price]

            orderbook_volume = min(picnic_ask_vol, abs(art_bid_vol))
            execute_volume = min(orderbook_volume, target_quantity)

            picnic_orders = [Order(basket, picnic_ask_price, execute_volume)]
            artificial_orders = [Order("ARTIFICAL1", art_bid_price, -execute_volume)]

            agg = self.convert_orders(artificial_orders, order_depths, picnic1)
            agg[basket] = picnic_orders
            return agg
        else:
            # SELL some of the basket
            if not picnic_od.buy_orders or not artificial_od.sell_orders:
                return None
            picnic_bid_price = max(picnic_od.buy_orders.keys())
            picnic_bid_vol = picnic_od.buy_orders[picnic_bid_price]

            art_ask_price = min(artificial_od.sell_orders.keys())
            art_ask_vol = abs(artificial_od.sell_orders[art_ask_price])

            orderbook_volume = min(picnic_bid_vol, art_ask_vol)
            execute_volume = min(orderbook_volume, target_quantity)

            picnic_orders = [Order(basket, picnic_bid_price, -execute_volume)]
            artificial_orders = [Order("ARTIFICAL1", art_ask_price, -execute_volume)]

            agg = self.convert_orders(artificial_orders, order_depths, picnic1)
            agg[basket] = picnic_orders
            return agg

    def spread_orders(self,
                      order_depths: dict,
                      picnic_product: str,
                      picnic_position: int,
                      spread_data: dict,
                      SPREAD: str,
                      picnic1: bool = True):
        if picnic1:
            basket = Product.PICNIC_BASKET1
        else:
            basket = Product.PICNIC_BASKET2

        if basket not in order_depths:
            return None

        picnic_od = order_depths[basket]
        artificial_od = self.artifical_order_depth(order_depths, picnic1)

        picnic_mprice = self.get_microprice(picnic_od)
        art_mprice = self.get_microprice(artificial_od)
        spread = picnic_mprice - art_mprice
        spread_data["spread_history"].append(spread)

        if len(spread_data["spread_history"]) < self.params[SPREAD]["spread_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[SPREAD]["spread_window"]:
            spread_data["spread_history"].pop(0)

        import numpy as np
        spread_array = np.array(spread_data["spread_history"])
        spread_std = np.std(spread_array)
        if spread_std == 0:
            return None

        zscore = (spread - self.params[SPREAD]["default_spread_mean"]) / spread_std
        if zscore >= self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != -self.params[SPREAD]["target_position"]:
                return self.execute_spreads(-self.params[SPREAD]["target_position"],
                                            picnic_position,
                                            order_depths,
                                            picnic1)
        elif zscore <= -self.params[SPREAD]["zscore_threshold"]:
            if picnic_position != self.params[SPREAD]["target_position"]:
                return self.execute_spreads(self.params[SPREAD]["target_position"],
                                            picnic_position,
                                            order_depths,
                                            picnic1)

        spread_data["prev_zscore"] = zscore
        return None

    def get_orders_djembes_and_croissants_and_basket1(self, state: TradingState, trader_state: dict):
        if "SPREAD1" not in trader_state:
            trader_state["SPREAD1"] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0
            }

        pb1_position = state.position.get(Product.PICNIC_BASKET1, 0)

        ret = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            pb1_position,
            trader_state["SPREAD1"],
            SPREAD="SPREAD1",
            picnic1=True
        )
        return ret if ret else {}

    ############################################################################
    #                           run(...) METHOD
    ############################################################################
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

        # JAMS
        jam_orders = self.get_orders_jams(state, trader_state)
        if jam_orders:
            result[Product.JAMS] = jam_orders
            logger.print("[DjembeJamTrader] JAMS =>", jam_orders)

        # RAINFOREST_RESIN
        if Product.RAINFOREST_RESIN in self.params:
            rr_orders = self.get_orders_rainforest_resin(state)
            if rr_orders:
                result[Product.RAINFOREST_RESIN] = rr_orders

        # KELP
        if Product.KELP in self.params:
            kelp_orders = self.get_orders_kelp(state)
            if kelp_orders:
                result[Product.KELP] = kelp_orders

        # SQUID_INK
        if Product.SQUID_INK in self.params:
            si_orders = self.get_orders_squid_ink(state, trader_state)
            if si_orders:
                result[Product.SQUID_INK] = si_orders

        # PICNIC_BASKET2 logic
        self.get_orders_picnic_basket2(state, trader_state, result)

        # NEW: run second code’s basket1 logic => Djembe & Croissants
        new_basket1_orders = self.get_orders_djembes_and_croissants_and_basket1(state, trader_state)
        for symb, orders_list in new_basket1_orders.items():
            if orders_list:
                result.setdefault(symb, []).extend(orders_list)

        conversions = 1

        ################################################################
        # (NEW) Now call VolcanicRock + Vouchers logic (unchanged code).
        ################################################################
        volc_orders, volc_conversions, volc_state = self.volcanic_manager.run(state)
        for symb, ordz in volc_orders.items():
            if ordz:
                result.setdefault(symb, []).extend(ordz)
        conversions += volc_conversions
        # We ignore volc_state here because we are not altering that code.

        final_trader_data = jsonpickle.encode(trader_state)
        logger.flush(state, result, conversions, final_trader_data)
        return result, conversions, final_trader_data


###############################################################################
#                     INSERT THE “VOLCANIC” CODE UNCHANGED
###############################################################################
import json
import statistics
from abc import abstractmethod
from collections import deque
from typing import Any, TypeAlias, List, Dict, Tuple

# (We repeat these imports unaltered; we are not mixing them, just leaving them intact)
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class IgneousStoneCouponManager:
    """
    The “VolcanicRock + Volcanic Vouchers” snippet, unchanged.
    """
    def run(self, snapshot: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        out_orders: Dict[str, List[Order]] = {}
        memo_state = ""
        total_convs = 0

        # figure out base quote
        base_rock_quote = None
        if "VOLCANIC_ROCK" in snapshot.order_depths:
            rocky_orders = snapshot.order_depths["VOLCANIC_ROCK"]
            if rocky_orders.buy_orders and rocky_orders.sell_orders:
                highest_bid = max(rocky_orders.buy_orders.keys())
                lowest_ask = min(rocky_orders.sell_orders.keys())
                base_rock_quote = (highest_bid + lowest_ask) / 2.0
            elif rocky_orders.buy_orders:
                base_rock_quote = max(rocky_orders.buy_orders.keys())
            elif rocky_orders.sell_orders:
                base_rock_quote = min(rocky_orders.sell_orders.keys())

        if base_rock_quote is None:
            return out_orders, total_convs, memo_state

        coupon_strikes = {
            "VOLCANIC_ROCK": 10312,
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }

        for cpn, strk in coupon_strikes.items():
            if cpn not in snapshot.order_depths:
                continue

            cpn_depth = snapshot.order_depths[cpn]
            intrinsic_eval = base_rock_quote - strk
            if intrinsic_eval < 0:
                intrinsic_eval = 0

            local_orders: List[Order] = []
            # buy logic
            if cpn_depth.sell_orders:
                lowest_offer = min(cpn_depth.sell_orders.keys())
                if lowest_offer < intrinsic_eval:
                    liquidity = abs(cpn_depth.sell_orders[lowest_offer])
                    local_orders.append(Order(cpn, lowest_offer, liquidity))

            # sell logic
            if cpn_depth.buy_orders:
                highest_deal = max(cpn_depth.buy_orders.keys())
                if highest_deal > intrinsic_eval:
                    liquidity = cpn_depth.buy_orders[highest_deal]
                    local_orders.append(Order(cpn, highest_deal, -liquidity))

            if local_orders:
                out_orders[cpn] = local_orders

        return out_orders, total_convs, memo_state
