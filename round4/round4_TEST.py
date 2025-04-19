from datamodel import OrderDepth, TradingState, Order
import jsonpickle
import numpy as np

# Define only the relevant products
class Product:
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"

# Parameters just for SPREAD1 and basket1
PARAMS = {
    "SPREAD1": {
        "default_spread_mean": 48.777856,
        "default_spread_std": 85.119723,
        "spread_window": 55,
        "zscore_threshold": 4,
        "target_position": 100
    },
    Product.PICNIC_BASKET1: {
        "adverse_volume": 999999,
        "b2_adjustment_factor": 0.05
    }
}

# Position limits for only the traded items
POSITION_LIMITS = {
    Product.DJEMBES: 60,
    Product.JAMS: 350,
    Product.CROISSANTS: 250,
    Product.PICNIC_BASKET1: 60
}

ROLLING_WINDOW = 20

class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.LIMIT = POSITION_LIMITS

    def get_microprice(self, od: OrderDepth) -> float:
        if not od.buy_orders or not od.sell_orders:
            return 0.0
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bid_vol = abs(od.buy_orders[best_bid])
        ask_vol = abs(od.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (ask_vol + bid_vol)

    def artifical_order_depth(self, order_depths: dict, picnic1: bool = True) -> OrderDepth:
        od = OrderDepth()
        # component ratios
        if picnic1:
            DJ_PER, CR_PER, JA_PER = 1, 6, 3
        else:
            CR_PER, JA_PER = 4, 2

        cro_od = order_depths.get(Product.CROISSANTS)
        ja_od = order_depths.get(Product.JAMS)
        if not cro_od or not ja_od:
            return od

        cbid = max(cro_od.buy_orders) if cro_od.buy_orders else 0
        cask = min(cro_od.sell_orders) if cro_od.sell_orders else float('inf')
        jbid = max(ja_od.buy_orders) if ja_od.buy_orders else 0
        jask = min(ja_od.sell_orders) if ja_od.sell_orders else float('inf')

        if picnic1:
            dj_od = order_depths.get(Product.DJEMBES)
            if not dj_od:
                return od
            dbid = max(dj_od.buy_orders) if dj_od.buy_orders else 0
            dask = min(dj_od.sell_orders) if dj_od.sell_orders else float('inf')
            bid = dbid*DJ_PER + cbid*CR_PER + jbid*JA_PER
            ask = dask*DJ_PER + cask*CR_PER + jask*JA_PER
        else:
            bid = cbid*CR_PER + jbid*JA_PER
            ask = cask*CR_PER + jask*JA_PER

        if 0 < bid < float('inf'):
            od.buy_orders[bid] = 1
        if 0 < ask < float('inf'):
            od.sell_orders[ask] = -1
        return od

    def convert_orders(self, artificial_orders: list, order_depths: dict, picnic1: bool = True):
        # prepare output dict
        if picnic1:
            out = {Product.DJEMBES: [], Product.CROISSANTS: [], Product.JAMS: []}
            DJ_PER, CR_PER, JA_PER = 1, 6, 3
        else:
            out = {Product.CROISSANTS: [], Product.JAMS: []}
            CR_PER, JA_PER = 4, 2

        art_od = self.artifical_order_depth(order_depths, picnic1)
        best_bid = max(art_od.buy_orders) if art_od.buy_orders else 0
        best_ask = min(art_od.sell_orders) if art_od.sell_orders else float('inf')

        for o in artificial_orders:
            price, qty = o.price, o.quantity
            if qty > 0 and price >= best_ask:
                cp = min(order_depths[Product.CROISSANTS].sell_orders)
                jp = min(order_depths[Product.JAMS].sell_orders)
                if picnic1:
                    dp = min(order_depths[Product.DJEMBES].sell_orders)
            elif qty < 0 and price <= best_bid:
                cp = max(order_depths[Product.CROISSANTS].buy_orders)
                jp = max(order_depths[Product.JAMS].buy_orders)
                if picnic1:
                    dp = max(order_depths[Product.DJEMBES].buy_orders)
            else:
                continue

            out[Product.CROISSANTS].append(Order(Product.CROISSANTS, cp, qty*CR_PER))
            out[Product.JAMS].append(Order(Product.JAMS, jp, qty*JA_PER))
            if picnic1:
                out[Product.DJEMBES].append(Order(Product.DJEMBES, dp, qty*DJ_PER))
        return out

    def execute_spreads(self, target_pos: int, basket_pos: int, order_depths: dict, picnic1: bool = True):
        basket = Product.PICNIC_BASKET1
        if target_pos == basket_pos:
            return None
        diff = abs(target_pos - basket_pos)
        pic_od = order_depths[basket]
        art_od = self.artifical_order_depth(order_depths, picnic1)
        # buy basket
        if target_pos > basket_pos:
            if not pic_od.sell_orders or not art_od.buy_orders:
                return None
            p_ask = min(pic_od.sell_orders)
            p_vol = abs(pic_od.sell_orders[p_ask])
            a_bid = max(art_od.buy_orders)
            a_vol = art_od.buy_orders[a_bid]
            vol = min(p_vol, abs(a_vol), diff)
            pic_orders = [Order(basket, p_ask, vol)]
            art_orders = [Order('ARTIFICAL1', a_bid, -vol)]
            agg = self.convert_orders(art_orders, order_depths, picnic1)
            agg[basket] = pic_orders
            return agg
        # sell basket
        if not pic_od.buy_orders or not art_od.sell_orders:
            return None
        p_bid = max(pic_od.buy_orders)
        p_vol = pic_od.buy_orders[p_bid]
        a_ask = min(art_od.sell_orders)
        a_vol = abs(art_od.sell_orders[a_ask])
        vol = min(p_vol, a_vol, diff)
        pic_orders = [Order(basket, p_bid, -vol)]
        art_orders = [Order('ARTIFICAL1', a_ask, -vol)]
        agg = self.convert_orders(art_orders, order_depths, picnic1)
        agg[basket] = pic_orders
        return agg

    def spread_orders(self, order_depths: dict, picnic_product: str, picnic_pos: int, spread_data: dict, SPREAD: str, picnic1: bool = True):
        basket = Product.PICNIC_BASKET1
        if basket not in order_depths:
            return None
        pic_od = order_depths[basket]
        art_od = self.artifical_order_depth(order_depths, picnic1)
        spread = self.get_microprice(pic_od) - self.get_microprice(art_od)
        spread_data['spread_history'].append(spread)
        if len(spread_data['spread_history']) < self.params[SPREAD]['spread_window']:
            return None
        if len(spread_data['spread_history']) > self.params[SPREAD]['spread_window']:
            spread_data['spread_history'].pop(0)
        arr = np.array(spread_data['spread_history'])
        std = np.std(arr)
        if std == 0:
            return None
        z = (spread - self.params[SPREAD]['default_spread_mean']) / std
        if z >= self.params[SPREAD]['zscore_threshold'] and picnic_pos != -self.params[SPREAD]['target_position']:
            return self.execute_spreads(-self.params[SPREAD]['target_position'], picnic_pos, order_depths, picnic1)
        if z <= -self.params[SPREAD]['zscore_threshold'] and picnic_pos != self.params[SPREAD]['target_position']:
            return self.execute_spreads(self.params[SPREAD]['target_position'], picnic_pos, order_depths, picnic1)
        spread_data['prev_zscore'] = z
        return None

    def get_orders_djembes_and_croissants_and_basket1(self, state: TradingState, trader_state: dict):
        if 'SPREAD1' not in trader_state:
            trader_state['SPREAD1'] = { 'spread_history': [], 'prev_zscore': 0 }
        pos = state.position.get(Product.PICNIC_BASKET1, 0)
        orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, pos, trader_state['SPREAD1'], SPREAD='SPREAD1', picnic1=True)
        return orders or {}

    def run(self, state: TradingState):
        try:
            ts = jsonpickle.decode(state.traderData)
            if not isinstance(ts, dict):
                ts = {}
        except:
            ts = {}
        result = {}
        new = self.get_orders_djembes_and_croissants_and_basket1(state, ts)
        for sym, ords in new.items():
            if ords:
                result[sym] = ords
        return result, 1, jsonpickle.encode(ts)
