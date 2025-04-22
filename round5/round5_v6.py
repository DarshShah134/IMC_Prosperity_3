import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import deque
import math
import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# --- Logger Class (unchanged) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 10000

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

# --- BS Approximations (unchanged) ---
def approx_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def approx_norm_pdf(x: float) -> float:
    if abs(x) > 30: return 0.0
    try: return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x**2)
    except OverflowError: return 0.0

# --- BS Pricing and Delta (unchanged) ---
def black_scholes_call_price_approx(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Optional[float]:
    # Calculation remains the same...
    if T <= 1e-9 or sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return max(0.0, S - K * math.exp(-r * T))
    sqrtT = math.sqrt(T)
    if sigma * sqrtT < 1e-9: return max(0.0, S - K * math.exp(-r * T))
    try:
        if S / K <= 1e-9 : return max(0.0, S - K * math.exp(-r * T))
        log_term = math.log(S / K)
        d1_num = log_term + (r + 0.5 * sigma**2) * T
        d1_den = sigma * sqrtT
        if abs(d1_den) < 1e-9: return max(0.0, S - K * math.exp(-r*T))
        d1 = d1_num / d1_den
        d2 = d1 - sigma * sqrtT
        call_price = (S * approx_norm_cdf(d1) - K * math.exp(-r * T) * approx_norm_cdf(d2))
        return max(0.0, call_price)
    except (ValueError, OverflowError, ZeroDivisionError): return max(0.0, S - K * math.exp(-r * T))

def black_scholes_delta_call_approx(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Optional[float]:
    # Calculation remains the same...
    if T <= 1e-9 or sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    if sigma * sqrtT < 1e-9: return 1.0 if S > K else 0.0
    try:
        if S / K <= 1e-9: return 1.0 if S > K else 0.0
        log_term = math.log(S / K)
        d1_num = log_term + (r + 0.5 * sigma**2) * T
        d1_den = sigma * sqrtT
        if abs(d1_den) < 1e-9: return 1.0 if S > K else 0.0
        d1 = d1_num / d1_den
        delta = approx_norm_cdf(d1)
        return delta
    except (ValueError, OverflowError, ZeroDivisionError): return 1.0 if S > K else 0.0

# --------------------------------------------------------------------------------------------------
# Trader Class
# --------------------------------------------------------------------------------------------------
class Trader:
    def __init__(self):
        self.underlying_symbol = "VOLCANIC_ROCK"
        self.voucher_symbols = [f"VOLCANIC_ROCK_VOUCHER_{s}" for s in [9500, 9750, 10000, 10250, 10500]]
        self.voucher_strikes = {symbol: int(symbol.split('_')[-1]) for symbol in self.voucher_symbols}

        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
        self.risk_free_rate = 0.0 # Keep as 0 for consistency

        # Voucher strategy parameters (UNCHANGED)
        self.voucher_trade_threshold_base_pct = 0.007 # Base threshold (0.7%) for BS mean reversion
        self.voucher_order_size = 7 # Base order size for BS mean reversion
        self.min_theoretical_iv = 0.01
        self.iv_param_a = 0.2131 # Volatility smile parameters
        self.iv_param_b = 0.0192
        self.iv_param_c = 0.1895
        self.voucher_aggressiveness = 1.5 # How much rock price dev affects thresholds/size in BS strategy
        self.iv_adjustment_sensitivity = 0.5 # How much rock price dev adjusts theoretical IV (0 to disable)

        # Volcanic Rock EMA parameters - OPTIMIZED FOR CONSISTENCY
        self.rock_ema_span = 75 # Increased from 50 for more stable price basis
        self.rock_volatility_window = 39 # Increased from 20 for more reliable volatility measurement
        self.rock_volatility_history = deque(maxlen=self.rock_volatility_window + 5) # Store recent mid-prices for volatility calc
        self.rock_price_history_ema = deque(maxlen=self.rock_ema_span + 10) # Separate deque for EMA calc
        self.current_rock_ema = None

        # Rock Market Making parameters - OPTIMIZED FOR CONSISTENCY
        self.rock_mm_spread_multiplier = 1.3  # Reduced from 1.5 for more balanced spreads
        self.rock_mm_min_spread = 3  # Increased from 2 for consistent protection
        self.rock_mm_base_order_size = 17  # Reduced from 12 for more conservative sizing
        self.rock_mm_inventory_skew_factor = 0.04  # Reduced from 0.05 for more balanced inventory management
        
        # Base IV tracking parameters - OPTIMIZED FOR CONSISTENCY
        self.base_iv_window = 30  # Increased from 20 for more stable IV signals
        self.base_iv_history = deque(maxlen=self.base_iv_window)  # Recent base IV values
        self.base_iv_ema = None  # EMA of base IV
        self.base_iv_ema_span = 20  # Increased from 10 for smoother trends
        self.iv_price_correlation_strength = 0.15  # Reduced from 0.4 for more conservative adjustments
        
        # Crash/spike protection parameters - OPTIMIZED FOR CONSISTENCY
        self.crash_protection_threshold = 2.0  # Increased from 2.5 for fewer false positives
        self.spike_protection_threshold = 3.0  # Increased from 2.5 for fewer false positives
        self.crash_protection_spread_multiplier = 1.7  # Reduced from 2.0 for more balanced protection
        self.crash_protection_size_reducer = 0.75  # Increased from 0.5 for less dramatic size reduction
        self.volatility_regime_window = 45  # Increased from 30 for more stable regime detection
        self.volatility_regime_threshold = 1.8  # Increased from 1.5 for more conservative regime switching
        
        # BS-based price adjustment parameters - OPTIMIZED FOR CONSISTENCY
        self.bs_adjustment_strength = 0.2  # Reduced from 0.3 for more conservative signals
        self.bs_moneyness_range = [-0.15, 0.0, 0.15]  # Narrower than [-0.2, 0.0, 0.2] for more reliable pricing
        
        # Mean reversion parameters - OPTIMIZED FOR CONSISTENCY
        self.rock_mr_std_dev_threshold = 2.25  # Increased from 1.5 for more reliable signals
        self.rock_mr_base_order_size = 12  # Reduced from 15 for more consistent sizing
        self.rock_mr_size_scaling_factor = 0.3  # Reduced from 0.5 for more consistent scaling

        # No persistent state needed in traderData

    # --- Helper Functions (unchanged) ---
    def _calculate_tte(self, timestamp: int) -> float:
        timesteps_per_day = 1_000_000
        days_total = 5
        days_passed = timestamp / float(timesteps_per_day)
        time_left_days = max(0.0, days_total - days_passed)
        return max(1e-9, time_left_days / 365.0)

    def _calculate_mt(self, strike: float, S: float, T: float) -> Optional[float]:
        if T <= 1e-9 or S <= 1e-9 or strike <= 1e-9: return None
        try:
            sqrtT = math.sqrt(T)
            if sqrtT < 1e-6: return None
            ratio = strike / S
            if ratio <= 1e-9: return None
            return math.log(ratio) / sqrtT
        except (ValueError, ZeroDivisionError, OverflowError): return None

    def _calculate_theoretical_iv(self, moneyness_m: float) -> float:
        theoretical_iv = (self.iv_param_a * (moneyness_m**2) +
                          self.iv_param_b * moneyness_m +
                          self.iv_param_c)
        return max(self.min_theoretical_iv, theoretical_iv)

    def _get_best_bid(self, symbol: Symbol, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders.keys()) if order_depth and order_depth.buy_orders else None
    def _get_best_ask(self, symbol: Symbol, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders.keys()) if order_depth and order_depth.sell_orders else None
    def _get_mid_price(self, symbol: Symbol, order_depth: OrderDepth) -> Optional[float]:
        best_bid = self._get_best_bid(symbol, order_depth)
        best_ask = self._get_best_ask(symbol, order_depth)
        if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
        if best_bid is not None: return float(best_bid)
        if best_ask is not None: return float(best_ask)
        return None

    def _calculate_rock_ema(self, current_rock_price: Optional[float]):
        """Updates EMA state"""
        if current_rock_price is not None:
            self.rock_price_history_ema.append(current_rock_price)
        if len(self.rock_price_history_ema) < self.rock_ema_span // 2:
            self.current_rock_ema = None
        else:
            prices = list(self.rock_price_history_ema)
            if self.current_rock_ema is None: self.current_rock_ema = np.mean(prices)
            else:
                 alpha = 2.0 / (self.rock_ema_span + 1.0)
                 price_to_use = current_rock_price if current_rock_price is not None else self.current_rock_ema
                 self.current_rock_ema = (price_to_use * alpha) + (self.current_rock_ema * (1.0 - alpha))
        return self.current_rock_ema

    def _calculate_rock_volatility(self, current_rock_price: Optional[float]) -> float:
        """Calculates recent price standard deviation as a volatility measure."""
        if current_rock_price is not None:
             self.rock_volatility_history.append(current_rock_price)
        if len(self.rock_volatility_history) < self.rock_volatility_window // 2:
             return 0.0
        recent_prices = list(self.rock_volatility_history)
        std_dev = np.std(recent_prices)
        return std_dev
    
    # --- Base IV and BS-related calculations (unchanged logic) ---
    def _estimate_base_iv(self) -> float:
        """Calculate base IV (at-the-money) from the volatility smile formula."""
        # When moneyness (m_t) = 0, the formula simplifies to just the constant term
        base_iv = self.iv_param_c
        return base_iv
    
    def _update_base_iv_history(self, current_base_iv: float):
        """Updates base IV history and EMA"""
        self.base_iv_history.append(current_base_iv)
        
        if len(self.base_iv_history) < self.base_iv_ema_span // 2:
            self.base_iv_ema = current_base_iv
        else:
            if self.base_iv_ema is None:
                self.base_iv_ema = np.mean(list(self.base_iv_history))
            else:
                alpha = 2.0 / (self.base_iv_ema_span + 1.0)
                self.base_iv_ema = (current_base_iv * alpha) + (self.base_iv_ema * (1.0 - alpha))
        
        return self.base_iv_ema
    
    def _detect_volatility_regime(self, recent_vol: float) -> bool:
        """Detects if we're in a high volatility regime"""
        if len(self.rock_volatility_history) < self.volatility_regime_window // 2:
            return False
            
        vol_history = list(self.rock_volatility_history)
        vol_mean = np.mean(vol_history)
        vol_std = np.std(vol_history)
        
        if vol_std > 1e-6:
            vol_z_score = (recent_vol - vol_mean) / vol_std
            return vol_z_score > self.volatility_regime_threshold
        return False
    
    def _detect_crash_or_spike(self, St: float, current_rock_ema: float, recent_vol: float) -> Tuple[bool, bool]:
        """Detects potential crash or spike conditions"""
        if recent_vol <= 1e-6 or St is None or current_rock_ema is None:
            return False, False
            
        # Calculate z-score of current price
        price_deviation = St - current_rock_ema
        price_z_score = price_deviation / recent_vol
        
        # Detect crash (price significantly below EMA)
        crash_detected = price_z_score < -self.crash_protection_threshold
        
        # Detect spike (price significantly above EMA)
        spike_detected = price_z_score > self.spike_protection_threshold
        
        return crash_detected, spike_detected
        
    def _calculate_bs_based_adjustment(self, St: float, TTE: float) -> float:
        """
        Uses BS pricing across different moneyness levels to estimate price trend
        Returns an adjustment factor (-1 to +1) to apply to fair value
        """
        if St is None or St <= 0 or TTE <= 1e-9:
            return 0.0
            
        # Calculate BS prices at different moneyness levels
        adjustment = 0.0
        valid_points = 0
        
        for m in self.bs_moneyness_range:
            # Convert moneyness to strike price
            K = St * math.exp(m * math.sqrt(TTE))
            
            # Calculate IV using volatility smile
            sigma = self._calculate_theoretical_iv(m)
            
            # Calculate BS price
            bs_price = black_scholes_call_price_approx(St, K, TTE, sigma)
            
            if bs_price is not None:
                # Calculate delta to understand directional exposure
                delta = black_scholes_delta_call_approx(St, K, TTE, sigma)
                
                if delta is not None:
                    # Delta ranges from 0 to 1, remap to -1 to +1 for symmetric impact
                    delta_centered = (delta - 0.5) * 2
                    
                    # Weight contribution by position in moneyness_range
                    # Center points count more
                    weight = 1.0 - abs(m) / 0.5  # Higher weight for central points
                    
                    adjustment += delta_centered * weight
                    valid_points += 1
        
        if valid_points > 0:
            adjustment /= valid_points
            
        # Limit adjustment to -1.0 to 1.0 range
        return max(-1.0, min(1.0, adjustment)) * self.bs_adjustment_strength

    # --- Order Generation Functions ---
    # UNCHANGED: Voucher Orders Function (as per instructions)
    def _generate_voucher_orders(self, state: TradingState, St: float, TTE: float, rock_price_deviation: Optional[float]) -> Tuple[Dict[Symbol, List[Order]], float]:
        """
        Generates orders for vouchers using mean reversion around the theoretical
        Black-Scholes price, incorporating volatility smile and adjustments.
        """
        voucher_orders: Dict[Symbol, List[Order]] = {}
        net_delta_exposure = 0.0

        if St is None or St <= 0 or TTE <= 1e-9:
            return voucher_orders, net_delta_exposure

        # --- Calculate Adjustment Factors for BS strategy ---
        buy_threshold_adjust = 1.0
        sell_threshold_adjust = 1.0
        size_adjustment_factor = 1.0
        iv_adjustment_factor = 1.0 # Factor to adjust theoretical IV

        if rock_price_deviation is not None:
            # Adjust thresholds, size, and IV based on underlying deviation
            threshold_impact = rock_price_deviation * self.voucher_aggressiveness * 0.5
            buy_threshold_adjust = max(0.5, min(1.5, 1.0 + threshold_impact))
            sell_threshold_adjust = max(0.5, min(1.5, 1.0 - threshold_impact))
            size_adjustment_factor = max(0.5, min(1.5, 1.0 - rock_price_deviation * self.voucher_aggressiveness))
            iv_adjustment = 1.0 - rock_price_deviation * self.iv_adjustment_sensitivity
            iv_adjustment_factor = max(0.7, min(1.3, iv_adjustment))

        for symbol in self.voucher_symbols:
            orders_for_symbol: List[Order] = []
            K = self.voucher_strikes[symbol]
            moneyness_m = self._calculate_mt(K, St, TTE)
            if moneyness_m is None:
                continue

            # --- Black-Scholes Calculation (This is the 'Mean' for reversion) ---
            sigma_theo = self._calculate_theoretical_iv(moneyness_m) # Uses vol smile
            sigma_adjusted = sigma_theo * iv_adjustment_factor # Apply adjustment
            bs_theoretical_price = black_scholes_call_price_approx(St, K, TTE, sigma_adjusted, self.risk_free_rate)
            if bs_theoretical_price is None:
                continue

            # --- Get Market Data ---
            order_depth = state.order_depths.get(symbol, OrderDepth())
            best_bid = self._get_best_bid(symbol, order_depth)
            best_ask = self._get_best_ask(symbol, order_depth)
            current_position = state.position.get(symbol, 0)
            position_limit = self.position_limits[symbol]

            # --- Calculate Effective BS Price Thresholds (Entry points for mean reversion) ---
            effective_buy_threshold_pct = self.voucher_trade_threshold_base_pct * buy_threshold_adjust
            effective_sell_threshold_pct = self.voucher_trade_threshold_base_pct * sell_threshold_adjust
            # Buy if market ask is below this price (BS Theo - % Threshold)
            bs_buy_entry_price = bs_theoretical_price * (1.0 - effective_buy_threshold_pct)
            # Sell if market bid is above this price (BS Theo + % Threshold)
            bs_sell_entry_price = bs_theoretical_price * (1.0 + effective_sell_threshold_pct)

            log_entry = {
                 "event": "VOUCHER_EVAL", "symbol": symbol, "St": round(St), "K": K, "TTE": round(TTE, 5),
                 "m": round(moneyness_m, 3), "sigma_adj": round(sigma_adjusted, 4),
                 "bs_theo_px": round(bs_theoretical_price, 2), # This is the mean
                 "bid": best_bid, "ask": best_ask,
                 "bs_buy_entry": round(bs_buy_entry_price, 2), # Buy threshold
                 "bs_sell_entry": round(bs_sell_entry_price, 2), # Sell threshold
                 "current_pos": current_position
            }
            # logger.print(log_entry) # Optional: Reduce logging verbosity

            # --- Trading Logic ---
            # BS Mean Reversion Buy Logic
            if best_ask is not None and best_ask < bs_buy_entry_price:
                mispricing = bs_theoretical_price - best_ask # How far below the mean is the ask?
                # Scale size based on magnitude of mispricing relative to threshold width
                threshold_width = bs_theoretical_price * effective_buy_threshold_pct + 1e-9
                size_factor = max(0.1, min(2.0, (mispricing / threshold_width)))
                buy_size_modifier = size_adjustment_factor # Adjust based on rock price deviation
                quantity = math.floor(self.voucher_order_size * buy_size_modifier * size_factor)
                quantity = max(1, quantity)
                available_buy_limit = position_limit - current_position
                order_qty = min(quantity, available_buy_limit)
                vol_at_ask = abs(order_depth.sell_orders.get(best_ask, 0))
                order_qty = min(order_qty, vol_at_ask)
                if order_qty > 0:
                    orders_for_symbol.append(Order(symbol, best_ask, order_qty))
                    logger.print({"event":"VOUCHER_ORDER_GEN", "reason": "BS_MR_BUY", "symbol":symbol, "price": best_ask, "qty": order_qty, "bs_theo": round(bs_theoretical_price,2), "misprice": round(mispricing,2)})

            # BS Mean Reversion Sell Logic
            elif best_bid is not None and best_bid > bs_sell_entry_price: # Use elif to avoid placing buy and sell in same iteration
                mispricing = best_bid - bs_theoretical_price # How far above the mean is the bid?
                # Scale size based on magnitude of mispricing relative to threshold width
                threshold_width = bs_theoretical_price * effective_sell_threshold_pct + 1e-9
                size_factor = max(0.1, min(2.0, (mispricing / threshold_width)))
                sell_size_modifier = (1.0 / size_adjustment_factor) if size_adjustment_factor > 1e-6 else 1.0 # Inverse adjustment for selling
                quantity = math.floor(self.voucher_order_size * sell_size_modifier * size_factor)
                quantity = max(1, quantity)
                available_sell_limit = position_limit + current_position
                order_qty = min(quantity, available_sell_limit)
                vol_at_bid = abs(order_depth.buy_orders.get(best_bid, 0))
                order_qty = min(order_qty, vol_at_bid)
                if order_qty > 0:
                   orders_for_symbol.append(Order(symbol, best_bid, -order_qty))
                   logger.print({"event":"VOUCHER_ORDER_GEN", "reason": "BS_MR_SELL", "symbol":symbol, "price": best_bid, "qty": -order_qty, "bs_theo": round(bs_theoretical_price,2), "misprice": round(mispricing,2)})

            if orders_for_symbol:
                voucher_orders[symbol] = orders_for_symbol
                # Delta calculation remains the same, based on final position and adjusted sigma
                final_position = current_position
                for order in orders_for_symbol: final_position += order.quantity
                delta = black_scholes_delta_call_approx(St, K, TTE, sigma_adjusted, self.risk_free_rate)
                if delta is not None:
                    position_delta = final_position * delta
                    net_delta_exposure += position_delta

        return voucher_orders, net_delta_exposure

    # Rock Market Making (unchanged logic, only parameters optimized)
    def _generate_rock_market_making_orders(self, state: TradingState, St: float, current_rock_ema: Optional[float], TTE: float) -> List[Order]:
        """Generates market making orders for VOLCANIC_ROCK with enhanced strategies."""
        mm_orders: List[Order] = []
        symbol = self.underlying_symbol

        if St is None:
            if current_rock_ema is None:
                logger.print({"event":"ROCK_MM_SKIP", "reason":"Missing St and EMA"})
                return mm_orders
            else:
                fair_value = current_rock_ema
                logger.print({"event":"ROCK_MM_WARN", "reason":"Using EMA as fair value"})
        else:
            # Use EMA as the fair value anchor for MM
            fair_value = current_rock_ema if current_rock_ema is not None else St
        
        # Original fair value before adjustments
        original_fair_value = fair_value
        
        # Calculate volatility for dynamic spread
        recent_vol = self._calculate_rock_volatility(St)
        
        # --- Base IV Analysis ---
        # Get current base IV and update history
        current_base_iv = self._estimate_base_iv()
        base_iv_ema = self._update_base_iv_history(current_base_iv)
        
        # Detect IV trend (is IV decreasing, which is often bullish for price?)
        iv_adjustment = 0.0
        if base_iv_ema is not None and len(self.base_iv_history) >= self.base_iv_window // 2:
            iv_deviation = current_base_iv - base_iv_ema
            if abs(base_iv_ema) > 1e-6:
                iv_pct_change = iv_deviation / base_iv_ema
                # Negative correlation: IV down → Price up and vice versa
                iv_signal = -iv_pct_change
                iv_adjustment = fair_value * iv_signal * self.iv_price_correlation_strength
                
                logger.print({
                    "event": "IV_ANALYSIS", 
                    "base_iv": round(current_base_iv, 4), 
                    "iv_ema": round(base_iv_ema, 4), 
                    "iv_pct_chg": round(iv_pct_change, 4),
                    "iv_signal": round(iv_signal, 4),
                    "adjustment": round(iv_adjustment, 2)
                })
                
                # Apply adjustment to fair value
                fair_value += iv_adjustment
        
        # --- BS-based Price Adjustment ---
        bs_adjustment = self._calculate_bs_based_adjustment(St, TTE)
        bs_adjustment_amount = fair_value * bs_adjustment * 0.01  # Max 1% adjustment
        fair_value += bs_adjustment_amount
        
        logger.print({
            "event": "BS_ADJUSTMENT",
            "signal": round(bs_adjustment, 4),
            "adjustment": round(bs_adjustment_amount, 2)
        })
        
        # --- Crash/Spike Detection ---
        crash_detected, spike_detected = False, False
        if St is not None and current_rock_ema is not None and recent_vol > 1e-6:
            crash_detected, spike_detected = self._detect_crash_or_spike(St, current_rock_ema, recent_vol)
            
        # --- Volatility Regime Detection ---
        high_vol_regime = self._detect_volatility_regime(recent_vol)
        
        # --- Dynamic Spread and Size Calculation ---
        # Base spread calculation
        spread = max(self.rock_mm_min_spread, recent_vol * self.rock_mm_spread_multiplier)
        
        # Adjust spread for crash/spike protection
        if crash_detected or spike_detected:
            spread *= self.crash_protection_spread_multiplier
            logger.print({
                "event": "CRASH_SPIKE_DETECTED",
                "crash": crash_detected,
                "spike": spike_detected,
                "spread_multiplier": self.crash_protection_spread_multiplier
            })
        
        # Adjust spread for volatility regime
        if high_vol_regime:
            spread *= 1.3  # Increase spread in high volatility regimes
            logger.print({
                "event": "HIGH_VOL_REGIME",
                "spread_multiplier": 1.3
            })
        
        # Calculate bid and ask prices
        our_bid = math.floor(fair_value - spread / 2.0)
        our_ask = math.ceil(fair_value + spread / 2.0)
        
        # Get market data
        order_depth = state.order_depths.get(symbol, OrderDepth())
        current_position = state.position.get(symbol, 0)
        position_limit = self.position_limits[symbol]
        
        # --- Inventory Management ---
        max_pos = position_limit
        min_pos = -position_limit
        inventory_pct = (current_position - min_pos) / (max_pos - min_pos) if (max_pos - min_pos) > 0 else 0.5
        
        # Skew prices based on inventory (higher bid if inventory low, higher ask if inventory high)
        inventory_skew = (0.5 - inventory_pct) * spread * 0.2
        our_bid += math.floor(inventory_skew)
        our_ask += math.floor(inventory_skew)
        
        # Base order size calculation
        base_size = self.rock_mm_base_order_size
        
        # Reduce size during crash/spike
        if crash_detected or spike_detected:
            base_size *= self.crash_protection_size_reducer
        
        # Reduce size in high volatility regimes
        if high_vol_regime:
            base_size *= 0.8
        
        # Skew size based on inventory
        buy_size_skew = max(0.5, min(1.5, 1.0 - (inventory_pct - 0.5) * 2 * self.rock_mm_inventory_skew_factor))
        sell_size_skew = max(0.5, min(1.5, 1.0 + (inventory_pct - 0.5) * 2 * self.rock_mm_inventory_skew_factor))
        
        # Special case handling for crash/spike
        if crash_detected:
            # During crash: Reduce buy size, increase sell size (to capture recovery)
            buy_size_skew *= 0.7
            # Only buy below EMA minus threshold
            if St is not None and current_rock_ema is not None:
                crash_buy_threshold = current_rock_ema - recent_vol * 1.0
                our_bid = min(our_bid, math.floor(crash_buy_threshold))
            
        if spike_detected:
            # During spike: Reduce sell size, increase buy size (to capture reversion)
            sell_size_skew *= 0.7
            # Only sell above EMA plus threshold
            if St is not None and current_rock_ema is not None:
                spike_sell_threshold = current_rock_ema + recent_vol * 1.0
                our_ask = max(our_ask, math.ceil(spike_sell_threshold))
        
        # Final size calculations
        buy_order_size = max(1, int(round(base_size * buy_size_skew)))
        sell_order_size = max(1, int(round(base_size * sell_size_skew)))
        
        # --- Place Orders ---
        # Buy Order
        available_buy_limit = position_limit - current_position
        final_buy_qty = min(buy_order_size, available_buy_limit)
        if final_buy_qty > 0:
            mm_orders.append(Order(symbol, our_bid, final_buy_qty))
        
        # Sell Order
        available_sell_limit = position_limit + current_position
        final_sell_qty = min(sell_order_size, available_sell_limit)
        if final_sell_qty > 0:
            mm_orders.append(Order(symbol, our_ask, -final_sell_qty))
        
        if mm_orders:
            logger.print({
                "event": "ROCK_MM_ORDERS",
                "orig_fv": round(original_fair_value, 1),
                "adj_fv": round(fair_value, 1),
                "iv_adj": round(iv_adjustment, 2),
                "bs_adj": round(bs_adjustment_amount, 2),
                "bid": our_bid,
                "ask": our_ask,
                "spread": round(spread, 1),
                "buy_sz": final_buy_qty,
                "sell_sz": -final_sell_qty,
                "inventory": current_position,
                "inv_pct": round(inventory_pct, 2),
                "crash": crash_detected,
                "spike": spike_detected,
                "high_vol": high_vol_regime
            })
        
        return mm_orders

    # --- Main Run Method (unchanged) ---
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data_json = "" # No persistent state needed

        # --- Calculations ---
        timestamp = state.timestamp
        TTE = self._calculate_tte(timestamp)
        rock_order_depth = state.order_depths.get(self.underlying_symbol, OrderDepth())
        St = self._get_mid_price(self.underlying_symbol, rock_order_depth)
        current_rock_ema = self._calculate_rock_ema(St) # Also updates self.rock_price_history_ema

        # --- Determine Rock Price Deviation (for voucher BS strategy adjustments) ---
        rock_price_deviation_pct = None # Relative deviation
        if St is not None and self.current_rock_ema is not None and abs(self.current_rock_ema) > 1e-6:
            rock_price_deviation_pct = (St - self.current_rock_ema) / self.current_rock_ema

        # --- Generate Voucher Orders (UNCHANGED BS Mean Reversion Strategy) ---
        #voucher_orders_dict, _net_delta_exposure = self._generate_voucher_orders(state, St, TTE, rock_price_deviation_pct)
        #result.update(voucher_orders_dict)

        # --- Generate Volcanic Rock Orders (NEW Enhanced Market Making Strategy) ---
        rock_orders_list = self._generate_rock_market_making_orders(state, St, current_rock_ema, TTE)
        if rock_orders_list:
            result[self.underlying_symbol] = rock_orders_list

        # --- Prepare Trader Data & Return ---
        logger.flush(state, result, conversions, trader_data_json)
        return result, conversions, trader_data_json
