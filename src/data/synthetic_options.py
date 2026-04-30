"""
Synthetic options chain generator.

Produces a Black-Scholes-consistent chain of calls and puts with a realistic
skew/smile and term structure. Used by the dashboard when the live options
feed is unavailable, and as a known-good input for surface construction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.pricing.black_scholes import BlackScholesModel, OptionGreeks
from src.pricing.implied_vol import ImpliedVolatilityCalculator
from src.data.price_provider import RealTimePriceProvider

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Symbol-level vol characteristics
# ----------------------------------------------------------------------

# Surface-shape parameters used by create_chain. base_vol is the ATM 30-day
# IV; skew is the slope of IV vs. log-moneyness; smile is the curvature.
_SURFACE_PARAMS: Dict[str, Dict[str, float]] = {
    'AAPL':  {'base_vol': 0.25, 'volume_mult':  800, 'skew': -0.08, 'smile': 0.030},
    'MSFT':  {'base_vol': 0.22, 'volume_mult':  600, 'skew': -0.06, 'smile': 0.020},
    'GOOGL': {'base_vol': 0.28, 'volume_mult':  400, 'skew': -0.07, 'smile': 0.025},
    'NVDA':  {'base_vol': 0.40, 'volume_mult': 1000, 'skew': -0.10, 'smile': 0.040},
    'TSLA':  {'base_vol': 0.50, 'volume_mult': 1200, 'skew': -0.12, 'smile': 0.050},
    'SPY':   {'base_vol': 0.15, 'volume_mult': 2000, 'skew': -0.05, 'smile': 0.020},
    'QQQ':   {'base_vol': 0.20, 'volume_mult': 1500, 'skew': -0.06, 'smile': 0.025},
    'META':  {'base_vol': 0.35, 'volume_mult':  700, 'skew': -0.09, 'smile': 0.035},
    'AMZN':  {'base_vol': 0.30, 'volume_mult':  600, 'skew': -0.08, 'smile': 0.030},
    'PLTR':  {'base_vol': 0.75, 'volume_mult':  500, 'skew': -0.15, 'smile': 0.060},
}
_DEFAULT_SURFACE = {'base_vol': 0.25, 'volume_mult': 200, 'skew': -0.08, 'smile': 0.03}


# Vol-regime profile used by Greek calculations and fallback paths.
def get_symbol_vol_characteristics(symbol: str) -> Dict[str, float]:
    sym = symbol.upper()
    if sym in {'PLTR', 'GME', 'RBLX'}:
        return {'base_vol': 0.75, 'skew_strength': -0.12, 'smile_curvature': 0.08,
                'term_structure': 0.15, 'vol_clustering': 0.12}
    if sym in {'TSLA', 'NVDA', 'AMD'}:
        return {'base_vol': 0.55, 'skew_strength': -0.10, 'smile_curvature': 0.06,
                'term_structure': 0.12, 'vol_clustering': 0.10}
    if sym in {'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'}:
        return {'base_vol': 0.28, 'skew_strength': -0.08, 'smile_curvature': 0.03,
                'term_structure': 0.08, 'vol_clustering': 0.06}
    if sym in {'SPY', 'QQQ', 'IWM', 'VTI'}:
        return {'base_vol': 0.18, 'skew_strength': -0.05, 'smile_curvature': 0.02,
                'term_structure': 0.05, 'vol_clustering': 0.04}
    if sym in {'JPM', 'BAC', 'WFC', 'GS'}:
        return {'base_vol': 0.35, 'skew_strength': -0.06, 'smile_curvature': 0.03,
                'term_structure': 0.08, 'vol_clustering': 0.08}
    return {'base_vol': 0.30, 'skew_strength': -0.08, 'smile_curvature': 0.03,
            'term_structure': 0.08, 'vol_clustering': 0.08}


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------

class SyntheticOptionsGenerator:
    """Build BS-consistent options chains and compute representative Greeks."""

    RISK_FREE_RATE = 0.05

    def __init__(self, price_provider: RealTimePriceProvider):
        self.price_provider = price_provider
        self.bs = BlackScholesModel()
        self.iv_calc = ImpliedVolatilityCalculator()

    # ------------------------------------------------------------------
    # Chain generation
    # ------------------------------------------------------------------

    def create_chain(self, symbol: str) -> pd.DataFrame:
        """Generate a synthetic options chain anchored to the live spot price."""
        spot = self.price_provider.get_live_price(symbol)
        params = _SURFACE_PARAMS.get(symbol.upper(), _DEFAULT_SURFACE)

        strikes = self._build_strikes(spot)
        expirations = self._build_expirations()

        rows = []
        today = datetime.now()
        for exp in expirations:
            T = (exp - today).days / 365.0
            for K in strikes:
                vol = self._iv_at(spot, K, T, params)
                call = max(0.01, self.bs.call_price(S=spot, K=K, T=T, r=self.RISK_FREE_RATE, sigma=vol))
                put = max(0.01, self.bs.put_price(S=spot, K=K, T=T, r=self.RISK_FREE_RATE, sigma=vol))
                base_volume = self._base_volume(spot, K, T, params['volume_mult'])
                rows.extend(self._option_rows(
                    symbol=symbol, exp=exp, spot=spot, strike=K, T=T,
                    vol=vol, call_price=call, put_price=put,
                    base_volume=base_volume, today=today,
                ))

        df = pd.DataFrame(rows)
        df['expiration'] = pd.to_datetime(df['expiration'])
        today_date = datetime.now().date()
        df['daysToExpiration'] = df['expiration'].apply(lambda d: (d.date() - today_date).days)
        df['bidAskSpread'] = df['ask'] - df['bid']
        df['bidAskSpreadPct'] = df['bidAskSpread'] / ((df['bid'] + df['ask']) / 2)
        return df

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def calculate_greeks(self, symbol: str, spot_price: float) -> Dict[str, float]:
        """Compute representative ATM Greeks at 30 DTE for ``symbol``."""
        characteristics = get_symbol_vol_characteristics(symbol)
        base_iv = characteristics['base_vol']
        iv_60d = base_iv * (1.0 + 0.05 * np.sqrt(60 / 30))
        iv_90d = base_iv * (1.0 + 0.08 * np.sqrt(90 / 30))
        T = 30 / 365
        r = self.RISK_FREE_RATE

        try:
            delta_call = OptionGreeks.delta(spot_price, spot_price, T, r, base_iv, 'call')
            delta_put = OptionGreeks.delta(spot_price, spot_price, T, r, base_iv, 'put')
            gamma = OptionGreeks.gamma(spot_price, spot_price, T, r, base_iv)
            theta_call = OptionGreeks.theta(spot_price, spot_price, T, r, base_iv, 'call')
            theta_put = OptionGreeks.theta(spot_price, spot_price, T, r, base_iv, 'put')
            vega = OptionGreeks.vega(spot_price, spot_price, T, r, base_iv)

            if not (0.01 <= vega <= 100):
                manual = self._vega_manual(spot_price, spot_price, T, r, base_iv)
                if 0.01 <= manual <= 100:
                    vega = manual

            return {
                'iv_30d': base_iv, 'iv_60d': iv_60d, 'iv_90d': iv_90d,
                'delta': delta_call, 'gamma': gamma,
                'theta': (theta_call + theta_put) / 2, 'vega': vega,
            }
        except Exception as e:
            logger.warning(f"Greeks calculation failed for {symbol}: {e}")
            return self._fallback_greeks(symbol, base_iv, iv_60d, iv_90d)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_strikes(spot: float, n: int = 25) -> list:
        if spot < 20:
            spacing = 1
        elif spot < 100:
            spacing = 2.5
        elif spot < 200:
            spacing = 5
        else:
            spacing = 10
        return [spot + i * spacing for i in range(-n // 2, n // 2 + 1) if spot + i * spacing > 0]

    @staticmethod
    def _build_expirations() -> list:
        """Generate weekly Fridays for 4 weeks plus monthly third-Fridays for 6 months."""
        today = datetime.now()
        out = []
        for i in range(1, 5):
            exp = today + timedelta(weeks=i)
            exp += timedelta(days=(4 - exp.weekday()) % 7)
            out.append(exp)
        for i in range(1, 7):
            exp = (today + timedelta(days=30 * i)).replace(day=15)
            exp += timedelta(days=(4 - exp.weekday()) % 7)
            out.append(exp)
        return out

    @staticmethod
    def _iv_at(spot: float, strike: float, T: float, params: Dict[str, float]) -> float:
        """Build IV for one (strike, T) point: term structure + skew + smile + noise.

        Uses standard log-moneyness ``log(K/S)``:
          - OTM puts have log(K/S) < 0; with negative skew this lifts their IV.
          - OTM calls have log(K/S) > 0; with negative skew this lowers their IV.
        Smile is symmetric in log-moneyness, lifting the wings on both sides.
        """
        log_money = np.log(strike / spot)
        iv = params['base_vol'] * (1.0 + 0.1 * np.sqrt(T))
        iv += params['skew'] * log_money
        iv += params['smile'] * log_money ** 2
        # Noise scaled to base vol so low-vol names (SPY) don't drown their skew.
        iv += np.random.normal(0, 0.005 * params['base_vol'])
        return float(np.clip(iv, 0.05, 2.5))

    @staticmethod
    def _base_volume(spot: float, strike: float, T: float, mult: float) -> int:
        moneyness = spot / strike
        factor = max(0.1, 1.5 - abs(1 - moneyness))
        return max(1, int(mult * factor * np.exp(-T * 1.2)))

    @staticmethod
    def _option_rows(*, symbol, exp, spot, strike, T, vol, call_price, put_price,
                     base_volume, today):
        moneyness = spot / strike
        rows = []
        for opt_type, price, vol_factor, oi_factor in (
            ('call', call_price, 1.0, 5),
            ('put',  put_price,  0.8, 4),
        ):
            bid = round(price * 0.995, 2)
            ask = round(price * 1.005, 2)
            rows.append({
                'symbol': symbol,
                'contractSymbol': f"{symbol}{exp.strftime('%y%m%d')}{opt_type[0].upper()}{int(strike * 1000):08d}",
                'strike': strike,
                'expiration': exp,
                'type': opt_type,
                'bid': bid,
                'ask': ask,
                'last': round(price, 2),
                'volume': int(base_volume * vol_factor),
                'openInterest': int(base_volume * oi_factor),
                'impliedVolatility': round(vol, 4),
                'timestamp': today.strftime('%Y-%m-%d'),
                'moneyness': moneyness,
                'time_to_expiry': T,
            })
        return rows

    @staticmethod
    def _vega_manual(S: float, K: float, T: float, r: float, sigma: float) -> float:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return max(0.0, S * norm.pdf(d1) * np.sqrt(T) / 100)

    @staticmethod
    def _fallback_greeks(symbol: str, iv_30d: float, iv_60d: float, iv_90d: float) -> Dict[str, float]:
        sym = symbol.upper()
        vol_factor = iv_30d / 0.25
        if sym in {'PLTR', 'GME', 'TSLA'}:
            return {
                'iv_30d': iv_30d, 'iv_60d': iv_60d, 'iv_90d': iv_90d,
                'delta': float(np.random.uniform(0.45, 0.65)),
                'gamma': float(np.random.uniform(0.015, 0.030) * vol_factor),
                'theta': float(-np.random.uniform(0.15, 0.30) * vol_factor),
                'vega':  float(np.random.uniform(0.30, 0.60) * np.sqrt(vol_factor)),
            }
        return {
            'iv_30d': iv_30d, 'iv_60d': iv_60d, 'iv_90d': iv_90d,
            'delta': float(np.random.uniform(0.40, 0.60)),
            'gamma': float(np.random.uniform(0.010, 0.020) * vol_factor),
            'theta': float(-np.random.uniform(0.05, 0.15) * vol_factor),
            'vega':  float(np.random.uniform(0.15, 0.35) * np.sqrt(vol_factor)),
        }
