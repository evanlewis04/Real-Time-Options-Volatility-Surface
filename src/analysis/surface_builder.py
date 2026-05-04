"""
Volatility-surface construction helpers.

Cleans an options chain for surface fitting, then attempts (in order):
1. The library VolatilitySurface class.
2. Manual RBF / linear interpolation onto a regular (T, moneyness) grid.
3. A simple per-(strike, expiry) bucket fill.
4. A parametric fallback grid built from per-symbol vol characteristics.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.analysis.vol_surface import VolatilitySurface
from src.data.synthetic_options import get_symbol_vol_characteristics

logger = logging.getLogger(__name__)

DEFAULT_RISK_FREE_RATE = 0.05


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def build_surface(options_data: pd.DataFrame, spot_price: float, symbol: str,
                  risk_free_rate: float | None = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a vol surface for ``symbol`` from ``options_data``.

    Returns ``(strikes, expiries_in_days, vols)`` arrays. ``strikes`` and
    ``expiries`` are 2D meshes when produced by interpolation, 1D otherwise.
    Falls through to a parametric fallback if every fit attempt fails.
    """
    if options_data.empty:
        return _parametric_fallback(symbol, spot_price)

    clean = _prepare_data(options_data, spot_price)
    if clean.empty:
        return _parametric_fallback(symbol, spot_price)

    try:
        surface = VolatilitySurface(clean, spot_price, _surface_rate(clean, risk_free_rate))
        result = surface.construct_surface(method='linear')
        if 'combined' in result:
            data = result['combined']
            strikes = data['strikes']
            times = data['times'] * 365
            vols = data['implied_vols']
            if _passes_quality_check(vols):
                return strikes, times, vols
    except Exception as e:
        logger.warning(f"VolatilitySurface fit failed: {e}")

    try:
        strikes, times, vols = _interpolate_grid(clean, spot_price)
        if _passes_quality_check(vols):
            return strikes, times, vols
    except Exception as e:
        logger.warning(f"Manual interpolation failed: {e}")

    try:
        return _bucket_fill(clean)
    except Exception as e:
        logger.warning(f"Bucket-fill construction failed: {e}")

    return _parametric_fallback(symbol, spot_price)


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def _prepare_data(options_data: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """Filter outliers, attach moneyness, and IQR-trim per expiration bucket."""
    required = {'strike', 'impliedVolatility', 'time_to_expiry', 'type'}
    if not required.issubset(options_data.columns):
        logger.error(f"Missing columns: {required - set(options_data.columns)}")
        return pd.DataFrame()

    df = options_data.copy()
    if 'moneyness' not in df.columns:
        df['moneyness'] = df['strike'] / spot_price

    df = df[
        (df['impliedVolatility'] > 0.02) & (df['impliedVolatility'] < 3.0) &
        (df['time_to_expiry'] > 0.003) & (df['time_to_expiry'] < 3.0) &
        (df['moneyness'] > 0.3) & (df['moneyness'] < 3.0) &
        (df['strike'] > 0)
    ]

    df['exp_group'] = (df['time_to_expiry'] * 365).round(0)
    cleaned = []
    for _, group in df.groupby('exp_group'):
        if len(group) < 3:
            continue
        q1 = group['impliedVolatility'].quantile(0.25)
        q3 = group['impliedVolatility'].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        kept = group[(group['impliedVolatility'] >= lo) & (group['impliedVolatility'] <= hi)]
        if len(kept) >= 3:
            cleaned.append(kept)

    if cleaned:
        df = pd.concat(cleaned, ignore_index=True)

    df = df.sort_values(['time_to_expiry', 'moneyness'])
    return df.drop('exp_group', axis=1, errors='ignore')


def _surface_rate(clean: pd.DataFrame, explicit_rate: float | None = None) -> float:
    """Select the rate used by model-backed surface construction."""
    if explicit_rate is not None and np.isfinite(explicit_rate):
        return float(explicit_rate)
    if "riskFreeRate" in clean.columns:
        rates = pd.to_numeric(clean["riskFreeRate"], errors="coerce").dropna()
        if not rates.empty:
            return float(rates.median())
    return DEFAULT_RISK_FREE_RATE


# ----------------------------------------------------------------------
# Construction strategies
# ----------------------------------------------------------------------

def _interpolate_grid(clean: pd.DataFrame, spot_price: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate IVs onto a 30x40 (T, moneyness) grid using RBF or linear."""
    from scipy.interpolate import griddata, RBFInterpolator

    times = clean['time_to_expiry'].values
    money = clean['moneyness'].values
    ivs = clean['impliedVolatility'].values

    t_grid = np.linspace(times.min(), times.max(), 30)
    m_grid = np.linspace(money.min(), money.max(), 40)
    T_mesh, M_mesh = np.meshgrid(t_grid, m_grid)
    strike_mesh = M_mesh * spot_price
    days_mesh = T_mesh * 365

    points = np.column_stack((times, money))
    grid_points = np.column_stack((T_mesh.ravel(), M_mesh.ravel()))

    iv_linear = griddata(points, ivs, grid_points, method='linear').reshape(T_mesh.shape)
    try:
        rbf = RBFInterpolator(points, ivs, kernel='thin_plate_spline', smoothing=0.1)
        iv_rbf = rbf(grid_points).reshape(T_mesh.shape)
        surface = iv_rbf if np.isnan(iv_rbf).sum() < np.isnan(iv_linear).sum() else iv_linear
    except (ImportError, Exception):
        surface = iv_linear

    if np.isnan(surface).any():
        nearest = griddata(points, ivs, grid_points, method='nearest').reshape(T_mesh.shape)
        surface = np.where(np.isnan(surface), nearest, surface)

    try:
        from scipy.ndimage import gaussian_filter
        surface = gaussian_filter(surface, sigma=0.5)
    except ImportError:
        pass

    surface = np.clip(surface, 0.05, 2.5)
    return strike_mesh, days_mesh, surface


def _bucket_fill(clean: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fall back to a per-(strike, expiry) IV mean grid with NN-fill for gaps."""
    strikes = sorted(clean['strike'].unique())
    expiries = sorted(clean['daysToExpiration'].unique())
    surface = np.full((len(expiries), len(strikes)), np.nan)

    for i, days in enumerate(expiries):
        for j, strike in enumerate(strikes):
            match = clean[
                (np.abs(clean['strike'] - strike) < 0.01) &
                (np.abs(clean['daysToExpiration'] - days) < 0.1)
            ]
            if not match.empty:
                surface[i, j] = match['impliedVolatility'].mean()

    _fill_nans(surface)
    return np.array(strikes), np.array(expiries), surface


def _parametric_fallback(symbol: str, spot_price: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic surface from per-symbol vol characteristics."""
    chars = get_symbol_vol_characteristics(symbol)
    base_vol = chars['base_vol']
    skew = chars['skew_strength']

    strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, 12)
    expiries = np.array([7, 14, 30, 60, 90, 180])
    surface = np.zeros((len(expiries), len(strikes)))

    for i, days in enumerate(expiries):
        T = days / 365.0
        term_adj = 1.0 + 0.1 * np.sqrt(T)
        for j, strike in enumerate(strikes):
            moneyness = strike / spot_price
            iv = base_vol * term_adj
            iv += skew * (moneyness - 1.0)
            iv += 0.03 * (moneyness - 1.0) ** 2
            surface[i, j] = max(0.05, iv)
    return strikes, expiries, surface


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _passes_quality_check(vols: np.ndarray) -> bool:
    if vols.size == 0 or np.all(np.isnan(vols)):
        return False
    if np.nanmin(vols) < 0.01 or np.nanmax(vols) > 5.0:
        return False
    valid_ratio = np.sum(~np.isnan(vols)) / vols.size
    return valid_ratio >= 0.5


def _fill_nans(surface: np.ndarray) -> None:
    """Fill NaN cells in-place using nearest-neighbor interpolation."""
    valid_mask = ~np.isnan(surface)
    if not valid_mask.any():
        surface[:] = 0.25
        return
    try:
        from scipy.interpolate import griddata
        coords = np.array([(i, j) for i in range(surface.shape[0])
                           for j in range(surface.shape[1]) if valid_mask[i, j]])
        values = surface[valid_mask]
        for i in range(surface.shape[0]):
            for j in range(surface.shape[1]):
                if np.isnan(surface[i, j]):
                    try:
                        surface[i, j] = griddata(coords, values, [(i, j)], method='nearest')[0]
                    except (ValueError, IndexError, TypeError):
                        surface[i, j] = 0.25
    except ImportError:
        surface[np.isnan(surface)] = 0.25
