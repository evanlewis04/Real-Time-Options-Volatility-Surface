"""Typed application settings with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ProviderSettings:
    """Market-data provider settings."""

    max_expirations: int = 8
    chain_cache_seconds: int = 300
    price_cache_seconds: int = 60
    max_quote_age_days: int = 5
    min_open_interest: int = 0
    min_volume: int = 0
    max_bid_ask_spread_pct: float = 1.5

    def __post_init__(self) -> None:
        _require_positive("max_expirations", self.max_expirations)
        _require_positive("chain_cache_seconds", self.chain_cache_seconds)
        _require_positive("price_cache_seconds", self.price_cache_seconds)
        _require_non_negative("max_quote_age_days", self.max_quote_age_days)
        _require_non_negative("min_open_interest", self.min_open_interest)
        _require_non_negative("min_volume", self.min_volume)
        _require_non_negative("max_bid_ask_spread_pct", self.max_bid_ask_spread_pct)


@dataclass(frozen=True)
class FitFilterSettings:
    """Surface-fitting row eligibility settings."""

    preset: str = "Standard"
    max_bid_ask_spread_pct: float = 0.75
    max_quote_age_days: int = 5
    min_volume: int = 0
    min_open_interest: int = 0
    moneyness_min: float = 0.50
    moneyness_max: float = 2.00
    max_raw_iv: float = 2.00
    no_arbitrage_policy: str = "exclude"
    last_only_policy: str = "allow_penalized"

    def __post_init__(self) -> None:
        _require_non_negative("max_bid_ask_spread_pct", self.max_bid_ask_spread_pct)
        _require_non_negative("max_quote_age_days", self.max_quote_age_days)
        _require_non_negative("min_volume", self.min_volume)
        _require_non_negative("min_open_interest", self.min_open_interest)
        _require_non_negative("moneyness_min", self.moneyness_min)
        _require_positive("moneyness_max", self.moneyness_max)
        _require_positive("max_raw_iv", self.max_raw_iv)
        if self.moneyness_min > self.moneyness_max:
            raise ValueError("moneyness_min must be less than or equal to moneyness_max")
        _require_choice("no_arbitrage_policy", self.no_arbitrage_policy, {"exclude", "penalize", "allow"})
        _require_choice("last_only_policy", self.last_only_policy, {"exclude", "allow_penalized", "allow"})


@dataclass(frozen=True)
class DemoSettings:
    """Deterministic demo-mode settings."""

    random_seed: int = 1729
    max_expirations: int = 8

    def __post_init__(self) -> None:
        _require_positive("max_expirations", self.max_expirations)


@dataclass(frozen=True)
class DashboardSettings:
    """Dashboard runtime settings."""

    update_interval_seconds: int = 30
    snapshot_dir: Path = Path("data/snapshots")

    def __post_init__(self) -> None:
        _require_positive("update_interval_seconds", self.update_interval_seconds)


@dataclass(frozen=True)
class LoggingSettings:
    """Logging settings."""

    level: str = "INFO"
    structured: bool = True
    log_file: Path = Path("volatility_system.log")

    def __post_init__(self) -> None:
        level = self.level.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"LOG_LEVEL must be a valid logging level, got {self.level!r}")


@dataclass(frozen=True)
class AppSettings:
    """Top-level application settings."""

    providers: ProviderSettings = field(default_factory=ProviderSettings)
    fit_filters: FitFilterSettings = field(default_factory=FitFilterSettings)
    demo: DemoSettings = field(default_factory=DemoSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


def load_app_settings(environ: dict[str, str] | None = None) -> AppSettings:
    """Load typed settings from ``VOL_SURFACE_*`` environment variables."""
    env = os.environ if environ is None else environ
    providers = ProviderSettings(
        max_expirations=_env(env, "VOL_SURFACE_MAX_EXPIRATIONS", int, ProviderSettings.max_expirations),
        chain_cache_seconds=_env(
            env,
            "VOL_SURFACE_CHAIN_CACHE_SECONDS",
            int,
            ProviderSettings.chain_cache_seconds,
        ),
        price_cache_seconds=_env(
            env,
            "VOL_SURFACE_PRICE_CACHE_SECONDS",
            int,
            ProviderSettings.price_cache_seconds,
        ),
        max_quote_age_days=_env(
            env,
            "VOL_SURFACE_MAX_QUOTE_AGE_DAYS",
            int,
            ProviderSettings.max_quote_age_days,
        ),
        min_open_interest=_env(env, "VOL_SURFACE_MIN_OPEN_INTEREST", int, ProviderSettings.min_open_interest),
        min_volume=_env(env, "VOL_SURFACE_MIN_VOLUME", int, ProviderSettings.min_volume),
        max_bid_ask_spread_pct=_env(
            env,
            "VOL_SURFACE_MAX_BID_ASK_SPREAD_PCT",
            float,
            ProviderSettings.max_bid_ask_spread_pct,
        ),
    )
    fit_filters = FitFilterSettings(
        preset=_env(env, "VOL_SURFACE_FIT_PRESET", str, FitFilterSettings.preset),
        max_bid_ask_spread_pct=_env(
            env,
            "VOL_SURFACE_FIT_MAX_BID_ASK_SPREAD_PCT",
            float,
            FitFilterSettings.max_bid_ask_spread_pct,
        ),
        max_quote_age_days=_env(
            env,
            "VOL_SURFACE_FIT_MAX_QUOTE_AGE_DAYS",
            int,
            FitFilterSettings.max_quote_age_days,
        ),
        min_volume=_env(env, "VOL_SURFACE_FIT_MIN_VOLUME", int, FitFilterSettings.min_volume),
        min_open_interest=_env(
            env,
            "VOL_SURFACE_FIT_MIN_OPEN_INTEREST",
            int,
            FitFilterSettings.min_open_interest,
        ),
        moneyness_min=_env(env, "VOL_SURFACE_FIT_MONEYNESS_MIN", float, FitFilterSettings.moneyness_min),
        moneyness_max=_env(env, "VOL_SURFACE_FIT_MONEYNESS_MAX", float, FitFilterSettings.moneyness_max),
        max_raw_iv=_env(env, "VOL_SURFACE_FIT_MAX_RAW_IV", float, FitFilterSettings.max_raw_iv),
        no_arbitrage_policy=_env(
            env,
            "VOL_SURFACE_FIT_NO_ARBITRAGE_POLICY",
            str,
            FitFilterSettings.no_arbitrage_policy,
        ),
        last_only_policy=_env(
            env,
            "VOL_SURFACE_FIT_LAST_ONLY_POLICY",
            str,
            FitFilterSettings.last_only_policy,
        ),
    )
    demo = DemoSettings(
        random_seed=_env(env, "VOL_SURFACE_DEMO_RANDOM_SEED", int, DemoSettings.random_seed),
        max_expirations=_env(env, "VOL_SURFACE_DEMO_MAX_EXPIRATIONS", int, DemoSettings.max_expirations),
    )
    dashboard = DashboardSettings(
        update_interval_seconds=_env(
            env,
            "VOL_SURFACE_UPDATE_INTERVAL_SECONDS",
            int,
            DashboardSettings.update_interval_seconds,
        ),
        snapshot_dir=Path(_env(env, "VOL_SURFACE_SNAPSHOT_DIR", str, str(DashboardSettings.snapshot_dir))),
    )
    logging = LoggingSettings(
        level=_env(env, "VOL_SURFACE_LOG_LEVEL", str, LoggingSettings.level),
        structured=_env_bool(env, "VOL_SURFACE_STRUCTURED_LOGS", LoggingSettings.structured),
        log_file=Path(_env(env, "VOL_SURFACE_LOG_FILE", str, str(LoggingSettings.log_file))),
    )
    return AppSettings(
        providers=providers,
        fit_filters=fit_filters,
        demo=demo,
        dashboard=dashboard,
        logging=logging,
    )


def _env(env: dict[str, str], key: str, caster: Callable[[str], T], default: T) -> T:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return caster(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} has invalid value {raw!r}") from exc


def _env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} has invalid boolean value {raw!r}")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
