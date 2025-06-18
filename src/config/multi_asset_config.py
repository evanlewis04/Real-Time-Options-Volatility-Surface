"""
Multi-Asset Configuration Management System
Centralized configuration for the Week 2 multi-asset volatility system
"""

import os
import yaml
import json
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AssetConfiguration:
    """Configuration for individual assets"""
    symbol: str
    enabled: bool = True
    priority: str = "medium"  # low, medium, high, critical
    update_frequency: int = 60  # seconds between updates
    
    # Data quality filters
    min_volume: int = 10
    max_spread: float = 0.10
    min_open_interest: int = 5
    min_time_to_expiry: int = 1  # days
    max_time_to_expiry: int = 365  # days
    
    # Risk management
    max_weight: float = 0.25  # maximum portfolio weight
    var_multiplier: float = 2.0  # VaR calculation multiplier
    
    # Sector classification
    sector: str = "other"
    sub_sector: str = "general"
    
    # Trading hours (if different from market hours)
    trading_start: Optional[time] = None
    trading_end: Optional[time] = None


@dataclass
class SystemConfiguration:
    """System-wide configuration"""
    # API settings
    api_rate_limit: int = 5  # calls per second
    api_timeout: int = 30  # seconds
    api_retry_attempts: int = 3
    api_retry_delay: float = 1.0  # seconds
    
    # Processing settings
    max_worker_threads: int = 10
    processing_timeout: int = 60  # seconds
    queue_max_size: int = 1000
    
    # Real-time settings
    real_time_enabled: bool = True
    update_batch_size: int = 5  # assets to update in parallel
    health_check_interval: int = 30  # seconds
    
    # Caching settings
    cache_enabled: bool = True
    cache_duration: int = 300  # seconds
    max_cache_size: int = 1000  # number of cached items
    
    # Logging settings
    log_level: str = "INFO"
    log_file_rotation: bool = True
    log_max_size: str = "100MB"
    log_backup_count: int = 5


@dataclass
class AlertConfiguration:
    """Alert system configuration"""
    # IV alert thresholds
    iv_spike_threshold: float = 0.20  # 20% change
    extreme_iv_threshold: float = 2.0  # 200% IV
    
    # Volume alert thresholds
    volume_spike_multiplier: float = 3.0  # 3x average volume
    zero_volume_threshold: float = 0.8  # 80% zero volume
    
    # Correlation alert thresholds
    correlation_break_threshold: float = 0.3  # 30% correlation drop
    min_correlation_history: int = 10  # minimum data points
    
    # Data quality thresholds
    data_gap_threshold: int = 300  # 5 minutes
    extreme_spread_threshold: float = 0.5  # 50% spread
    extreme_spread_ratio: float = 0.2  # 20% of contracts
    
    # Alert delivery
    email_alerts: bool = False
    console_alerts: bool = True
    file_alerts: bool = True
    alert_cooldown: int = 60  # seconds between similar alerts


@dataclass
class PortfolioConfiguration:
    """Portfolio-level configuration"""
    # Risk management
    max_portfolio_var: float = 0.05  # 5% max VaR
    concentration_limit: float = 0.4  # 40% max single asset weight
    diversification_target: float = 0.8  # target diversification ratio
    
    # Correlation analysis
    correlation_window: int = 20  # periods for correlation calculation
    correlation_update_frequency: int = 300  # seconds
    min_correlation_pairs: int = 3  # minimum pairs for analysis
    
    # Signal detection
    signal_strength_threshold: float = 0.7  # minimum signal strength
    signal_history_length: int = 100  # number of signals to keep
    cross_asset_enabled: bool = True
    
    # Rebalancing
    rebalancing_enabled: bool = False
    rebalancing_frequency: int = 86400  # daily rebalancing
    rebalancing_threshold: float = 0.05  # 5% weight deviation


class ConfigurationManager:
    """
    Centralized configuration management for multi-asset system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default configuration file
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "..", "..", "config", "multi_asset_config.yaml")
        
        self.config_file = Path(config_file)
        
        # Configuration objects
        self.system_config = SystemConfiguration()
        self.alert_config = AlertConfiguration()
        self.portfolio_config = PortfolioConfiguration()
        self.asset_configs: Dict[str, AssetConfiguration] = {}
        
        # Load configuration
        self._load_configuration()
        self._validate_configuration()
        
        self.logger.info(f"Configuration manager initialized with {len(self.asset_configs)} assets")
    
    def _load_configuration(self):
        """Load configuration from file or create defaults"""
        try:
            if self.config_file.exists():
                self.logger.info(f"Loading configuration from {self.config_file}")
                self._load_from_file()
            else:
                self.logger.info("Configuration file not found, creating default configuration")
                self._create_default_configuration()
                self.save_configuration()
        
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            self._create_default_configuration()
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        with open(self.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Load system configuration
        if 'system' in config_data:
            self.system_config = SystemConfiguration(**config_data['system'])
        
        # Load alert configuration
        if 'alerts' in config_data:
            self.alert_config = AlertConfiguration(**config_data['alerts'])
        
        # Load portfolio configuration
        if 'portfolio' in config_data:
            self.portfolio_config = PortfolioConfiguration(**config_data['portfolio'])
        
        # Load asset configurations
        if 'assets' in config_data:
            for symbol, asset_data in config_data['assets'].items():
                # Convert time strings to time objects if present
                if 'trading_start' in asset_data and asset_data['trading_start']:
                    asset_data['trading_start'] = time.fromisoformat(asset_data['trading_start'])
                if 'trading_end' in asset_data and asset_data['trading_end']:
                    asset_data['trading_end'] = time.fromisoformat(asset_data['trading_end'])
                
                self.asset_configs[symbol] = AssetConfiguration(symbol=symbol, **asset_data)
    
    def _create_default_configuration(self):
        """Create default configuration for common assets"""
        # Default asset configurations
        default_assets = {
            'AAPL': AssetConfiguration(
                symbol='AAPL',
                priority='high',
                update_frequency=30,
                min_volume=50,
                max_spread=0.05,
                min_open_interest=20,
                max_weight=0.20,
                sector='technology',
                sub_sector='consumer_electronics'
            ),
            'MSFT': AssetConfiguration(
                symbol='MSFT',
                priority='high',
                update_frequency=45,
                min_volume=30,
                max_spread=0.06,
                min_open_interest=15,
                max_weight=0.20,
                sector='technology',
                sub_sector='software'
            ),
            'GOOGL': AssetConfiguration(
                symbol='GOOGL',
                priority='high',
                update_frequency=45,
                min_volume=20,
                max_spread=0.08,
                min_open_interest=10,
                max_weight=0.18,
                sector='technology',
                sub_sector='internet'
            ),
            'NVDA': AssetConfiguration(
                symbol='NVDA',
                priority='high',
                update_frequency=30,
                min_volume=40,
                max_spread=0.07,
                min_open_interest=15,
                max_weight=0.15,
                sector='technology',
                sub_sector='semiconductors'
            ),
            'TSLA': AssetConfiguration(
                symbol='TSLA',
                priority='high',
                update_frequency=30,
                min_volume=60,
                max_spread=0.10,
                min_open_interest=25,
                max_weight=0.10,
                sector='automotive',
                sub_sector='electric_vehicles'
            ),
            'SPY': AssetConfiguration(
                symbol='SPY',
                priority='critical',
                update_frequency=15,
                min_volume=100,
                max_spread=0.02,
                min_open_interest=50,
                max_weight=0.30,
                sector='broad_market',
                sub_sector='large_cap_etf'
            ),
            'QQQ': AssetConfiguration(
                symbol='QQQ',
                priority='high',
                update_frequency=30,
                min_volume=80,
                max_spread=0.03,
                min_open_interest=40,
                max_weight=0.25,
                sector='technology',
                sub_sector='tech_etf'
            ),
            'JPM': AssetConfiguration(
                symbol='JPM',
                priority='medium',
                update_frequency=60,
                min_volume=20,
                max_spread=0.08,
                min_open_interest=10,
                max_weight=0.15,
                sector='financial',
                sub_sector='banking'
            ),
            'BAC': AssetConfiguration(
                symbol='BAC',
                priority='medium',
                update_frequency=60,
                min_volume=15,
                max_spread=0.09,
                min_open_interest=8,
                max_weight=0.12,
                sector='financial',
                sub_sector='banking'
            ),
            'GME': AssetConfiguration(
                symbol='GME',
                priority='medium',
                update_frequency=45,
                min_volume=30,
                max_spread=0.15,
                min_open_interest=10,
                max_weight=0.05,
                sector='consumer_discretionary',
                sub_sector='gaming'
            )
        }
        
        self.asset_configs = default_assets
    
    def _validate_configuration(self):
        """Validate configuration values"""
        # Validate system configuration
        if self.system_config.max_worker_threads <= 0:
            self.system_config.max_worker_threads = 10
            self.logger.warning("Invalid max_worker_threads, set to 10")
        
        if not (0 < self.system_config.api_rate_limit <= 100):
            self.system_config.api_rate_limit = 5
            self.logger.warning("Invalid api_rate_limit, set to 5")
        
        # Validate alert configuration
        if not (0 < self.alert_config.iv_spike_threshold <= 1):
            self.alert_config.iv_spike_threshold = 0.20
            self.logger.warning("Invalid iv_spike_threshold, set to 0.20")
        
        # Validate portfolio configuration
        if not (0 < self.portfolio_config.concentration_limit <= 1):
            self.portfolio_config.concentration_limit = 0.4
            self.logger.warning("Invalid concentration_limit, set to 0.4")
        
        # Validate asset configurations
        for symbol, config in self.asset_configs.items():
            if config.update_frequency <= 0:
                config.update_frequency = 60
                self.logger.warning(f"Invalid update_frequency for {symbol}, set to 60")
            
            if not (0 < config.max_weight <= 1):
                config.max_weight = 0.25
                self.logger.warning(f"Invalid max_weight for {symbol}, set to 0.25")
    
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            # Create config directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare configuration data
            config_data = {
                'system': asdict(self.system_config),
                'alerts': asdict(self.alert_config),
                'portfolio': asdict(self.portfolio_config),
                'assets': {}
            }
            
            # Convert asset configurations
            for symbol, config in self.asset_configs.items():
                asset_data = asdict(config)
                
                # Convert time objects to strings
                if asset_data['trading_start']:
                    asset_data['trading_start'] = asset_data['trading_start'].isoformat()
                if asset_data['trading_end']:
                    asset_data['trading_end'] = asset_data['trading_end'].isoformat()
                
                # Remove symbol from asset data (it's the key)
                del asset_data['symbol']
                config_data['assets'][symbol] = asset_data
            
            # Add metadata
            config_data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '2.0',
                'description': 'Multi-asset real-time volatility system configuration'
            }
            
            # Save to YAML file
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def get_asset_config(self, symbol: str) -> Optional[AssetConfiguration]:
        """Get configuration for a specific asset"""
        return self.asset_configs.get(symbol.upper())
    
    def add_asset_config(self, config: AssetConfiguration):
        """Add or update asset configuration"""
        self.asset_configs[config.symbol.upper()] = config
        self.logger.info(f"Added/updated configuration for {config.symbol}")
    
    def remove_asset_config(self, symbol: str):
        """Remove asset configuration"""
        symbol = symbol.upper()
        if symbol in self.asset_configs:
            del self.asset_configs[symbol]
            self.logger.info(f"Removed configuration for {symbol}")
    
    def get_enabled_assets(self) -> List[str]:
        """Get list of enabled asset symbols"""
        return [symbol for symbol, config in self.asset_configs.items() if config.enabled]
    
    def get_assets_by_priority(self, priority: str) -> List[str]:
        """Get assets with specific priority level"""
        return [
            symbol for symbol, config in self.asset_configs.items()
            if config.enabled and config.priority == priority
        ]
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        enabled_assets = self.get_enabled_assets()
        
        # Asset statistics
        priority_counts = {}
        sector_counts = {}
        
        for symbol in enabled_assets:
            config = self.asset_configs[symbol]
            priority_counts[config.priority] = priority_counts.get(config.priority, 0) + 1
            sector_counts[config.sector] = sector_counts.get(config.sector, 0) + 1
        
        # Update frequency statistics
        update_frequencies = [self.asset_configs[symbol].update_frequency for symbol in enabled_assets]
        
        return {
            'assets': {
                'total_configured': len(self.asset_configs),
                'enabled': len(enabled_assets),
                'disabled': len(self.asset_configs) - len(enabled_assets),
                'by_priority': priority_counts,
                'by_sector': sector_counts,
                'avg_update_frequency': sum(update_frequencies) / len(update_frequencies) if update_frequencies else 0,
                'min_update_frequency': min(update_frequencies) if update_frequencies else 0,
                'max_update_frequency': max(update_frequencies) if update_frequencies else 0
            },
            'system': {
                'max_worker_threads': self.system_config.max_worker_threads,
                'api_rate_limit': self.system_config.api_rate_limit,
                'real_time_enabled': self.system_config.real_time_enabled,
                'cache_enabled': self.system_config.cache_enabled
            },
            'alerts': {
                'iv_spike_threshold': self.alert_config.iv_spike_threshold,
                'volume_spike_multiplier': self.alert_config.volume_spike_multiplier,
                'email_alerts': self.alert_config.email_alerts,
                'console_alerts': self.alert_config.console_alerts
            },
            'portfolio': {
                'max_portfolio_var': self.portfolio_config.max_portfolio_var,
                'concentration_limit': self.portfolio_config.concentration_limit,
                'cross_asset_enabled': self.portfolio_config.cross_asset_enabled,
                'rebalancing_enabled': self.portfolio_config.rebalancing_enabled
            }
        }
    
    def create_environment_config(self, environment: str) -> 'ConfigurationManager':
        """Create configuration for specific environment (dev, staging, prod)"""
        if environment == 'development':
            # Development settings: faster updates, more logging, relaxed limits
            self.update_system_config(
                api_rate_limit=10,
                log_level='DEBUG',
                real_time_enabled=True,
                max_worker_threads=5
            )
            
            self.update_alert_config(
                console_alerts=True,
                file_alerts=True,
                email_alerts=False
            )
            
            # Enable fewer assets for development
            for symbol, config in self.asset_configs.items():
                if symbol not in ['AAPL', 'MSFT', 'SPY']:
                    config.enabled = False
                else:
                    config.update_frequency = 30  # Faster updates for testing
        
        elif environment == 'staging':
            # Staging settings: production-like but with more monitoring
            self.update_system_config(
                api_rate_limit=5,
                log_level='INFO',
                real_time_enabled=True,
                max_worker_threads=8
            )
            
            self.update_alert_config(
                console_alerts=True,
                file_alerts=True,
                email_alerts=True
            )
        
        elif environment == 'production':
            # Production settings: optimized for performance and stability
            self.update_system_config(
                api_rate_limit=3,
                log_level='WARNING',
                real_time_enabled=True,
                max_worker_threads=10
            )
            
            self.update_alert_config(
                console_alerts=False,
                file_alerts=True,
                email_alerts=True,
                alert_cooldown=300  # 5 minutes cooldown in production
            )
            
            # Stricter data quality filters for production
            for config in self.asset_configs.values():
                config.min_volume = max(config.min_volume, 20)
                config.max_spread = min(config.max_spread, 0.08)
        
        self.logger.info(f"Configuration optimized for {environment} environment")
        return self
    
    def export_configuration(self, format: str = 'yaml') -> str:
        """Export configuration in specified format"""
        config_data = {
            'system': asdict(self.system_config),
            'alerts': asdict(self.alert_config),
            'portfolio': asdict(self.portfolio_config),
            'assets': {symbol: asdict(config) for symbol, config in self.asset_configs.items()}
        }
        
        if format.lower() == 'json':
            return json.dumps(config_data, indent=2, default=str)
        elif format.lower() == 'yaml':
            return yaml.dump(config_data, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_configuration(self, config_str: str, format: str = 'yaml'):
        """Import configuration from string"""
        try:
            if format.lower() == 'json':
                config_data = json.loads(config_str)
            elif format.lower() == 'yaml':
                config_data = yaml.safe_load(config_str)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # Update configurations
            if 'system' in config_data:
                self.system_config = SystemConfiguration(**config_data['system'])
            
            if 'alerts' in config_data:
                self.alert_config = AlertConfiguration(**config_data['alerts'])
            
            if 'portfolio' in config_data:
                self.portfolio_config = PortfolioConfiguration(**config_data['portfolio'])
            
            if 'assets' in config_data:
                self.asset_configs = {}
                for symbol, asset_data in config_data['assets'].items():
                    self.asset_configs[symbol] = AssetConfiguration(symbol=symbol, **asset_data)
            
            self._validate_configuration()
            self.logger.info("Configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            raise
    
    def get_update_schedule(self) -> Dict[str, int]:
        """Get update frequency schedule for all assets"""
        return {
            symbol: config.update_frequency
            for symbol, config in self.asset_configs.items()
            if config.enabled
        }
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits for all assets"""
        return {
            symbol: config.max_weight
            for symbol, config in self.asset_configs.items()
            if config.enabled
        }
    
    def update_system_config(self, **kwargs):
        """Update system configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.system_config, key):
                setattr(self.system_config, key, value)
                self.logger.info(f"Updated system config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown system config parameter: {key}")
        
        self._validate_configuration()
    
    def update_alert_config(self, **kwargs):
        """Update alert configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.alert_config, key):
                setattr(self.alert_config, key, value)
                self.logger.info(f"Updated alert config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown alert config parameter: {key}")
        
        self._validate_configuration()
    
    def update_portfolio_config(self, **kwargs):
        """Update portfolio configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.portfolio_config, key):
                setattr(self.portfolio_config, key, value)
                self.logger.info(f"Updated portfolio config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown portfolio config parameter: {key}")
        
        self._validate_configuration()
    
    def is_trading_hours(self, symbol: str) -> bool:
        """Check if it's trading hours for a specific asset"""
        config = self.get_asset_config(symbol)
        if not config or not config.trading_start or not config.trading_end:
            return True  # Default to always trading if no specific hours set
        
        current_time = datetime.now().time()
        return config.trading_start <= current_time <= config.trading_end
    
    def get_data_quality_filters(self, symbol: str) -> Dict[str, Union[int, float]]:
        """Get data quality filters for an asset"""
        config = self.get_asset_config(symbol)
        if not config:
            return {}
        
        return {
            'min_volume': config.min_volume,
            'max_spread': config.max_spread,
            'min_open_interest': config.min_open_interest,
            'min_time_to_expiry': config.min_time_to_expiry,
            'max_time_to_expiry': config.max_time_to_expiry
        }
    
    def export_configuration(self, format: str = 'yaml') -> str:
        """Export configuration in specified format"""
        config_data = {
            'system': asdict(self.system_config),
            'alerts': asdict(self.alert_config),
            'portfolio': asdict(self.portfolio_config),
            'assets': {symbol: asdict(config) for symbol, config in self.asset_configs.items()}
        }
        
        if format.lower() == 'json':
            return json.dumps(config_data, indent=2, default=str)
        elif format.lower() == 'yaml':
            return yaml.dump(config_data, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_configuration(self, config_str: str, format: str = 'yaml'):
        """Import configuration from string"""
        try:
            if format.lower() == 'json':
                config_data = json.loads(config_str)
            elif format.lower() == 'yaml':
                config_data = yaml.safe_load(config_str)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # Update configurations
            if 'system' in config_data:
                self.system_config = SystemConfiguration(**config_data['system'])
            
            if 'alerts' in config_data:
                self.alert_config = AlertConfiguration(**config_data['alerts'])
            
            if 'portfolio' in config_data:
                self.portfolio_config = PortfolioConfiguration(**config_data['portfolio'])
            
            if 'assets' in config_data:
                self.asset_configs = {}
                for symbol, asset_data in config_data['assets'].items():
                    self.asset_configs[symbol] = AssetConfiguration(symbol=symbol, **asset_data)
            
            self._validate_configuration()
            self.logger.info("Configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            raise
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        enabled_assets = self.get_enabled_assets()
        
        # Asset statistics
        priority_counts = {}
        sector_counts = {}
        
        for symbol in enabled_assets:
            config = self.asset_configs[symbol]
            priority_counts[config.priority] = priority_counts.get(config.priority, 0) + 1
            sector_counts[config.sector] = sector_counts.get(config.sector, 0) + 1
        
        # Update frequency statistics
        update_frequencies = [self.asset_configs[symbol].update_frequency for symbol in enabled_assets]
        
        return {
            'assets': {
                'total_configured': len(self.asset_configs),
                'enabled': len(enabled_assets),
                'disabled': len(self.asset_configs) - len(enabled_assets),
                'by_priority': priority_counts,
                'by_sector': sector_counts,
                'avg_update_frequency': sum(update_frequencies) / len(update_frequencies) if update_frequencies else 0,
                'min_update_frequency': min(update_frequencies) if update_frequencies else 0,
                'max_update_frequency': max(update_frequencies) if update_frequencies else 0
            },
            'system': {
                'max_worker_threads': self.system_config.max_worker_threads,
                'api_rate_limit': self.system_config.api_rate_limit,
                'real_time_enabled': self.system_config.real_time_enabled,
                'cache_enabled': self.system_config.cache_enabled
            },
            'alerts': {
                'iv_spike_threshold': self.alert_config.iv_spike_threshold,
                'volume_spike_multiplier': self.alert_config.volume_spike_multiplier,
                'email_alerts': self.alert_config.email_alerts,
                'console_alerts': self.alert_config.console_alerts
            },
            'portfolio': {
                'max_portfolio_var': self.portfolio_config.max_portfolio_var,
                'concentration_limit': self.portfolio_config.concentration_limit,
                'cross_asset_enabled': self.portfolio_config.cross_asset_enabled,
                'rebalancing_enabled': self.portfolio_config.rebalancing_enabled
            }
        }
    
    def create_environment_config(self, environment: str) -> 'ConfigurationManager':
        """Create configuration for specific environment (dev, staging, prod)"""
        if environment == 'development':
            # Development settings: faster updates, more logging, relaxed limits
            self.update_system_config(
                api_rate_limit=10,
                log_level='DEBUG',
                real_time_enabled=True,
                max_worker_threads=5
            )
            
            self.update_alert_config(
                console_alerts=True,
                file_alerts=True,
                email_alerts=False
            )
            
            # Enable fewer assets for development
            for symbol, config in self.asset_configs.items():
                if symbol not in ['AAPL', 'MSFT', 'SPY']:
                    config.enabled = False
                else:
                    config.update_frequency = 30  # Faster updates for testing
        
        elif environment == 'staging':
            # Staging settings: production-like but with more monitoring
            self.update_system_config(
                api_rate_limit=5,
                log_level='INFO',
                real_time_enabled=True,
                max_worker_threads=8
            )
            
            self.update_alert_config(
                console_alerts=True,
                file_alerts=True,
                email_alerts=True
            )
        
        elif environment == 'production':
            # Production settings: optimized for performance and stability
            self.update_system_config(
                api_rate_limit=3,
                log_level='WARNING',
                real_time_enabled=True,
                max_worker_threads=10
            )
            
            self.update_alert_config(
                console_alerts=False,
                file_alerts=True,
                email_alerts=True,
                alert_cooldown=300  # 5 minutes cooldown in production
            )
            
            # Stricter data quality filters for production
            for config in self.asset_configs.values():
                config.min_volume = max(config.min_volume, 20)
                config.max_spread = min(config.max_spread, 0.08)
        
        self.logger.info(f"Configuration optimized for {environment} environment")
        return self


# Configuration presets for common use cases
class ConfigurationPresets:
    """Predefined configuration presets for common scenarios"""
    
    @staticmethod
    def conservative_trading() -> ConfigurationManager:
        """Conservative trading configuration with strict risk controls"""
        config_manager = ConfigurationManager()
        
        config_manager.update_portfolio_config(
            max_portfolio_var=0.03,  # 3% max VaR
            concentration_limit=0.2,  # 20% max single asset
            diversification_target=0.9
        )
        
        config_manager.update_alert_config(
            iv_spike_threshold=0.15,  # 15% IV spike threshold
            volume_spike_multiplier=2.0
        )
        
        # Conservative asset weights
        for config in config_manager.asset_configs.values():
            config.max_weight = min(config.max_weight, 0.15)
            config.min_volume = max(config.min_volume, 50)
        
        return config_manager
    
    @staticmethod
    def aggressive_trading() -> ConfigurationManager:
        """Aggressive trading configuration with higher risk tolerance"""
        config_manager = ConfigurationManager()
        
        config_manager.update_portfolio_config(
            max_portfolio_var=0.08,  # 8% max VaR
            concentration_limit=0.5,  # 50% max single asset
            diversification_target=0.6
        )
        
        config_manager.update_alert_config(
            iv_spike_threshold=0.30,  # 30% IV spike threshold
            volume_spike_multiplier=5.0
        )
        
        config_manager.update_system_config(
            api_rate_limit=8,  # Faster data updates
            update_batch_size=8
        )
        
        return config_manager
    
    @staticmethod
    def research_mode() -> ConfigurationManager:
        """Research configuration with comprehensive data collection"""
        config_manager = ConfigurationManager()
        
        config_manager.update_system_config(
            log_level='DEBUG',
            cache_duration=3600,  # 1 hour cache
            max_cache_size=5000
        )
        
        config_manager.update_alert_config(
            console_alerts=True,
            file_alerts=True,
            alert_cooldown=0  # No cooldown for research
        )
        
        # Enable all assets for research
        for config in config_manager.asset_configs.values():
            config.enabled = True
            config.min_volume = 1  # Accept low volume for research
            config.max_spread = 1.0  # Accept wide spreads
        
        return config_manager


def create_sample_config_file():
    """Create a sample configuration file"""
    config_manager = ConfigurationManager()
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Save sample configuration
    sample_file = config_dir / "sample_multi_asset_config.yaml"
    config_manager.config_file = sample_file
    config_manager.save_configuration()
    
    print(f"Sample configuration created at: {sample_file}")
    return str(sample_file)


if __name__ == "__main__":
    # Create sample configuration file
    sample_file = create_sample_config_file()
    
    # Demonstrate configuration management
    print("\nConfiguration Management Demo")
    print("=" * 40)
    
    # Load configuration
    config_manager = ConfigurationManager(sample_file)
    
    # Show configuration summary
    summary = config_manager.get_configuration_summary()
    print(f"Total assets configured: {summary['assets']['total_configured']}")
    print(f"Enabled assets: {summary['assets']['enabled']}")
    print(f"Assets by priority: {summary['assets']['by_priority']}")
    print(f"Assets by sector: {summary['assets']['by_sector']}")
    
    # Show environment-specific configurations
    print("\nEnvironment Configurations:")
    for env in ['development', 'staging', 'production']:
        env_config = ConfigurationManager(sample_file)
        env_config.create_environment_config(env)
        print(f"{env.title()}: {len(env_config.get_enabled_assets())} enabled assets")
    
    # Show presets
    print("\nConfiguration Presets:")
    conservative = ConfigurationPresets.conservative_trading()
    aggressive = ConfigurationPresets.aggressive_trading()
    research = ConfigurationPresets.research_mode()
    
    print(f"Conservative: VaR limit {conservative.portfolio_config.max_portfolio_var:.1%}")
    print(f"Aggressive: VaR limit {aggressive.portfolio_config.max_portfolio_var:.1%}")
    print(f"Research: {len(research.get_enabled_assets())} assets enabled")


# Configuration presets for common use cases
class ConfigurationPresets:
    """Predefined configuration presets for common scenarios"""
    
    @staticmethod
    def conservative_trading():
        """Conservative trading configuration with strict risk controls"""
        config_manager = ConfigurationManager()
        
        config_manager.update_portfolio_config(
            max_portfolio_var=0.03,  # 3% max VaR
            concentration_limit=0.2,  # 20% max single asset
            diversification_target=0.9
        )
        
        config_manager.update_alert_config(
            iv_spike_threshold=0.15,  # 15% IV spike threshold
            volume_spike_multiplier=2.0
        )
        
        # Conservative asset weights
        for config in config_manager.asset_configs.values():
            config.max_weight = min(config.max_weight, 0.15)
            config.min_volume = max(config.min_volume, 50)
        
        return config_manager
    
    @staticmethod
    def aggressive_trading():
        """Aggressive trading configuration with higher risk tolerance"""
        config_manager = ConfigurationManager()
        
        config_manager.update_portfolio_config(
            max_portfolio_var=0.08,  # 8% max VaR
            concentration_limit=0.5,  # 50% max single asset
            diversification_target=0.6
        )
        
        config_manager.update_alert_config(
            iv_spike_threshold=0.30,  # 30% IV spike threshold
            volume_spike_multiplier=5.0
        )
        
        config_manager.update_system_config(
            api_rate_limit=8,  # Faster data updates
            update_batch_size=8
        )
        
        return config_manager
    
    @staticmethod
    def research_mode():
        """Research configuration with comprehensive data collection"""
        config_manager = ConfigurationManager()
        
        config_manager.update_system_config(
            log_level='DEBUG',
            cache_duration=3600,  # 1 hour cache
            max_cache_size=5000
        )
        
        config_manager.update_alert_config(
            console_alerts=True,
            file_alerts=True,
            alert_cooldown=0  # No cooldown for research
        )
        
        # Enable all assets for research
        for config in config_manager.asset_configs.values():
            config.enabled = True
            config.min_volume = 1  # Accept low volume for research
            config.max_spread = 1.0  # Accept wide spreads
        
        return config_manager


def create_sample_config_file():
    """Create a sample configuration file"""
    config_manager = ConfigurationManager()
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Save sample configuration
    sample_file = config_dir / "sample_multi_asset_config.yaml"
    config_manager.config_file = sample_file
    config_manager.save_configuration()
    
    print(f"Sample configuration created at: {sample_file}")
    return str(sample_file)