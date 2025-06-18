# config.py - Updated for yfinance integration

"""
Configuration file for Options Volatility Surface system with yfinance
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration - yfinance doesn't need keys, but keep for backwards compatibility
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')

# yfinance is the primary data source now - no API key needed!
USE_YFINANCE = True
YFINANCE_CACHE_DURATION = 300  # 5 minutes cache

# Risk-free rate (can be made dynamic later)
RISK_FREE_RATE = 0.05

# Default settings with current symbols
DEFAULT_SYMBOLS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Google
    'NVDA',   # NVIDIA
    'TSLA',   # Tesla
    'SPY',    # S&P 500 ETF
    'QQQ',    # NASDAQ ETF
    'META',   # Meta
    'AMZN',   # Amazon
    'JPM',    # JPMorgan
]

# Data validation settings (more lenient for yfinance)
MIN_VOLUME = 1           # yfinance provides good data even for lower volume
MAX_BID_ASK_SPREAD_PCT = 1.0  # Allow wider spreads since we're not actually trading
MIN_OPTION_PRICE = 0.01  # Minimum option price to consider

# Real-time update settings
UPDATE_INTERVAL_SECONDS = 60    # How often to refresh data
PRICE_CACHE_SECONDS = 300       # How long to cache stock prices

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_FILE = 'volatility_system.log'

# Dashboard settings
DASHBOARD_TITLE = "Real-Time Options Volatility Surface (Powered by yfinance)"
DASHBOARD_PORT = 8501
AUTO_REFRESH_DEFAULT = False    # Don't auto-refresh by default to be nice to yfinance

# Performance settings
MAX_CONCURRENT_REQUESTS = 5     # Don't overload yfinance
REQUEST_DELAY_SECONDS = 0.1     # Small delay between requests