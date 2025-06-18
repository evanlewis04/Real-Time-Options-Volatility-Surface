# Compatibility layer for old imports
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Config class for backward compatibility"""
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY') 
    IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
    RISK_FREE_RATE = 0.05
    DEFAULT_SYMBOL = 'AAPL'
    BASE_URL_ALPHA_VANTAGE = 'https://www.alphavantage.co/query'
    MIN_VOLUME = 10
    MAX_BID_ASK_SPREAD_PCT = 0.5

# Also provide module-level variables for compatibility
ALPHA_VANTAGE_API_KEY = Config.ALPHA_VANTAGE_API_KEY
RISK_FREE_RATE = Config.RISK_FREE_RATE
DEFAULT_SYMBOL = Config.DEFAULT_SYMBOL