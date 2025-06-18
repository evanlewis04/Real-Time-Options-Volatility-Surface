import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')

# Default settings
DEFAULT_SYMBOL = 'AAPL'
RISK_FREE_RATE = 0.05  # 5% - we'll make this dynamic later
BASE_URL_ALPHA_VANTAGE = 'https://www.alphavantage.co/query'

# Data validation settings
MIN_VOLUME = 10
MAX_BID_ASK_SPREAD_PCT = 0.5  # 50% max spread