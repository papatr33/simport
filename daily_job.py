import socket
import logging
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timedelta
import os
import toml
import pandas as pd
import time
import pytz

# --- 1. CONFIGURATION (Must match app.py) ---
# IPv4 Force for GitHub Actions
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DailyJob")

# UPDATED MARKET CONFIG with delays (minutes)
# Ensure this matches app.py additions (Netherlands)
MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD", "delay_min": 0},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD", "delay_min": 0},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY", "delay_min": 30},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY", "delay_min": 30},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY", "delay_min": 20},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP", "delay_min": 15},
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR", "delay_min": 15},
    "Netherlands": {"suffix": ".AS", "fx": "EUR=X", "currency": "EUR", "delay_min": 15}
}

# --- 2. DB SETUP ---
def get_engine():
    db_url = None
    if "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    else:
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            if "DATABASE_URL" in secrets.get("general", {}): db_url = secrets["general"]["DATABASE_URL"]
            elif "DATABASE_URL" in secrets: db_url = secrets["DATABASE_URL"]
        except: pass
    
    if not db_url: 
        logger.warning("No DATABASE_URL found. Using local SQLite.")
        return create_engine('sqlite:///portfolio.db', connect_args={'check_same_thread': False})
    
    if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    # pool_pre_ping=True is CRITICAL for Supabase/Cloud SQL to avoid stale connection errors
    return create_engine(db_url, pool_pre_ping=True)

Base = declarative_base()

# --- 3. MODELS (Must match app.py exactly) ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    initial_capital = Column(Float, default=5000000.0)
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String, nullable=False)
    market = Column(String, nullable=True) 
    trans_type = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.now)
    
    price = Column(Float, nullable=True)       # USD Price
    local_price = Column(Float, nullable=True) # Local Currency Price
    quantity = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)      # USD Value
    
    status = Column(String, default='PENDING')
    notes = Column(Text, nullable=True)
    
    user = relationship("User", back_populates="transactions")

engine = get_engine()
Session = sessionmaker(bind=engine)
session = Session()

# --- 4. DATA HELPERS ---
def extract_scalar(val):
    try:
        if isinstance(val, (pd.Series, pd.DataFrame, list)):
            val = val.values.flatten()[0] if hasattr(val, 'values') else val[0]
        return float(val)
    except: return 0.0

def get_intraday_price(ticker, trade_time_utc):
    """
    Fetches 1-minute/5-minute data around the trade time to find the best fill price.
    """
    try:
        # Buffer: fetch data from trade_time to +60 mins to ensure we catch a candle
        start_t = trade_time_utc
        end_t = trade_time_utc + timedelta(minutes=60) 
        
        # Download 5m data (reliable intraday)
        # yfinance handles timezone logic internally, usually returning UTC or local.
        df = yf.download(ticker, start=start_t, end=end_t, interval="5m", progress=False)
        
        if df.empty:
            logger.warning(f"No intraday data for {ticker} at {start_t}. Will try fallback.")
            return None
        
        # Get the first available row (closest to trade time)
        # Using 'Open' of the candle
        price = extract_scalar(df['Open'].iloc[0])
        return price
    except Exception as e:
        logger.error(f"Intraday fetch failed for {ticker}: {e}")
        return None

def get_fill_data(ticker, market, trade_time):
    """
    Fetches the price respecting delays.
    """
    try:
        delay = MARKET_CONFIG.get(market, {}).get('delay_min', 0)
        
        # Calculate when data becomes available
        data_available_time = trade_time + timedelta(minutes=delay)
        now = datetime.now()
        
        # If current time is BEFORE data is available, we wait.
        if now < data_available_time:
            logger.info(f"WAITING: {ticker} needs delay of {delay}m. Data avail at {data_available_time.strftime('%H:%M')}. Current: {now.strftime('%H:%M')}")
            return None, None
        
        # Data is theoretically available, fetch specific intraday candle
        local_p = get_intraday_price(ticker, trade_time)
        
        if not local_p:
             # Fallback to daily 'Open' if intraday fails 
             # (e.g. trade was days ago, or yfinance didn't return 5m data)
             logger.info(f"Intraday failed for {ticker}, trying Daily history...")
             stock = yf.Ticker(ticker)
             hist = stock.history(period="5d")
             if not hist.empty:
                 local_p = extract_scalar(hist['Open'].iloc[-1])
             else:
                 logger.error(f"Daily history also failed for {ticker}")
                 return None, None

        # FX Conversion
        usd_p = local_p
        
        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                # Simplification: Grab recent history close for FX
                fx_hist = yf.Ticker(cfg['fx']).history(period="5d")
                if not fx_hist.empty:
                    rate = extract_scalar(fx_hist['Close'].iloc[-1]) 
                    
                    if market == "UK": local_p = local_p / 100.0
                    
                    if rate > 0:
                        usd_p = (local_p / 100.0) if market == "UK" else local_p
                        usd_p = usd_p / rate
                    else:
                        return local_p, 0.0
                else:
                    return local_p, 0.0
        
        return local_p, usd_p

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None

# --- 5. MAIN TASK ---
def task_fill_orders():
    logger.info("Starting High-Frequency Fill Job...")
    
    pending = session.query(Transaction).filter_by(status='PENDING').all()
    
    if not pending:
        logger.info("No pending orders.")
        return

    logger.info(f"Found {len(pending)} pending orders.")

    for t in pending:
        try:
            logger.info(f"Checking {t.ticker} (Trade Time: {t.date})...")
            
            market = t.market if t.market else "US"
            
            # Pass trade time to get precise price
            local_p, usd_p = get_fill_data(t.ticker, market, t.date)
            
            if usd_p and usd_p > 0:
                # Execution Logic:
                # Qty = USD Amount / USD Price
                qty = t.amount / usd_p
                
                t.price = usd_p
                t.local_price = local_p
                t.quantity = qty
                t.status = 'FILLED'
                
                # Note: We keep t.date as the original trade time for PnL tracking consistency
                
                logger.info(f"FILLED: {t.ticker} @ ${usd_p:.2f} USD")
            else:
                pass # Logged inside get_fill_data if waiting
                
        except Exception as e:
            logger.error(f"Error processing {t.ticker}: {e}")
    
    session.commit()
    logger.info("Cycle Complete.")

if __name__ == "__main__":
    task_fill_orders()
