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

MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD"},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD"},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY"},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY"},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY"},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP"},
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR"}
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

def get_fill_data(ticker, market):
    """
    Fetches the most recent OPEN price and FX rate.
    Returns: (local_price, usd_price)
    """
    try:
        # Fetch 5 days to ensure we get the latest trading day (handle weekends/holidays)
        # We use 'Open' price for fills as per rules
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        
        if hist.empty:
            logger.error(f"No data found for {ticker}")
            return None, None
            
        # Get the latest available 'Open'
        local_p = extract_scalar(hist['Open'].iloc[-1])
        
        # FX Conversion
        usd_p = local_p
        
        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                fx_hist = yf.Ticker(cfg['fx']).history(period="5d")
                if not fx_hist.empty:
                    rate = extract_scalar(fx_hist['Close'].iloc[-1]) # Use Close for FX
                    
                    # UK Adjustment
                    if market == "UK": local_p = local_p / 100.0
                    
                    # Convert to USD (Assuming rate is Local per USD, e.g., HKD=X is 7.8)
                    if rate > 0:
                        usd_p = (local_p / 100.0) if market == "UK" else local_p
                        usd_p = usd_p / rate
                    else:
                        logger.error(f"Zero FX rate for {ticker}")
                        return local_p, 0.0
                else:
                    logger.error(f"No FX data for {cfg['fx']}")
                    return local_p, 0.0
        
        return local_p, usd_p

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None

# --- 5. MAIN TASK ---
def task_fill_orders():
    logger.info("Starting Daily Fill Job...")
    
    # Fetch only PENDING orders
    pending = session.query(Transaction).filter_by(status='PENDING').all()
    
    if not pending:
        logger.info("No pending orders found.")
        return

    logger.info(f"Found {len(pending)} pending orders.")

    for t in pending:
        try:
            logger.info(f"Processing {t.trans_type} {t.ticker} ({t.market})...")
            
            # Determine Market if missing (Fallback)
            market = t.market if t.market else "US"
            
            local_p, usd_p = get_fill_data(t.ticker, market)
            
            if usd_p and usd_p > 0:
                # Calculate Quantity based on USD Amount and Calculated USD Price
                # Qty = USD Amount / USD Price
                qty = t.amount / usd_p
                
                t.price = usd_p
                t.local_price = local_p
                t.quantity = qty
                t.status = 'FILLED'
                t.date = datetime.now() # Mark filled time (now) or use market open time
                
                logger.info(f"SUCCESS: {t.ticker} filled @ ${usd_p:.2f} USD ({t.quantity:.4f} shares)")
            else:
                logger.warning(f"SKIPPED: Could not determine valid price for {t.ticker}")
                
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing {t.ticker}: {e}")
            # Optional: t.notes = f"Auto-fill failed: {str(e)}"
    
    session.commit()
    logger.info("Job Complete. Database updated.")

if __name__ == "__main__":
    task_fill_orders()
