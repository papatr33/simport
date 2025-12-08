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

# --- 1. CONFIGURATION ---
# IPv4 Force for GitHub Actions
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DailyJob")

# MARKET CONFIG (Must match app.py)
MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD", "delay_min": 0},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD", "delay_min": 15},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY", "delay_min": 30},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY", "delay_min": 30},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY", "delay_min": 20},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP", "delay_min": 20},
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR", "delay_min": 15},
    "Netherlands": {"suffix": ".AS", "fx": "EUR=X", "currency": "EUR", "delay_min": 15}
}

# TIMEZONE MAP (For accurate delay calculation)
MARKET_TIMEZONES = {
    "US": "US/Eastern",
    "Hong Kong": "Asia/Hong_Kong",
    "China (Shanghai)": "Asia/Shanghai",
    "China (Shenzhen)": "Asia/Shanghai",
    "Japan": "Asia/Tokyo",
    "UK": "Europe/London",
    "France": "Europe/Paris",
    "Netherlands": "Europe/Amsterdam"
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
    
    return create_engine(db_url, pool_pre_ping=True)

Base = declarative_base()

# --- 3. MODELS ---
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
    
    price = Column(Float, nullable=True)       
    local_price = Column(Float, nullable=True) 
    quantity = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)      
    
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

def get_fill_data(ticker, market, trade_time):
    """
    Fetches the price with STRICT Timezone Alignment.
    Prevents 'Naive Comparison' bugs where 09:30 (Local) > 05:44 (UTC).
    """
    try:
        delay = MARKET_CONFIG.get(market, {}).get('delay_min', 0)
        
        # --- 1. TIMEZONE & DELAY CHECK ---
        tz_name = MARKET_TIMEZONES.get(market, "UTC")
        local_tz = pytz.timezone(tz_name)
        
        # Localize Trade Time
        if trade_time.tzinfo is None:
            trade_time_aware = local_tz.localize(trade_time)
        else:
            trade_time_aware = trade_time.astimezone(local_tz)
            
        now_utc = datetime.now(pytz.utc)
        trade_time_utc = trade_time_aware.astimezone(pytz.utc)
        data_available_utc = trade_time_utc + timedelta(minutes=delay)
        
        if now_utc < data_available_utc:
            logger.info(f"WAITING {ticker}: Current UTC {now_utc.strftime('%H:%M')} < Avail {data_available_utc.strftime('%H:%M')} (Delay {delay}m)")
            return None, None
        
        # --- 2. FETCH DATA ---
        day_start = trade_time_aware.replace(hour=0, minute=0, second=0, microsecond=0)
        search_end = day_start + timedelta(days=5)
        
        interval = "5m"
        if (datetime.now(local_tz) - trade_time_aware).days > 55:
            interval = "1d" 
            
        df = yf.download(ticker, start=day_start, end=search_end, interval=interval, progress=False)
        
        local_p = None
        
        if not df.empty:
            # --- 2B. STRICT TIMEZONE NORMALIZATION ---
            # Problem: yfinance index can be "Naive Local", "Naive UTC", or "Aware".
            # Solution: Force everything to UTC Aware.
            
            if df.index.tz is None:
                # If naive, assume it implies Market Local Time (e.g. 09:30 for HK Open)
                # Localize to HK Time, then convert to UTC
                df.index = df.index.tz_localize(local_tz, ambiguous='NaT', nonexistent='shift_forward')
                df.index = df.index.tz_convert(pytz.utc)
            else:
                # If already aware, just ensure it's UTC
                df.index = df.index.tz_convert(pytz.utc)
            
            # Now both sides of the comparison are strictly UTC
            target_time_utc = trade_time_aware.astimezone(pytz.utc)
            
            # Filter: Candles >= Trade Time
            valid_candles = df[df.index >= target_time_utc]
            
            if not valid_candles.empty:
                local_p = extract_scalar(valid_candles['Open'].iloc[0])
                found_time = valid_candles.index[0]
                logger.info(f"Price matched: Trade {target_time_utc.strftime('%H:%M')} UTC -> Fill {found_time.strftime('%H:%M')} UTC @ {local_p}")
            else:
                logger.info(f"No intraday data found after {target_time_utc.strftime('%H:%M')} UTC on same day. Looking for next open...")
                future_candles = df[df.index > target_time_utc]
                if not future_candles.empty:
                    local_p = extract_scalar(future_candles['Open'].iloc[0])
        
        # Fallback to Daily
        if local_p is None:
            if interval == "5m":
                logger.warning(f"5m data missing/filtered out for {ticker}. Falling back to Daily.")
                df_daily = yf.download(ticker, start=day_start, end=search_end, interval="1d", progress=False)
                if not df_daily.empty:
                    valid_days = df_daily[df_daily.index.date >= trade_time_aware.date()]
                    if not valid_days.empty:
                         local_p = extract_scalar(valid_days['Open'].iloc[0])
        
        if local_p is None:
            logger.info(f"No market data found for {ticker} (Market closed/Holiday?)")
            return None, None

        # --- 3. FX CONVERSION ---
        usd_p = local_p
        
        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                fx_hist = yf.Ticker(cfg['fx']).history(period="5d")
                if not fx_hist.empty:
                    rate = extract_scalar(fx_hist['Close'].iloc[-1]) 
                    if market == "UK": local_p = local_p / 100.0
                    if rate > 0:
                        usd_p = (local_p / 100.0) if market == "UK" else local_p
                        usd_p = usd_p / rate
                    else:
                        logger.error(f"FX Rate 0 detected for {cfg['fx']}")
                        return local_p, 0.0 
                else:
                    return local_p, 0.0 
        
        return local_p, usd_p

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None

# --- 5. MAIN TASK ---
def task_fill_orders():
    logger.info("Starting High-Frequency Fill Job (Strict TZ)...")
    
    pending = session.query(Transaction).filter_by(status='PENDING').all()
    
    if not pending:
        logger.info("No pending orders.")
        return

    logger.info(f"Found {len(pending)} pending orders.")

    for t in pending:
        try:
            logger.info(f"Checking {t.ticker} (Trade Time: {t.date} | Market: {t.market})...")
            
            market = t.market if t.market else "US"
            
            local_p, usd_p = get_fill_data(t.ticker, market, t.date)
            
            if usd_p and usd_p > 0:
                qty = t.quantity
                if not qty or qty <= 0:
                    qty = t.amount / usd_p if t.amount else 0
                
                final_amount = qty * usd_p
                
                t.price = usd_p
                t.local_price = local_p
                t.quantity = qty
                t.amount = final_amount
                t.status = 'FILLED'
                
                logger.info(f"FILLED: {t.ticker} | Qty: {qty} | Price: ${usd_p:.2f} | Amt: ${final_amount:.0f}")
            else:
                pass 
                
        except Exception as e:
            logger.error(f"Error processing {t.ticker}: {e}")
    
    session.commit()
    logger.info("Cycle Complete.")

if __name__ == "__main__":
    task_fill_orders()
