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
    Fetches the price respecting delays and Timezones.
    IMPROVED: Fetches full day intraday data to prevent "Day Open" fallback errors.
    """
    try:
        delay = MARKET_CONFIG.get(market, {}).get('delay_min', 0)
        
        # --- 1. TIMEZONE & DELAY CHECK ---
        # Determine Market Timezone
        tz_name = MARKET_TIMEZONES.get(market, "UTC")
        local_tz = pytz.timezone(tz_name)
        
        # A. Localize the Trade Time
        if trade_time.tzinfo is None:
            trade_time_aware = local_tz.localize(trade_time)
        else:
            trade_time_aware = trade_time.astimezone(local_tz)
            
        # B. Get "Now" in UTC
        now_utc = datetime.now(pytz.utc)
        
        # C. Convert Trade Time to UTC for comparison
        trade_time_utc = trade_time_aware.astimezone(pytz.utc)
        
        # D. Calculate when data becomes available (Trade Time + Delay)
        data_available_utc = trade_time_utc + timedelta(minutes=delay)
        
        # E. The Check
        if now_utc < data_available_utc:
            logger.info(f"WAITING {ticker}: Current UTC {now_utc.strftime('%H:%M')} < Avail {data_available_utc.strftime('%H:%M')} (Delay {delay}m)")
            return None, None
        
        # --- 2. FETCH DATA (ROBUST METHOD) ---
        # Strategy: Fetch data starting from the BEGINNING of the trade day.
        # This increases the success rate of yfinance returning 5m data.
        
        # Start of the day (Midnight local time)
        day_start = trade_time_aware.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # End search 5 days later (to handle weekends if trade was Friday)
        search_end = day_start + timedelta(days=5)
        
        # Try 5m interval for precision if within last 55 days
        interval = "5m"
        if (datetime.now(local_tz) - trade_time_aware).days > 55:
            interval = "1d" 
            logger.warning(f"{ticker}: Trade > 55 days old, forced to use Daily Open price.")
            
        # Download data starting from Midnight
        df = yf.download(ticker, start=day_start, end=search_end, interval=interval, progress=False)
        
        local_p = None
        
        if not df.empty:
            # IMPORTANT: Filter for candles strictly AFTER or ON the specific trade execution time
            # We must ensure we filter using the same timezone awareness
            
            # yfinance returns timezone-aware indexes. Ensure trade_time_aware matches.
            # We align everything to the DataFrame's timezone to be safe
            df_tz = df.index.tz
            target_time = trade_time_aware.astimezone(df_tz)
            
            # Slice: Give me all candles that happened AFTER my button click
            valid_candles = df[df.index >= target_time]
            
            if not valid_candles.empty:
                # The first available candle is our execution price
                local_p = extract_scalar(valid_candles['Open'].iloc[0])
                found_time = valid_candles.index[0]
                logger.info(f"Price matched: Trade {trade_time_aware.strftime('%H:%M')} -> Fill {found_time.strftime('%H:%M')} @ {local_p}")
            else:
                # We have data for the day, but nothing after the trade time (e.g. trade placed at 15:59 and market closed)
                # In this case, we look for the next day in the original df
                logger.info(f"No intraday data found after {trade_time_aware.strftime('%H:%M')} on same day. Looking for next open...")
                future_candles = df[df.index > target_time]
                if not future_candles.empty:
                    local_p = extract_scalar(future_candles['Open'].iloc[0])
        
        # Fallback: If 5m completely failed (empty), try Daily
        if local_p is None:
            if interval == "5m":
                logger.warning(f"5m data missing for {ticker}. Falling back to Daily Open.")
                df_daily = yf.download(ticker, start=day_start, end=search_end, interval="1d", progress=False)
                # Same logic: Find first day >= trade date
                if not df_daily.empty:
                    # Filter for days on or after trade date
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
                # Fetch FX rate for the same time window
                fx_hist = yf.Ticker(cfg['fx']).history(period="5d")
                if not fx_hist.empty:
                    rate = extract_scalar(fx_hist['Close'].iloc[-1]) 
                    
                    if market == "UK": local_p = local_p / 100.0
                    
                    if rate > 0:
                        usd_p = (local_p / 100.0) if market == "UK" else local_p
                        usd_p = usd_p / rate
                    else:
                        logger.error(f"FX Rate 0 detected for {cfg['fx']}")
                        return local_p, 0.0 # Error
                else:
                    return local_p, 0.0 # FX missing
        
        return local_p, usd_p

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    
# --- 5. MAIN TASK ---
def task_fill_orders():
    logger.info("Starting High-Frequency Fill Job (Timezone Aware)...")
    
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
                # --- MODIFIED EXECUTION LOGIC ---
                # Use the Quantity the Analyst requested.
                # Recalculate the USD Amount based on execution price.
                qty = t.quantity
                if not qty or qty <= 0:
                    # Fallback if quantity wasn't saved correctly (backward compatibility)
                    qty = t.amount / usd_p if t.amount else 0
                
                final_amount = qty * usd_p
                
                t.price = usd_p
                t.local_price = local_p
                t.quantity = qty
                t.amount = final_amount # Update with actual filled value
                t.status = 'FILLED'
                
                logger.info(f"FILLED: {t.ticker} | Qty: {qty} | Price: ${usd_p:.2f} | Amt: ${final_amount:.0f}")
            else:
                pass # Already logged waiting message
                
        except Exception as e:
            logger.error(f"Error processing {t.ticker}: {e}")
    
    session.commit()
    logger.info("Cycle Complete.")

if __name__ == "__main__":
    task_fill_orders()
