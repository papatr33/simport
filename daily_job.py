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
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DailyJob")

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

# CRITICAL: Map Markets to their Local Timezones to interpret YFinance data correctly
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
    Fetches price with STRICT UTC NORMALIZATION for both Trade Time and Market Data.
    """
    try:
        delay = MARKET_CONFIG.get(market, {}).get('delay_min', 0)
        
        # 1. SETUP TIMEZONES
        # We need the Market TZ to interpret Naive YFinance data correctly
        tz_name = MARKET_TIMEZONES.get(market, "UTC")
        market_tz = pytz.timezone(tz_name)
        
        # 2. STANDARDIZE TRADE TIME TO UTC
        # The DB stores naive UTC (per your setup). We force it to be UTC Aware.
        if trade_time.tzinfo is None:
            trade_time_utc = trade_time.replace(tzinfo=pytz.utc)
        else:
            trade_time_utc = trade_time.astimezone(pytz.utc)
        
        # 3. CHECK DELAY
        now_utc = datetime.now(pytz.utc)
        data_available_utc = trade_time_utc + timedelta(minutes=delay)
        
        if now_utc < data_available_utc:
            logger.info(f"WAITING {ticker}: Current {now_utc.strftime('%H:%M')} UTC < Avail {data_available_utc.strftime('%H:%M')} UTC")
            return None, None
        
        # 4. FETCH DATA (Full Day Strategy)
        # We search from the beginning of the trade day (in UTC)
        day_start_utc = trade_time_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        search_end_utc = day_start_utc + timedelta(days=5)
        
        interval = "5m"
        # If trade is older than 55 days, yfinance 5m is unavailable
        if (datetime.now(pytz.utc) - trade_time_utc).days > 55:
            interval = "1d"
            
        df = yf.download(ticker, start=day_start_utc, end=search_end_utc, interval=interval, progress=False)
        
        local_p = None
        
        if not df.empty:
            # --- CRITICAL: NORMALIZE YFINANCE INDEX TO UTC ---
            # Scenario A: YFinance returns Naive timestamps (e.g. 09:30).
            # We MUST assume this is Market Local Time (Asia/Hong_Kong).
            if df.index.tz is None:
                # 1. Localize to Market Time (09:30 HK)
                df.index = df.index.tz_localize(market_tz, ambiguous='NaT', nonexistent='shift_forward')
                # 2. Convert to UTC (01:30 UTC)
                df.index = df.index.tz_convert(pytz.utc)
            else:
                # Scenario B: YFinance returns Aware timestamps. Just convert to UTC.
                df.index = df.index.tz_convert(pytz.utc)
            
            # 5. FILTER: Find candles AFTER the Trade Time
            # Now both side are strictly UTC. 
            # 01:30 UTC (Open) < 05:44 UTC (Trade) --> False
            # 05:45 UTC (Next Candle) > 05:44 UTC (Trade) --> True (Match)
            valid_candles = df[df.index >= trade_time_utc]
            
            if not valid_candles.empty:
                local_p = extract_scalar(valid_candles['Open'].iloc[0])
                found_time = valid_candles.index[0]
                logger.info(f"MATCH: Trade {trade_time_utc.strftime('%H:%M')} UTC -> Fill {found_time.strftime('%H:%M')} UTC @ {local_p}")
            else:
                # No data found today after trade time (e.g. trade placed after close)
                # Look for the very next candle available in the future
                future_candles = df[df.index > trade_time_utc]
                if not future_candles.empty:
                    local_p = extract_scalar(future_candles['Open'].iloc[0])
                    found_time = future_candles.index[0]
                    logger.info(f"NEXT OPEN: Trade {trade_time_utc.strftime('%H:%M')} UTC -> Fill {found_time.strftime('%H:%M')} UTC @ {local_p}")

        # Fallback to Daily if 5m failed
        if local_p is None:
            if interval == "5m":
                logger.warning(f"5m data missing for {ticker}. Trying Daily.")
                df_daily = yf.download(ticker, start=day_start_utc, end=search_end_utc, interval="1d", progress=False)
                if not df_daily.empty:
                    # Normalize Daily Index too
                    if df_daily.index.tz is None:
                        df_daily.index = df_daily.index.tz_localize(market_tz).tz_convert(pytz.utc)
                    else:
                        df_daily.index = df_daily.index.tz_convert(pytz.utc)
                        
                    valid_days = df_daily[df_daily.index >= trade_time_utc]
                    if not valid_days.empty:
                         local_p = extract_scalar(valid_days['Open'].iloc[0])
        
        if local_p is None:
            logger.info(f"No market data found for {ticker} (Market closed/Holiday?)")
            return None, None

        # --- 6. FX CONVERSION ---
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
                        return local_p, 0.0 
                else:
                    return local_p, 0.0 
        
        return local_p, usd_p

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None

def task_fill_orders():
    logger.info("Starting High-Frequency Fill Job (Strict UTC)...")
    
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
                # Use Qty from input, calculate Amount
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
