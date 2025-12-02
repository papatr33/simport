# ==========================================
# 1. FORCE IPv4 (FIX FOR GITHUB ACTIONS)
# ==========================================
import socket
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ==========================================

import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import os
import toml

# ==========================================
# DATABASE CONNECTION
# ==========================================
def get_engine():
    db_url = None
    if "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    else:
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            if "DATABASE_URL" in secrets.get("general", {}):
                db_url = secrets["general"]["DATABASE_URL"]
            elif "DATABASE_URL" in secrets:
                db_url = secrets["DATABASE_URL"]
        except: pass

    if not db_url:
        return create_engine('sqlite:///portfolio.db', connect_args={'check_same_thread': False})

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

# ==========================================
# MODELS
# ==========================================
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    initial_capital = Column(Float, default=5000000.0)
    trades = relationship("Trade", back_populates="user")

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String)
    direction = Column(String)
    
    entry_date = Column(DateTime)
    entry_price = Column(Float)
    quantity = Column(Float)
    trade_amount = Column(Float)
    
    status = Column(String)
    user = relationship("User", back_populates="trades")

engine = get_engine()
Base.metadata.create_all(engine) # Ensure tables exist
Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# LOGIC
# ==========================================
def log(msg):
    print(f"[{datetime.now()}] {msg}")

def task_fill_orders():
    log("Checking Pending Orders...")
    pending = session.query(Trade).filter_by(status='PENDING').all()
    
    for t in pending:
        try:
            log(f"Processing {t.ticker}...")
            hist = yf.Ticker(t.ticker).history(period="5d")
            if not hist.empty:
                fill_price = hist['Open'].iloc[-1]
                
                # Calculate Quantity based on allocated Amount
                # Qty = Amount / Price
                if fill_price > 0:
                    t.entry_price = fill_price
                    t.quantity = t.trade_amount / fill_price
                    t.status = 'OPEN'
                    log(f"Filled {t.ticker}: ${t.trade_amount:,.0f} -> {t.quantity:.0f} shares @ ${fill_price:.2f}")
                else:
                    log(f"Price error for {t.ticker}")
        except Exception as e:
            log(f"Error {t.ticker}: {e}")
    session.commit()

# Note: Check Stops logic is simpler now, mostly for notification
# We don't auto-close trades in this version, just alert.

if __name__ == "__main__":
    task_fill_orders()