import socket
# IPv4 Force for GitHub Actions
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import os
import toml

# --- DB SETUP ---
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
    
    if not db_url: return create_engine('sqlite:///portfolio.db', connect_args={'check_same_thread': False})
    if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
    return create_engine(db_url)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    initial_capital = Column(Float, default=5000000.0)
    transactions = relationship("Transaction", back_populates="user")

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String)
    trans_type = Column(String)
    date = Column(DateTime)
    price = Column(Float)
    quantity = Column(Float)
    amount = Column(Float)
    status = Column(String)
    user = relationship("User", back_populates="transactions")

engine = get_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# --- LOGIC ---
def log(msg): print(f"[{datetime.now()}] {msg}")

def task_fill_orders():
    log("Checking Pending Transactions...")
    pending = session.query(Transaction).filter_by(status='PENDING').all()
    
    if not pending: log("No pending orders."); return
    
    for t in pending:
        try:
            log(f"Processing {t.trans_type} {t.ticker}...")
            # Get Price
            hist = yf.Ticker(t.ticker).history(period="5d")
            if not hist.empty:
                fill_price = float(hist['Open'].iloc[-1])
                
                # Calculate Quantity: Qty = Amount / Price
                if fill_price > 0:
                    t.price = fill_price
                    t.quantity = t.amount / fill_price
                    t.status = 'FILLED'
                    t.date = datetime.now() # Mark filled time
                    log(f"FILLED at ${fill_price:.2f} -> {t.quantity:.2f} shares")
                else:
                    log(f"Invalid price for {t.ticker}")
            else:
                log(f"No data for {t.ticker}")
        except Exception as e:
            log(f"Error {t.ticker}: {e}")
    session.commit()

if __name__ == "__main__":
    task_fill_orders()
