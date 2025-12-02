import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Email Config (Can also use os.environ here for security)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@fund.com") 
ADMIN_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_password")

# ==========================================
# DATABASE SETUP (Must match app.py)
# ==========================================
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String)
    trades = relationship("Trade", back_populates="user")

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String)
    direction = Column(String)
    status = Column(String)
    cost_basis = Column(Float)
    user = relationship("User", back_populates="trades")

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

# Connect to DB using Environment Variable (GitHub Actions style)
if "DATABASE_URL" in os.environ:
    db_url = os.environ["DATABASE_URL"]
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
    print(">>> Connected to Cloud Database (Supabase)")
else:
    # Fallback for local testing
    engine = create_engine('sqlite:///portfolio.db', connect_args={'check_same_thread': False})
    print(">>> Connected to Local SQLite")

Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# LOGIC
# ==========================================

def log(msg):
    print(f"[{datetime.now()}] {msg}")

def send_alert_email(to_email, ticker, pnl_pct, direction, current_price, entry_price):
    subject = f"⚠️ STOP LOSS ALERT: {ticker}"
    body = f"""
    ACTION REQUIRED
    
    Position: {ticker} ({direction})
    Current PnL: {pnl_pct:.2f}% (Threshold: -20%)
    
    Entry Price: ${entry_price:.2f}
    Current Price: ${current_price:.2f}
    
    Please review your position immediately.
    """
    
    print("------------------------------------------------")
    print(f"📧 EMAIL TO: {to_email}")
    print(f"SUBJECT: {subject}")
    print(body)
    print("------------------------------------------------")
    
    # UNCOMMENT TO ENABLE REAL EMAILS (Requires Env Vars set in GitHub)
    # if ADMIN_PASSWORD != "your_password":
    #     try:
    #         msg = MIMEText(body)
    #         msg['Subject'] = subject
    #         msg['From'] = ADMIN_EMAIL
    #         msg['To'] = to_email
    #         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
    #             server.starttls()
    #             server.login(ADMIN_EMAIL, ADMIN_PASSWORD)
    #             server.send_message(msg)
    #     except Exception as e:
    #         log(f"Failed to send email: {e}")

def task_fill_orders():
    """Checks PENDING trades and fills them at today's OPEN price."""
    log("Checking for Pending Orders...")
    pending_trades = session.query(Trade).filter_by(status='PENDING').all()
    
    if not pending_trades:
        log("No pending orders found.")
        return

    for trade in pending_trades:
        try:
            log(f"Processing {trade.ticker}...")
            # Fetch OHLC data 
            ticker = yf.Ticker(trade.ticker)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                log(f"Skipping {trade.ticker}: No market data found.")
                continue

            # Assuming script runs post-open, take latest candle open
            todays_data = hist.iloc[-1]
            fill_price = todays_data['Open']
            
            trade.cost_basis = fill_price
            trade.status = 'OPEN'
            
            log(f"Filled {trade.ticker} {trade.direction} at ${fill_price:.2f}")
            
        except Exception as e:
            log(f"Error filling {trade.ticker}: {e}")
    
    session.commit()

def task_check_stops():
    """Checks OPEN trades. If PnL <= -20%, sends email."""
    log("Checking Stop Losses...")
    active_trades = session.query(Trade).filter_by(status='OPEN').all()
    
    for trade in active_trades:
        try:
            # Get live price
            hist = yf.Ticker(trade.ticker).history(period="1d")
            if hist.empty: continue
            
            current_price = hist['Close'].iloc[-1]
            
            # Calc PnL
            if trade.direction == 'Long':
                pnl = (current_price - trade.cost_basis) / trade.cost_basis
            else:
                pnl = (trade.cost_basis - current_price) / trade.cost_basis
            
            # Check Threshold (-0.20 = -20%)
            if pnl <= -0.20:
                if trade.user.email:
                    send_alert_email(
                        trade.user.email, 
                        trade.ticker, 
                        pnl*100, 
                        trade.direction, 
                        current_price, 
                        trade.cost_basis
                    )
                else:
                    log(f"Stop loss hit for {trade.ticker} but analyst has no email.")

        except Exception as e:
            log(f"Error checking {trade.ticker}: {e}")

if __name__ == "__main__":
    log("=== STARTING DAILY JOB ===")
    task_fill_orders()
    task_check_stops()
    log("=== JOB COMPLETE ===")