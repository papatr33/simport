import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timedelta
import pandas as pd
import toml

# Import shared logic
from core_logic import calculate_portfolio_state, get_ytd_performance

# --- DB CONFIG (Reuse from daily_job) ---
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    role = Column(String, nullable=False)
    initial_capital = Column(Float, default=5000000.0)
    transactions = relationship("Transaction", back_populates="user")

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
    user = relationship("User", back_populates="transactions")

def get_engine():
    if "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    else:
        # Fallback for local testing
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            if "DATABASE_URL" in secrets.get("general", {}): db_url = secrets["general"]["DATABASE_URL"]
            elif "DATABASE_URL" in secrets: db_url = secrets["DATABASE_URL"]
        except: db_url = 'sqlite:///portfolio.db'

    if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
    return create_engine(db_url)

def fetch_user_transactions(session, user_id):
    txs = session.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    return [{
        'ticker': t.ticker, 'market': t.market, 'trans_type': t.trans_type,
        'date': t.date, 'amount': t.amount, 'quantity': t.quantity,
        'local_price': t.local_price, 'price': t.price
    } for t in txs]

def generate_monthly_html(df_curve, initial_capital):
    if df_curve.empty: return "<p>No data</p>"
    df_curve['Month'] = df_curve['Date'].dt.to_period('M')
    monthly_stats = []
    prev_equity = initial_capital
    for month, group in df_curve.groupby('Month'):
        month_end_equity = group['Equity'].iloc[-1]
        if prev_equity == 0: prev_equity = initial_capital
        total_ret = (month_end_equity - prev_equity) / prev_equity
        monthly_stats.append(f"<tr><td>{month}</td><td>{total_ret*100:+.2f}%</td></tr>")
        prev_equity = month_end_equity
    
    return f"""
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;"><th>Month</th><th>Return</th></tr>
        {''.join(monthly_stats)}
    </table>
    """

def run_weekly_report():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    analysts = session.query(User).filter(User.role.in_(['analyst', 'trader'])).all()
    
    email_body = "<html><body><h2>Weekly Portfolio Report</h2>"
    
    now = datetime.now()
    two_weeks_ago = now - timedelta(days=14)

    for user in analysts:
        email_body += f"<hr><h3>👤 {user.username} ({user.role})</h3>"
        
        txs = fetch_user_transactions(session, user.id)
        
        # 1. Calculate States
        state_now = calculate_portfolio_state(txs, user.initial_capital, now)
        state_prev = calculate_portfolio_state(txs, user.initial_capital, two_weeks_ago)
        
        # 2. Monthly Returns
        df_curve, _ = get_ytd_performance(txs, user.initial_capital)
        email_body += "<h4>Monthly Returns</h4>"
        email_body += generate_monthly_html(df_curve, user.initial_capital)

        # 3. Position Changes (Now vs 2 Weeks Ago)
        email_body += "<h4>Position Changes (Last 14 Days)</h4>"
        email_body += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
        email_body += "<tr style='background-color: #f2f2f2;'><th>Ticker</th><th>Type</th><th>Size 2W Ago</th><th>Size Now</th><th>Change</th></tr>"

        all_tickers = set(list(state_now['positions'].keys()) + list(state_prev['positions'].keys()))
        
        for tik in all_tickers:
            # Get Prev Stats
            p_prev = state_prev['positions'].get(tik, {})
            val_prev = p_prev.get('mkt_val', 0)
            equity_prev = state_prev['equity'] if state_prev['equity'] > 0 else 1
            pct_prev = (val_prev / equity_prev) * 100
            
            # Get Curr Stats
            p_curr = state_now['positions'].get(tik, {})
            val_curr = p_curr.get('mkt_val', 0)
            equity_curr = state_now['equity'] if state_now['equity'] > 0 else 1
            pct_curr = (val_curr / equity_curr) * 100
            
            # Formatting logic
            side = p_curr.get('type') or p_prev.get('type') or "FLAT"
            
            if abs(pct_curr - pct_prev) > 0.1: # Only show meaningful changes
                arrow = "unch"
                if pct_curr > pct_prev: arrow = "⬆️"
                elif pct_curr < pct_prev: arrow = "⬇️"
                
                email_body += f"""
                <tr>
                    <td>{tik}</td>
                    <td>{side}</td>
                    <td>{pct_prev:.1f}%</td>
                    <td>{pct_curr:.1f}%</td>
                    <td>{arrow} {pct_prev:.1f}% &rarr; {pct_curr:.1f}%</td>
                </tr>
                """
        
        email_body += "</table>"
        email_body += f"<p><strong>Current Equity:</strong> ${state_now['equity']:,.0f}</p>"

    email_body += "</body></html>"

    # SEND EMAIL
    sender_email = os.environ.get("EMAIL_SENDER") # e.g., "mybot@gmail.com"
    sender_pass = os.environ.get("EMAIL_PASSWORD") # App Password
    receiver_email = os.environ.get("EMAIL_RECEIVER") # "boss@fund.com"

    if sender_email and sender_pass and receiver_email:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"AlphaTracker Weekly Report - {now.strftime('%Y-%m-%d')}"
        msg.attach(MIMEText(email_body, 'html'))

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_pass)
                server.send_message(msg)
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")
    else:
        print("Skipping Email: Credentials not found in environment variables.")

if __name__ == "__main__":
    run_weekly_report()
