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

# --- NEW: Horizontal Monthly Table Logic ---
def generate_monthly_html(df_curve, initial_capital):
    if df_curve.empty: return "<p>No data</p>"
    
    # Ensure date column is datetime
    df_curve['Date'] = pd.to_datetime(df_curve['Date'])
    df_curve['Month'] = df_curve['Date'].dt.to_period('M')
    
    stats = {} # Key: Month Period, Value: {Long, Short, Total}
    
    prev_equity = initial_capital
    
    # Group by month to calculate attribution
    for month, group in df_curve.groupby('Month'):
        month_end_equity = group['Equity'].iloc[-1]
        if prev_equity == 0: prev_equity = initial_capital 
        
        # Sum daily PnL for the month
        long_pnl_sum = group['Long PnL'].sum()
        short_pnl_sum = group['Short PnL'].sum()
        
        # Calculate Returns relative to Month Start Equity
        long_ret = (long_pnl_sum / prev_equity) * 100
        short_ret = (short_pnl_sum / prev_equity) * 100
        total_ret = ((month_end_equity - prev_equity) / prev_equity) * 100
        
        stats[month] = {
            "Long": long_ret,
            "Short": short_ret,
            "Total": total_ret
        }
        prev_equity = month_end_equity # Update for next loop

    if not stats: return "<p>No data</p>"

    # Build Horizontal Table HTML
    months = list(stats.keys())
    # Format Month Header (e.g., "Jan", "Feb")
    month_headers = "".join([f"<th style='padding: 8px; text-align: center;'>{m.strftime('%b')}</th>" for m in months])
    
    # Generate Data Rows
    long_cells = "".join([f"<td style='padding: 8px; text-align: center; color: {'#10B981' if stats[m]['Long'] >= 0 else '#EF4444'};'>{stats[m]['Long']:+.2f}%</td>" for m in months])
    short_cells = "".join([f"<td style='padding: 8px; text-align: center; color: {'#10B981' if stats[m]['Short'] >= 0 else '#EF4444'};'>{stats[m]['Short']:+.2f}%</td>" for m in months])
    total_cells = "".join([f"<td style='padding: 8px; text-align: center; font-weight: bold; color: {'#10B981' if stats[m]['Total'] >= 0 else '#EF4444'};'>{stats[m]['Total']:+.2f}%</td>" for m in months])

    html = f"""
    <table border='1' cellpadding='0' cellspacing='0' style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;'>
        <tr style='background-color: #f8f9fa;'>
            <th style='padding: 8px; text-align: left; min-width: 100px;'>Metric</th>
            {month_headers}
        </tr>
        <tr>
            <td style='padding: 8px; font-weight: bold;'>Long Return</td>
            {long_cells}
        </tr>
        <tr>
            <td style='padding: 8px; font-weight: bold;'>Short Return</td>
            {short_cells}
        </tr>
        <tr style='background-color: #f1f5f9;'>
            <td style='padding: 8px; font-weight: bold;'>Total Return</td>
            {total_cells}
        </tr>
    </table>
    """
    return html

def run_weekly_report():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    # Filter: Analysts and Traders only (No PM)
    analysts = session.query(User).filter(User.role.in_(['analyst', 'trader'])).all()
    
    email_body = "<html><body style='font-family: Arial, sans-serif;'>"
    email_body += "<h2>Weekly Portfolio Report</h2>"
    
    now = datetime.now()
    two_weeks_ago = now - timedelta(days=14)

    for user in analysts:
        email_body += f"<hr style='border: 1px solid #eee; margin-top: 30px;'><h3>👤 {user.username} ({user.role})</h3>"
        
        txs = fetch_user_transactions(session, user.id)
        
        # 1. Calculate States
        state_now = calculate_portfolio_state(txs, user.initial_capital, now)
        state_prev = calculate_portfolio_state(txs, user.initial_capital, two_weeks_ago)
        
        # 2. Monthly Returns (Horizontal Table)
        df_curve, _ = get_ytd_performance(txs, user.initial_capital)
        email_body += "<h4>Monthly Returns Breakdown</h4>"
        email_body += generate_monthly_html(df_curve, user.initial_capital)

        # 3. Position Changes (Now vs 2 Weeks Ago)
        email_body += "<h4>Position Changes (Last 14 Days)</h4>"
        email_body += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; font-size: 14px;'>"
        email_body += "<tr style='background-color: #f2f2f2;'><th style='text-align: left;'>Ticker</th><th style='text-align: center;'>Type</th><th style='text-align: right;'>Change</th></tr>"

        all_tickers = set(list(state_now['positions'].keys()) + list(state_prev['positions'].keys()))
        
        has_changes = False
        for tik in sorted(list(all_tickers)): # Sorted for neatness
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
            
            # Filter: Show if position existed in either period
            if abs(pct_curr) > 0.01 or abs(pct_prev) > 0.01:
                has_changes = True
                arrow = "&nbsp;" # default space
                if pct_curr > pct_prev + 0.1: arrow = "<span style='color: green;'>⬆️</span>"
                elif pct_curr < pct_prev - 0.1: arrow = "<span style='color: red;'>⬇️</span>"
                
                email_body += f"""
                <tr>
                    <td>{tik}</td>
                    <td style='text-align: center;'>{side}</td>
                    <td style='text-align: right;'>{arrow} {pct_prev:.1f}% &rarr; {pct_curr:.1f}%</td>
                </tr>
                """
        
        if not has_changes:
             email_body += "<tr><td colspan='3'>No active positions.</td></tr>"

        email_body += "</table>"
        email_body += f"<p><strong>Current Equity:</strong> ${state_now['equity']:,.0f}</p>"

    email_body += "</body></html>"

    # SEND EMAIL
    sender_email = os.environ.get("EMAIL_SENDER")
    sender_pass = os.environ.get("EMAIL_PASSWORD")
    receiver_email = os.environ.get("EMAIL_RECEIVER")

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
