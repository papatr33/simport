import streamlit as st
import pandas as pd
import yfinance as yf
import bcrypt
import os
import plotly.express as px
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timedelta, date
import time

# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    email = Column(String, nullable=True)
    initial_capital = Column(Float, default=5000000.0)
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    
    # Execution Details
    entry_date = Column(DateTime, default=datetime.now)
    entry_price = Column(Float, nullable=True) # Price per share
    quantity = Column(Float, nullable=True)    # Number of shares
    trade_amount = Column(Float, nullable=True) # Total $ allocated
    
    # Exit Details
    exit_date = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, default=0.0)
    
    status = Column(String, default='PENDING') # PENDING, OPEN, CLOSED
    notes = Column(String, nullable=True)
    
    user = relationship("User", back_populates="trades")

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

# --- DB Connection ---
def get_db_engine():
    # 1. Streamlit Cloud Secrets
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    # 2. Env Vars (GitHub Actions)
    elif "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    # 3. Local Secrets File (Manual)
    else:
        # Fallback to SQLite
        db_url = 'sqlite:///portfolio.db'

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    if 'sqlite' in db_url:
        return create_engine(db_url, connect_args={'check_same_thread': False})
    else:
        return create_engine(db_url)

engine = get_db_engine()
# Attempt to create tables (will skip if they exist)
try:
    Base.metadata.create_all(engine)
except:
    pass # Handle connection errors gracefully in UI
Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# 2. ANALYTICS ENGINE
# ==========================================

def calculate_portfolio_metrics(user_id):
    """Calculates live cash, equity, and positions."""
    user = session.query(User).filter_by(id=user_id).first()
    trades = session.query(Trade).filter_by(user_id=user_id).all()
    
    cash = user.initial_capital
    portfolio_value = 0.0
    active_holdings = []
    
    for t in trades:
        # PENDING: Cash blocked
        if t.status == 'PENDING':
            cash -= t.trade_amount
            
        # OPEN: Mark to Market
        elif t.status == 'OPEN':
            cash -= t.trade_amount # Deduct initial cost
            
            curr_price = get_live_price(t.ticker)
            
            if t.direction == 'Long':
                position_value = t.quantity * curr_price
            else:
                # Short Value = Initial Capital + (Entry - Current)*Qty
                short_pnl = (t.entry_price * t.quantity) - (curr_price * t.quantity)
                position_value = t.trade_amount + short_pnl
            
            portfolio_value += position_value
            
            # PnL Calc
            if t.entry_price > 0:
                pnl_pct = ((curr_price - t.entry_price)/t.entry_price) if t.direction == 'Long' else ((t.entry_price - curr_price)/t.entry_price)
            else:
                pnl_pct = 0.0

            active_holdings.append({
                "Ticker": t.ticker,
                "Side": t.direction,
                "Date": t.entry_date.strftime("%Y-%m-%d"),
                "Qty": t.quantity,
                "Entry": t.entry_price,
                "Current": curr_price,
                "Value": position_value,
                "PnL %": pnl_pct * 100,
                "Notes": t.notes
            })
            
        # CLOSED: Cash returned
        elif t.status == 'CLOSED':
            cash += t.realized_pnl + t.trade_amount 

    total_equity = cash + portfolio_value
    return {
        "cash": cash,
        "equity": total_equity,
        "holdings": active_holdings,
        "trades": trades
    }

def generate_pnl_curve(user_id):
    """
    Reconstructs history. 
    Crucial for Test Mode: When a backdated trade is added, this function 
    finds the earliest date and downloads ALL data from then until now.
    """
    user = session.query(User).filter_by(id=user_id).first()
    trades = session.query(Trade).filter_by(user_id=user_id).all()
    
    if not trades:
        return pd.DataFrame()

    # 1. Find earliest date (handles backdated trades)
    start_date = min([t.entry_date for t in trades])
    end_date = datetime.now()
    
    # 2. Get Universe
    tickers = list(set([t.ticker for t in trades]))
    
    # 3. Batch Download
    if tickers:
        # 'Close' column only
        raw_data = yf.download(tickers, start=start_date, end=end_date)
        
        # Handle yfinance structure variations
        if 'Close' in raw_data:
            data = raw_data['Close']
        else:
            data = raw_data # Fallback
            
        # Handle Single Ticker (returns Series) vs Multiple (returns DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
    else:
        return pd.DataFrame()

    # 4. Reconstruct Daily Value
    daterange = pd.date_range(start=start_date, end=end_date, freq='B')
    curve = []
    
    for d in daterange:
        d_cash = user.initial_capital
        d_equity = 0.0
        
        for t in trades:
            # Only count trade if it existed on date 'd'
            if t.entry_date <= d:
                is_open = (t.status == 'OPEN') or (t.status == 'CLOSED' and t.exit_date > d)
                is_closed_before = (t.status == 'CLOSED' and t.exit_date <= d)

                if is_open:
                    d_cash -= t.trade_amount
                    # Value Position
                    try:
                        # Find price on date 'd'
                        # Use nearest backward fill if exact date missing (holidays)
                        if t.ticker in data.columns:
                            # Access specific ticker column
                            series = data[t.ticker]
                            # Get price at date d (or last available)
                            # We use asof to get nearest past data
                            idx = data.index.get_indexer([d], method='pad')[0]
                            if idx != -1:
                                price = series.iloc[idx]
                            else:
                                price = t.entry_price
                        else:
                            price = t.entry_price
                        
                        if t.direction == 'Long':
                            d_equity += (t.quantity * price)
                        else:
                            short_pnl = (t.entry_price * t.quantity) - (price * t.quantity)
                            d_equity += (t.trade_amount + short_pnl)
                    except:
                        d_equity += t.trade_amount 

                elif is_closed_before:
                    d_cash += t.realized_pnl
        
        total_val = d_cash + d_equity
        curve.append({"Date": d, "Portfolio Value": total_val})
        
    return pd.DataFrame(curve)

# ==========================================
# 3. HELPERS
# ==========================================
def get_live_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty: return data['Close'].iloc[-1]
        return 0.0
    except: return 0.0

def get_historical_price(ticker, date_obj):
    """Finds close price for a specific historical date."""
    try:
        start = date_obj
        end = date_obj + timedelta(days=5) # 5 day buffer for weekends/holidays
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            # Handle different yfinance return shapes
            if 'Close' in df.columns:
                return df['Close'].iloc[0]
            return df.iloc[0,0] # Fallback
        return 0.0
    except: return 0.0

def format_ticker(symbol, market):
    symbol = symbol.strip().upper()
    suffixes = {"US": "", "Hong Kong": ".HK", "China (Shanghai)": ".SS", "China (Shenzhen)": ".SZ", "Japan": ".T", "UK": ".L", "France": ".PA"}
    return f"{symbol}{suffixes.get(market, '')}"

def is_test_mode():
    try:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        return cfg.value == 'True' if cfg else False
    except:
        return False

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def init_db():
    try:
        if not session.query(User).filter_by(username='admin').first():
            pw = hash_password('8848')
            session.add(User(username='admin', password_hash=pw, role='admin', email='admin@fund.com'))
            session.commit()
        if not session.query(SystemConfig).filter_by(key='test_mode').first():
            session.add(SystemConfig(key='test_mode', value='False'))
            session.commit()
    except: pass

# ==========================================
# 4. UI PAGES
# ==========================================

def admin_page():
    st.title("🛠️ Admin Dashboard")
    
    # 1. Config Section
    st.subheader("System Configuration")
    curr_mode = is_test_mode()
    new_mode = st.toggle("Enable Test Mode (Backdating)", value=curr_mode, help="Allows analysts to input past dates for trades.")
    if new_mode != curr_mode:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        cfg.value = str(new_mode)
        session.commit()
        st.success("Configuration updated.")
        time.sleep(1)
        st.rerun()

    st.divider()

    col1, col2 = st.columns([1, 2])
    
    # 2. Create User
    with col1:
        st.subheader("Create New User")
        with st.form("create_user"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            r = st.selectbox("Role", ["analyst", "pm"])
            cap = st.number_input("Initial Capital", value=5000000.0)
            if st.form_submit_button("Create User"):
                if session.query(User).filter_by(username=u).first():
                    st.error("User already exists")
                else:
                    session.add(User(username=u, password_hash=hash_password(p), role=r, initial_capital=cap))
                    session.commit()
                    st.success(f"User {u} created!")
                    st.rerun()

    # 3. Manage Users (List & Delete)
    with col2:
        st.subheader("Manage Existing Users")
        users = session.query(User).all()
        
        if users:
            # Display Table
            user_data = []
            for usr in users:
                user_data.append({
                    "ID": usr.id,
                    "Username": usr.username,
                    "Role": usr.role,
                    "Capital": f"${usr.initial_capital:,.0f}"
                })
            
            st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)
            
            # Delete Interface
            # Filter out current admin to prevent self-deletion if logged in as generic admin
            # (Though 'admin' user is protected by logic below usually)
            delete_candidates = [usr.username for usr in users if usr.username != 'admin']
            
            if delete_candidates:
                st.write("---")
                col_del_1, col_del_2 = st.columns([2,1])
                with col_del_1:
                    user_to_delete = st.selectbox("Select User to Delete", [""] + delete_candidates)
                with col_del_2:
                    st.write("") # Spacer
                    st.write("")
                    if st.button("🗑️ Delete User", type="primary"):
                        if user_to_delete:
                            u_obj = session.query(User).filter_by(username=user_to_delete).first()
                            session.delete(u_obj)
                            session.commit()
                            st.warning(f"User {user_to_delete} has been permanently deleted.")
                            time.sleep(1)
                            st.rerun()
            else:
                st.info("No users available to delete.")
        else:
            st.info("No users found.")

def analyst_page(user):
    st.title(f"👨‍💻 {user.username} | Portfolio")
    
    # --- Metrics ---
    metrics = calculate_portfolio_metrics(user.id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Equity", f"${metrics['equity']:,.0f}")
    c2.metric("Cash Balance", f"${metrics['cash']:,.0f}")
    pnl_total = metrics['equity'] - user.initial_capital
    c3.metric("Total PnL", f"${pnl_total:,.0f}", delta_color="normal" if pnl_total >=0 else "inverse")
    
    st.divider()

    # --- PnL Curve ---
    with st.expander("📈 Historical Performance Curve", expanded=True):
        if len(metrics['trades']) > 0:
            # Only generate curve if we have trades
            df_curve = generate_pnl_curve(user.id)
            if not df_curve.empty:
                fig = px.line(df_curve, x='Date', y='Portfolio Value', title='Equity Curve')
                fig.add_hline(y=user.initial_capital, line_dash="dot", annotation_text="Initial Capital")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not retrieve historical data for chart.")
        else:
            st.info("No trades executed yet.")

    # --- Trade Entry ---
    st.subheader("📝 Enter Trade")
    
    test_mode = is_test_mode()
    if test_mode:
        st.info("🛠️ TEST MODE ON: You can backdate trades. Chart will update automatically.")

    with st.form("trade_entry"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            market = st.selectbox("Market", ["US", "Hong Kong", "China (Shanghai)", "China (Shenzhen)", "Japan", "UK", "France"])
        with col2:
            ticker_raw = st.text_input("Ticker").strip()
        with col3:
            direction = st.selectbox("Direction", ["Long", "Short"])
        with col4:
            allocation = st.number_input("Capital Allocation ($)", min_value=1000.0, step=10000.0)

        # Date Input (Only visible/used in Test Mode)
        trade_date = datetime.now()
        if test_mode:
            trade_date = st.date_input("Trade Date (Backdate)", value="today")

        notes = st.text_area("Notes")
        submit = st.form_submit_button("Submit Order")

        if submit:
            if not ticker_raw:
                st.error("Ticker required.")
            elif allocation > metrics['cash']:
                st.error(f"Insufficient Cash. Available: ${metrics['cash']:,.0f}")
            else:
                final_ticker = format_ticker(ticker_raw, market)
                
                if test_mode:
                    # BACKDATING LOGIC
                    # 1. Find price at that specific date
                    hist_date = datetime.combine(trade_date, datetime.min.time())
                    fill_price = get_historical_price(final_ticker, hist_date)
                    
                    if fill_price > 0:
                        qty = allocation / fill_price
                        new_trade = Trade(
                            user_id=user.id, ticker=final_ticker, direction=direction,
                            status='OPEN', entry_price=fill_price, quantity=qty,
                            trade_amount=allocation, entry_date=hist_date, notes=f"[BACKDATED] {notes}"
                        )
                        session.add(new_trade)
                        session.commit()
                        st.success(f"Trade Backdated! Filled at ${fill_price:.2f} on {hist_date.date()}. Refreshing chart...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"No price data found for {final_ticker} on {trade_date}. Try a different date.")
                else:
                    # LIVE LOGIC
                    new_trade = Trade(
                        user_id=user.id, ticker=final_ticker, direction=direction,
                        status='PENDING', trade_amount=allocation, notes=notes
                    )
                    session.add(new_trade)
                    session.commit()
                    st.success("Order Submitted. Will fill at next Open.")
                    time.sleep(1)
                    st.rerun()

    # --- Holdings & History ---
    tab1, tab2 = st.tabs(["Active Holdings", "Trade History"])
    
    with tab1:
        if metrics['holdings']:
            df_h = pd.DataFrame(metrics['holdings'])
            st.dataframe(df_h.style.format({
                "Value": "${:,.0f}", "Entry": "${:,.2f}", "Current": "${:,.2f}", "PnL %": "{:.2f}%", "Qty": "{:,.0f}"
            }), use_container_width=True)
        else:
            st.info("No active positions.")
            
    with tab2:
        closed_trades = [t for t in metrics['trades'] if t.status == 'CLOSED']
        if closed_trades:
            hist_data = [{
                "Ticker": t.ticker, "Side": t.direction, 
                "Entry Date": t.entry_date.strftime("%Y-%m-%d"),
                "Exit Date": t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "-",
                "Realized PnL": t.realized_pnl
            } for t in closed_trades]
            st.dataframe(pd.DataFrame(hist_data).style.format({"Realized PnL": "${:,.0f}"}), use_container_width=True)
        else:
            st.info("No closed trades.")

def pm_page(user):
    st.title("🏦 PM Dashboard")
    analysts = session.query(User).filter_by(role='analyst').all()
    
    # Aggregated View
    rows = []
    for a in analysts:
        m = calculate_portfolio_metrics(a.id)
        rows.append({
            "Analyst": a.username,
            "Equity": m['equity'],
            "Cash": m['cash'],
            "Positions": len(m['holdings']),
            "PnL": m['equity'] - a.initial_capital
        })
    
    st.subheader("Team Overview")
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"Equity": "${:,.0f}", "Cash": "${:,.0f}", "PnL": "${:,.0f}"}), use_container_width=True)
    else:
        st.info("No analysts found.")
    
    st.divider()
    
    # Individual Drilldown
    if analysts:
        selected_analyst = st.selectbox("Select Analyst for Detail", [a.username for a in analysts])
        target_user = session.query(User).filter_by(username=selected_analyst).first()
        
        if target_user:
            metrics = calculate_portfolio_metrics(target_user.id)
            
            # Plot Chart
            df_curve = generate_pnl_curve(target_user.id)
            if not df_curve.empty:
                fig = px.line(df_curve, x='Date', y='Portfolio Value', title=f"{target_user.username} Equity Curve")
                fig.add_hline(y=target_user.initial_capital, line_dash="dot", annotation_text="Initial Capital")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Analyst has no history to plot.")
                
            st.subheader("Current Holdings")
            if metrics['holdings']:
                st.dataframe(pd.DataFrame(metrics['holdings']), use_container_width=True)
            else:
                st.info("No active holdings.")

# --- MAIN ---
def main():
    st.set_page_config(layout="wide", page_title="AlphaTracker Pro")
    init_db()
    
    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    # LOGIN SCREEN
    if not st.session_state.user_id:
        c1,c2,c3=st.columns([1,1,1])
        with c2:
            st.title("AlphaTracker Pro")
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    user = session.query(User).filter_by(username=u).first()
                    if user and check_password(p, user.password_hash):
                        st.session_state.user_id = user.id
                        st.session_state.role = user.role
                        st.rerun()
                    else: st.error("Invalid Username or Password")
    # APP SCREEN
    else:
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        if not user:
            st.session_state.user_id = None
            st.rerun()
            
        with st.sidebar:
            st.write(f"Logged in as: **{user.username}**")
            st.write(f"Role: {user.role.upper()}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.rerun()
        
        if user.role == 'admin': admin_page()
        elif user.role == 'analyst': analyst_page(user)
        elif user.role == 'pm': pm_page(user)

if __name__ == "__main__":
    main()
