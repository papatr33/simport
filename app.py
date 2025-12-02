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
import numpy as np

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
    
    entry_date = Column(DateTime, default=datetime.now)
    entry_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=True)
    trade_amount = Column(Float, nullable=True)
    
    exit_date = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, default=0.0)
    
    status = Column(String, default='PENDING')
    notes = Column(String, nullable=True)
    
    user = relationship("User", back_populates="trades")

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

def get_db_engine():
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    elif "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    else:
        db_url = 'sqlite:///portfolio.db'

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    if 'sqlite' in db_url:
        return create_engine(db_url, connect_args={'check_same_thread': False})
    else:
        return create_engine(db_url)

engine = get_db_engine()
try:
    Base.metadata.create_all(engine)
except:
    pass
Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# 2. DATA FETCHING HELPERS (ROBUST)
# ==========================================
def extract_scalar(val):
    """Helper to ensure we get a single float from yfinance result"""
    if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, list)):
        # Flatten and take first item
        try:
            val = val.values.flatten()[0]
        except:
            if hasattr(val, 'iloc'):
                val = val.iloc[0]
            elif len(val) > 0:
                val = val[0]
    return float(val)

def get_live_price(ticker):
    try:
        # Request only 1 day of history
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            # Safe extraction
            return extract_scalar(data['Close'].iloc[-1])
        return 0.0
    except:
        return 0.0

def get_historical_price(ticker, date_obj):
    try:
        start = date_obj
        end = date_obj + timedelta(days=5) # 5 day buffer for weekends
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if not df.empty:
            if 'Close' in df.columns:
                val = df['Close'].iloc[0]
            else:
                val = df.iloc[0,0] # Fallback
            return extract_scalar(val)
        return 0.0
    except:
        return 0.0

# ==========================================
# 3. ANALYTICS ENGINE
# ==========================================

def calculate_portfolio_metrics(user_id):
    user = session.query(User).filter_by(id=user_id).first()
    trades = session.query(Trade).filter_by(user_id=user_id).all()
    
    cash = user.initial_capital
    portfolio_value = 0.0
    active_holdings = []
    
    for t in trades:
        # Safe checks for none values
        t_amt = t.trade_amount or 0.0
        t_qty = t.quantity or 0.0
        t_entry = t.entry_price or 0.0

        if t.status == 'PENDING':
            cash -= t_amt
            
        elif t.status == 'OPEN':
            cash -= t_amt
            
            curr_price = get_live_price(t.ticker)
            
            if t.direction == 'Long':
                position_value = t_qty * curr_price
            else:
                short_pnl = (t_entry * t_qty) - (curr_price * t_qty)
                position_value = t_amt + short_pnl
            
            portfolio_value += position_value
            
            if t_entry > 0:
                if t.direction == 'Long':
                    pnl_pct = (curr_price - t_entry) / t_entry
                else:
                    pnl_pct = (t_entry - curr_price) / t_entry
            else:
                pnl_pct = 0.0

            active_holdings.append({
                "Ticker": t.ticker,
                "Side": t.direction,
                "Date": t.entry_date.strftime("%Y-%m-%d"),
                "Qty": t_qty,
                "Entry": t_entry,
                "Current": curr_price,
                "Value": position_value,
                "PnL %": pnl_pct * 100,
                "Notes": t.notes
            })
            
        elif t.status == 'CLOSED':
            cash += (t.realized_pnl or 0.0) + t_amt

    total_equity = cash + portfolio_value
    return {
        "cash": cash,
        "equity": total_equity,
        "holdings": active_holdings,
        "trades": trades
    }

def generate_pnl_curve(user_id):
    user = session.query(User).filter_by(id=user_id).first()
    trades = session.query(Trade).filter_by(user_id=user_id).all()
    
    if not trades:
        return pd.DataFrame()

    start_date = min([t.entry_date for t in trades])
    end_date = datetime.now()
    tickers = list(set([t.ticker for t in trades]))
    
    if tickers:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Robust data extraction for different yfinance versions
        if 'Close' in raw_data:
            data = raw_data['Close']
        else:
            data = raw_data
            
        # Ensure data is DataFrame even if single ticker (Series)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
    else:
        return pd.DataFrame()

    daterange = pd.date_range(start=start_date, end=end_date, freq='B')
    curve = []
    
    for d in daterange:
        d_cash = user.initial_capital
        d_equity = 0.0
        
        for t in trades:
            if t.entry_date <= d:
                is_open = (t.status == 'OPEN') or (t.status == 'CLOSED' and t.exit_date > d)
                is_closed_before = (t.status == 'CLOSED' and t.exit_date <= d)

                if is_open:
                    d_cash -= (t.trade_amount or 0.0)
                    try:
                        price = 0.0
                        # Get price logic with duplicate column protection
                        if t.ticker in data.columns:
                            # Handle case where one ticker appears multiple times
                            col_data = data[t.ticker]
                            if isinstance(col_data, pd.DataFrame):
                                col_data = col_data.iloc[:, 0] # Take first if duplicate
                            
                            idx = data.index.get_indexer([d], method='pad')[0]
                            if idx != -1:
                                price = extract_scalar(col_data.iloc[idx])
                            else:
                                price = t.entry_price
                        else:
                            price = t.entry_price
                        
                        qty = t.quantity or 0.0
                        if t.direction == 'Long':
                            d_equity += (qty * price)
                        else:
                            short_pnl = (t.entry_price * qty) - (price * qty)
                            d_equity += (t.trade_amount + short_pnl)
                    except:
                        d_equity += (t.trade_amount or 0.0)

                elif is_closed_before:
                    d_cash += (t.realized_pnl or 0.0)
        
        total_val = d_cash + d_equity
        curve.append({"Date": d, "Portfolio Value": total_val})
        
    return pd.DataFrame(curve)

# ==========================================
# 4. GENERAL HELPERS
# ==========================================
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
# 5. UI PAGES
# ==========================================
def admin_page():
    st.title("🛠️ Admin Dashboard")
    st.subheader("System Configuration")
    curr_mode = is_test_mode()
    new_mode = st.toggle("Enable Test Mode (Backdating)", value=curr_mode)
    if new_mode != curr_mode:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        cfg.value = str(new_mode)
        session.commit()
        st.success("Updated!")
        time.sleep(1)
        st.rerun()
    
    st.divider()
    c1, c2 = st.columns([1,2])
    
    with c1:
        st.subheader("Create User")
        with st.form("create_user"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            r = st.selectbox("Role", ["analyst", "pm"])
            cap = st.number_input("Initial Capital", value=5000000.0)
            if st.form_submit_button("Create"):
                if session.query(User).filter_by(username=u).first():
                    st.error("User exists")
                else:
                    session.add(User(username=u, password_hash=hash_password(p), role=r, initial_capital=cap))
                    session.commit()
                    st.success("Created!")
                    st.rerun()
                    
    with c2:
        st.subheader("Manage Users")
        users = session.query(User).all()
        if users:
            data = [{"ID":x.id, "User":x.username, "Role":x.role, "Capital":f"${x.initial_capital:,.0f}"} for x in users]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            
            del_list = [u.username for u in users if u.username != 'admin']
            if del_list:
                to_del = st.selectbox("Delete User", [""] + del_list)
                if st.button("Delete Selected") and to_del:
                    session.delete(session.query(User).filter_by(username=to_del).first())
                    session.commit()
                    st.warning(f"Deleted {to_del}")
                    time.sleep(1)
                    st.rerun()

def analyst_page(user):
    st.title(f"👨‍💻 {user.username} | Portfolio")
    metrics = calculate_portfolio_metrics(user.id)
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Equity", f"${metrics['equity']:,.0f}")
    c2.metric("Cash", f"${metrics['cash']:,.0f}")
    pnl = metrics['equity'] - user.initial_capital
    c3.metric("PnL", f"${pnl:,.0f}")
    
    st.divider()
    with st.expander("📈 Performance", expanded=True):
        if metrics['trades']:
            df_c = generate_pnl_curve(user.id)
            if not df_c.empty:
                fig = px.line(df_c, x='Date', y='Portfolio Value')
                fig.add_hline(y=user.initial_capital, line_dash="dot")
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("No trades to plot.")
        
    st.subheader("Enter Trade")
    test_mode = is_test_mode()
    if test_mode: st.info("TEST MODE: Backdating Enabled")
    
    with st.form("trade"):
        c1,c2,c3,c4 = st.columns(4)
        with c1: mkt = st.selectbox("Market", ["US", "Hong Kong", "China (Shanghai)", "China (Shenzhen)", "Japan", "UK", "France"])
        with c2: tik = st.text_input("Ticker")
        with c3: direct = st.selectbox("Side", ["Long", "Short"])
        with c4: alloc = st.number_input("Amount ($)", min_value=1000.0, step=10000.0)
        
        t_date = datetime.now()
        if test_mode: t_date = st.date_input("Date", value="today")
        note = st.text_area("Notes")
        
        if st.form_submit_button("Submit"):
            if not tik: st.error("Ticker needed")
            elif alloc > metrics['cash']: st.error("Insufficient Cash")
            else:
                final_tik = format_ticker(tik, mkt)
                if test_mode:
                    h_date = datetime.combine(t_date, datetime.min.time())
                    price = get_historical_price(final_tik, h_date)
                    if price > 0:
                        qty = alloc / price
                        t = Trade(user_id=user.id, ticker=final_tik, direction=direct, status='OPEN',
                                  entry_price=price, quantity=qty, trade_amount=alloc, entry_date=h_date, notes=f"[BACKDATE] {note}")
                        session.add(t)
                        session.commit()
                        st.success(f"Filled at ${price:.2f}")
                        time.sleep(1)
                        st.rerun()
                    else: st.error("Price not found")
                else:
                    t = Trade(user_id=user.id, ticker=final_tik, direction=direct, status='PENDING', trade_amount=alloc, notes=note)
                    session.add(t)
                    session.commit()
                    st.success("Pending Open")
                    time.sleep(1)
                    st.rerun()

    st.subheader("Holdings")
    if metrics['holdings']:
        st.dataframe(pd.DataFrame(metrics['holdings']).style.format({"Value":"${:,.0f}","PnL %":"{:.2f}%"}), use_container_width=True)
    else: st.info("Empty")

def pm_page(user):
    st.title("PM Dashboard")
    analysts = session.query(User).filter_by(role='analyst').all()
    
    rows = []
    for a in analysts:
        m = calculate_portfolio_metrics(a.id)
        rows.append({"Analyst": a.username, "Equity": m['equity'], "Cash": m['cash'], "PnL": m['equity'] - a.initial_capital})
    
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"Equity":"${:,.0f}","Cash":"${:,.0f}"}), use_container_width=True)
    
    sel = st.selectbox("Analyst Detail", [a.username for a in analysts]) if analysts else None
    if sel:
        target = session.query(User).filter_by(username=sel).first()
        df_c = generate_pnl_curve(target.id)
        if not df_c.empty:
            fig = px.line(df_c, x='Date', y='Portfolio Value', title=f"{sel} Curve")
            st.plotly_chart(fig, use_container_width=True)
        
        m = calculate_portfolio_metrics(target.id)
        if m['holdings']:
            st.dataframe(pd.DataFrame(m['holdings']), use_container_width=True)

def main():
    st.set_page_config(layout="wide", page_title="AlphaTracker Pro")
    init_db()
    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    if not st.session_state.user_id:
        c1,c2,c3 = st.columns([1,1,1])
        with c2:
            st.title("AlphaTracker Login")
            with st.form("login"):
                u = st.text_input("User")
                p = st.text_input("Pass", type="password")
                if st.form_submit_button("Login"):
                    user = session.query(User).filter_by(username=u).first()
                    if user and check_password(p, user.password_hash):
                        st.session_state.user_id = user.id
                        st.session_state.role = user.role
                        st.rerun()
                    else: st.error("Invalid")
    else:
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        if not user:
            st.session_state.user_id = None
            st.rerun()
            
        with st.sidebar:
            st.write(f"User: {user.username}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.rerun()
        
        if user.role == 'admin': admin_page()
        elif user.role == 'analyst': analyst_page(user)
        elif user.role == 'pm': pm_page(user)

if __name__ == "__main__":
    main()
