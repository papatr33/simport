import streamlit as st
import pandas as pd
import yfinance as yf
import bcrypt
import os
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timedelta, date
import time
import numpy as np

# ==========================================
# 1. CONFIGURATION & CACHING
# ==========================================

# Market Configuration maps Markets to Ticker Suffixes and FX Tickers
# FX Assumption: Ticker represents "Local Units per 1 USD" (e.g. HKD=X is ~7.8)
MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD"},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD"},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY"},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY"},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY"},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP"}, # UK prices often in Pence
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR"}
}

# --- DATABASE CONNECTION (Cached) ---
@st.cache_resource(ttl="2h")
def get_db_engine():
    # 1. Streamlit Cloud Secrets
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    # 2. Env Vars (GitHub Actions)
    elif "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    # 3. Local Fallback
    else:
        db_url = 'sqlite:///portfolio.db'

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    if 'sqlite' in db_url:
        return create_engine(db_url, connect_args={'check_same_thread': False})
    else:
        return create_engine(db_url)

Base = declarative_base()

# --- DB MODELS ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False) # 'analyst', 'pm', 'admin'
    initial_capital = Column(Float, default=5000000.0)
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String, nullable=False)
    market = Column(String, nullable=True) 
    trans_type = Column(String, nullable=False) # BUY, SELL, SHORT_SELL, BUY_TO_COVER
    date = Column(DateTime, default=datetime.now)
    
    price = Column(Float, nullable=True) # USD Price
    local_price = Column(Float, nullable=True) # Local Currency Price
    quantity = Column(Float, nullable=True)
    amount = Column(Float, nullable=True) # USD Value
    
    status = Column(String, default='PENDING') # PENDING, FILLED
    notes = Column(Text, nullable=True)
    
    user = relationship("User", back_populates="transactions")

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

engine = get_db_engine()

# --- CRITICAL CHANGE: Removing silent failure ---
# If table creation fails, we want to know immediately rather than crashing later.
try:
    Base.metadata.create_all(engine)
except Exception as e:
    st.error(f"Failed to connect to Database or Create Tables: {e}")
    st.stop()

Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# 2. DATA ENGINE (ROBUST & CACHED)
# ==========================================

def extract_scalar(val):
    """Safely extracts a single float from numpy/pandas objects."""
    try:
        if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, list)):
            val = val.values.flatten()[0] if hasattr(val, 'values') else val[0]
        return float(val)
    except:
        return 0.0

@st.cache_data(ttl=300) # Cache for 5 mins
def fetch_batch_data(tickers, start_date):
    """Downloads batch data for efficiency."""
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start_date, progress=False)['Close']
        # Normalize MultiIndex vs Series
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        return data.ffill()
    except:
        return pd.DataFrame()

def get_historical_price(ticker, date_obj, market):
    """
    Fetches historical price for a specific date and converts to USD.
    Used for Backdating trades.
    """
    try:
        # 1. Download Local Price (small window to handle weekends)
        start = date_obj
        end = date_obj + timedelta(days=5)
        
        # Fetch Stock Data
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return 0.0, 0.0
        
        # Extract Close Price
        if 'Close' in df.columns:
            local_p = extract_scalar(df['Close'].iloc[0])
        else:
            local_p = extract_scalar(df.iloc[0,0])

        # 2. Handle FX Conversion
        usd_p = local_p
        
        # Check if FX needed
        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                # Fetch FX Rate for same date
                fx_df = yf.download(cfg['fx'], start=start, end=end, progress=False)
                if not fx_df.empty:
                    if 'Close' in fx_df.columns:
                        rate = extract_scalar(fx_df['Close'].iloc[0])
                    else:
                        rate = extract_scalar(fx_df.iloc[0,0])
                    
                    # UK Adjustment (Pence to Pounds)
                    if market == "UK": local_p = local_p / 100.0
                    
                    # Convert to USD (Assuming Rate is Local/USD)
                    # If Rate is 0 or nan, fail safe
                    if rate > 0:
                        usd_p = local_p / rate
                    else:
                        usd_p = 0.0 # Error in FX data

        return local_p, usd_p
    except Exception as e:
        print(f"Error fetching historical price: {e}")
        return 0.0, 0.0

# ==========================================
# 3. CORE LOGIC (PORTFOLIO STATE)
# ==========================================

def calculate_portfolio_state(user_id):
    user = session.query(User).filter_by(id=user_id).first()
    txs = session.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    
    # --- 1. Identify Data Needs ---
    # We need live prices for all held positions + FX
    # Optimization: We gather tickers during the loop, but for live marking we need them all.
    # To save time, we do the logic first, find active positions, then fetch live prices once.
    
    state = {
        "cash": user.initial_capital,
        "positions": {}, # {ticker: {qty, avg_cost, type, market, first_entry}}
        "realized_pnl_ytd": {},
        "total_realized": 0.0,
        "equity": 0.0
    }

    # --- 2. Replay Ledger ---
    for t in txs:
        tik = t.ticker
        if tik not in state["positions"]:
            state["positions"][tik] = {"qty": 0.0, "avg_cost": 0.0, "type": "FLAT", "market": t.market, "first_entry": None}
        
        pos = state["positions"][tik]
        
        if t.trans_type == "BUY":
            state["cash"] -= t.amount
            new_val = (pos["qty"] * pos["avg_cost"]) + t.amount
            pos["qty"] += t.quantity
            pos["avg_cost"] = new_val / pos["qty"] if pos["qty"] > 0 else 0.0
            pos["type"] = "LONG"
            if not pos["first_entry"]: pos["first_entry"] = t.date

        elif t.trans_type == "SELL":
            state["cash"] += t.amount
            cost_basis = t.quantity * pos["avg_cost"]
            pnl = t.amount - cost_basis
            state["total_realized"] += pnl
            state["realized_pnl_ytd"][tik] = state["realized_pnl_ytd"].get(tik, 0) + pnl
            pos["qty"] -= t.quantity
            if pos["qty"] <= 0.001: del state["positions"][tik]

        elif t.trans_type == "SHORT_SELL":
            state["cash"] += t.amount
            curr_val = abs(pos["qty"]) * pos["avg_cost"]
            new_val = curr_val + t.amount
            pos["qty"] -= t.quantity # Negative Qty
            pos["avg_cost"] = new_val / abs(pos["qty"]) if abs(pos["qty"]) > 0 else 0.0
            pos["type"] = "SHORT"
            if not pos["first_entry"]: pos["first_entry"] = t.date

        elif t.trans_type == "BUY_TO_COVER":
            state["cash"] -= t.amount
            cost_basis = t.quantity * pos["avg_cost"]
            pnl = cost_basis - t.amount
            state["total_realized"] += pnl
            state["realized_pnl_ytd"][tik] = state["realized_pnl_ytd"].get(tik, 0) + pnl
            pos["qty"] += t.quantity
            if abs(pos["qty"]) <= 0.001: del state["positions"][tik]

    # --- 3. Mark to Market (Live) ---
    active_tickers = list(state["positions"].keys())
    active_markets = {tik: pos['market'] for tik, pos in state["positions"].items()}
    
    # Collect FX needed
    fx_needed = []
    for m in active_markets.values():
        if m and MARKET_CONFIG[m]['fx']: fx_needed.append(MARKET_CONFIG[m]['fx'])
    
    # Fetch Data (Cached)
    # Using generic start date (e.g., 5 days ago) to ensure we get "today's" or "last close"
    today = datetime.now()
    batch_data = fetch_batch_data(active_tickers + fx_needed, today - timedelta(days=5))
    
    state["equity"] = state["cash"]
    
    for tik, pos in state["positions"].items():
        # Get Local Price
        usd_p = 0.0
        try:
            if not batch_data.empty:
                # Handle MultiIndex vs Single Index
                if tik in batch_data.columns:
                    raw_p = extract_scalar(batch_data[tik].iloc[-1])
                elif tik in batch_data.columns.levels[0]: # MultiIndex level
                    raw_p = extract_scalar(batch_data[tik].iloc[-1])
                else:
                    raw_p = pos["avg_cost"] # Fallback
            else:
                 raw_p = pos["avg_cost"]
                 
            # FX Conversion
            mkt = pos.get('market', 'US')
            fx_tik = MARKET_CONFIG.get(mkt, {}).get('fx')
            
            if fx_tik and fx_tik in batch_data.columns:
                 rate = extract_scalar(batch_data[fx_tik].iloc[-1])
                 if mkt == "UK": raw_p = raw_p / 100
                 usd_p = raw_p / rate if rate > 0 else 0.0
            else:
                 usd_p = raw_p
        except:
             usd_p = pos["avg_cost"] # Fallback to cost if live fail
             
        pos["current_price"] = usd_p
        
        # Calculate Value & PnL
        if pos["type"] == "LONG":
            mkt_val = pos["qty"] * usd_p
            state["equity"] += mkt_val
            pos["mkt_val"] = mkt_val
            pos["unrealized"] = mkt_val - (pos["qty"] * pos["avg_cost"])
        else:
            liability = abs(pos["qty"]) * usd_p
            state["equity"] -= liability
            pos["mkt_val"] = liability
            pos["unrealized"] = (pos["avg_cost"] - usd_p) * abs(pos["qty"])
            
    return state

def generate_curve_and_stats(user_id):
    txs = session.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    if not txs: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_date = txs[0].date
    # Fetch all data needed for history
    tickers = list(set([t.ticker for t in txs]))
    # Simplification: We only fetch Tickers, assume we use USD cost basis for history to avoid massive FX complication in history loop
    # Ideally, you'd fetch FX history too. For speed, we rely on the Transaction 'price' (USD) stored in DB for cost basis, 
    # but for market value we need history.
    
    batch_data = fetch_batch_data(tickers, start_date)
    
    user = session.query(User).filter_by(id=user_id).first()
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='B')
    
    curve = []
    monthly_stats = []
    
    for d in dates:
        curr_cash = user.initial_capital
        holdings = {}
        
        # 1. Reconstruct Portfolio Holdings at date d
        # Optimization: Pre-filter transactions
        relevant_txs = [t for t in txs if t.date <= d]
        
        for t in relevant_txs:
            if t.trans_type == 'BUY':
                curr_cash -= t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) + t.quantity
            elif t.trans_type == 'SELL':
                curr_cash += t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) - t.quantity
            elif t.trans_type == 'SHORT_SELL':
                curr_cash += t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) - t.quantity
            elif t.trans_type == 'BUY_TO_COVER':
                curr_cash -= t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) + t.quantity
        
        # 2. Value Holdings
        long_val = 0.0
        short_val = 0.0
        
        if not batch_data.empty:
            # Find closest index
            try:
                idx = batch_data.index.get_indexer([d], method='pad')[0]
                if idx != -1:
                    row = batch_data.iloc[idx]
                    
                    for tik, qty in holdings.items():
                        if abs(qty) > 0.001:
                            # Simplification: Assume historical data is USD adjusted or close enough 
                            # (Real app needs historical FX map too)
                            # For now, we take raw price. 
                            try: p = extract_scalar(row[tik])
                            except: p = 0.0
                            
                            val = qty * p
                            if qty > 0: long_val += val
                            else: short_val += abs(val) # Liability
                else:
                    pass
            except: pass
            
        total_equity = curr_cash + long_val - short_val # Wait: Cash includes short proceeds. Equity = Cash - Liability. Correct.
        
        curve.append({"Date": d, "Equity": total_equity, "Return %": ((total_equity/user.initial_capital)-1)*100})
        monthly_stats.append({"Date": d, "Longs": long_val, "Shorts": -short_val, "Total": total_equity})

    df_curve = pd.DataFrame(curve)
    df_monthly = pd.DataFrame(monthly_stats)
    
    # SPY Benchmark
    try:
        spy = fetch_batch_data(["SPY"], start_date)
        if not spy.empty:
            spy_start = extract_scalar(spy.iloc[0])
            spy_ret = ((spy['SPY'] / spy_start) - 1) * 100
        else:
            spy_ret = pd.Series()
    except:
        spy_ret = pd.Series()
        
    return df_curve, spy_ret, df_monthly

# ==========================================
# 4. HELPERS
# ==========================================
def format_ticker(symbol, market):
    symbol = symbol.strip().upper()
    suffix = MARKET_CONFIG.get(market, {}).get('suffix', '')
    return f"{symbol}{suffix}"

def is_test_mode():
    try: cfg = session.query(SystemConfig).filter_by(key='test_mode').first(); return cfg.value == 'True' if cfg else False
    except: return False
def hash_password(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
def check_password(p, h): return bcrypt.checkpw(p.encode(), h.encode())

def init_db():
    try:
        if not session.query(User).filter_by(username='admin').first():
            session.add(User(username='admin', password_hash=hash_password('8848'), role='admin'))
            session.commit()
        if not session.query(SystemConfig).filter_by(key='test_mode').first():
            session.add(SystemConfig(key='test_mode', value='False'))
            session.commit()
    except: pass

# ==========================================
# 5. UI PAGES
# ==========================================

def analyst_page(user):
    st.title(f"Portfolio: {user.username}")
    
    # --- METRICS ---
    state = calculate_portfolio_state(user.id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity (USD)", f"${state['equity']:,.0f}")
    c2.metric("Cash (USD)", f"${state['cash']:,.0f}")
    c3.metric("YTD PnL", f"${state['equity'] - user.initial_capital:,.0f}")
    
    st.divider()

    # --- CHARTS & MONTHLY ---
    df_c, spy_c, df_m = generate_curve_and_stats(user.id)
    
    t1, t2 = st.tabs(["Performance Chart", "Monthly Breakdown"])
    
    with t1:
        if not df_c.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['Return %'], name="Portfolio", line=dict(color='blue')))
            if not spy_c.empty:
                fig.add_trace(go.Scatter(x=spy_c.index, y=spy_c.values, name="SPY", line=dict(color='gray', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No history.")
        
    with t2:
        if not df_m.empty:
            df_m.set_index('Date', inplace=True)
            monthly = df_m.resample('ME').last()
            monthly['Return %'] = monthly['Total'].pct_change() * 100
            st.dataframe(monthly[['Total', 'Return %']].style.format({"Total":"${:,.0f}", "Return %":"{:.2f}%"}), use_container_width=True)
        else: st.info("No data.")

    # --- HOLDINGS ---
    st.subheader("Holdings & Attribution")
    
    # Prepare Data
    holdings_data = []
    for tik, pos in state['positions'].items():
        holdings_data.append({
            "Ticker": tik, "Side": pos['type'], "Qty": pos['qty'], 
            "Avg Cost": pos['avg_cost'], "Current": pos.get('current_price',0),
            "Unrealized": pos.get('unrealized',0),
            "Realized YTD": state['realized_pnl_ytd'].get(tik, 0)
        })
    # Add closed
    for tik, pnl in state['realized_pnl_ytd'].items():
        if tik not in state['positions']:
            holdings_data.append({
                "Ticker": tik, "Side": "FLAT", "Qty": 0, "Avg Cost": 0, "Current": 0,
                "Unrealized": 0, "Realized YTD": pnl
            })
            
    if holdings_data:
        h_df = pd.DataFrame(holdings_data)
        h_df['Total PnL'] = h_df['Unrealized'] + h_df['Realized YTD']
        st.dataframe(h_df.style.format({
            "Avg Cost":"${:,.2f}", "Current":"${:,.2f}", 
            "Unrealized":"${:,.0f}", "Realized YTD":"${:,.0f}", "Total PnL":"${:,.0f}"
        }), use_container_width=True)

    # --- ORDER ENTRY ---
    st.divider()
    st.subheader("Trade Execution")
    
    with st.expander("Compliance Rules", expanded=False):
        st.markdown("""
        * **Longs:** \$500k - \$2M (Max 5 names)
        * **Shorts:** \$300k - \$1.2M (Max 3 names)
        * **Lockup:** > 30 Days hold
        * **Cash:** < \$1.5M target
        """)

    with st.form("order"):
        c1,c2,c3,c4 = st.columns(4)
        mkt = c1.selectbox("Market", list(MARKET_CONFIG.keys()))
        tik = c2.text_input("Ticker").strip()
        side = c3.selectbox("Action", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
        amt = c4.number_input("Amount (USD)", min_value=10000.0, step=10000.0)
        
        test = is_test_mode()
        d_val = st.date_input("Date", value="today") if test else datetime.now()
        note = st.text_area("Notes")
        
        if st.form_submit_button("Submit"):
            final_tik = format_ticker(tik, mkt)
            valid = True
            msg = ""
            
            # --- COMPLIANCE ---
            # 1. Counts
            l_count = len([p for p in state['positions'].values() if p['type']=='LONG'])
            s_count = len([p for p in state['positions'].values() if p['type']=='SHORT'])
            
            if side == 'BUY' and final_tik not in state['positions'] and l_count >= 5:
                valid = False; msg = "Max Longs (5)"
            if side == 'SHORT_SELL' and final_tik not in state['positions'] and s_count >= 3:
                valid = False; msg = "Max Shorts (3)"
                
            # 2. Sizes (Opening trades only)
            if side in ['BUY', 'SHORT_SELL']:
                curr_sz = state['positions'].get(final_tik, {}).get('mkt_val', 0)
                new_sz = abs(curr_sz) + amt
                if side == 'BUY' and not (500000 <= new_sz <= 2000000):
                    valid = False; msg = f"Long Size limits ($500k-$2M). Proj: ${new_sz:,.0f}"
                if side == 'SHORT_SELL' and not (300000 <= new_sz <= 1200000):
                    valid = False; msg = f"Short Size limits ($300k-$1.2M). Proj: ${new_sz:,.0f}"

            # 3. Lockup (Closing trades)
            if side in ['SELL', 'BUY_TO_COVER']:
                entry = state['positions'].get(final_tik, {}).get('first_entry')
                check_d = datetime.combine(d_val, datetime.min.time()) if test else datetime.now()
                if entry and (check_d - entry).days < 30:
                    valid = False; msg = f"Held < 30 Days (Entry: {entry.date()})"
            
            if not valid:
                st.error(msg)
            else:
                # EXECUTION
                if test:
                    h_date = datetime.combine(d_val, datetime.min.time())
                    local_p, usd_p = get_historical_price(final_tik, h_date, mkt)
                    
                    if usd_p > 0:
                        qty = amt / usd_p
                        session.add(Transaction(
                            user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                            status='FILLED', date=h_date, amount=amt, quantity=qty,
                            local_price=local_p, price=usd_p, notes=f"[BACKDATE] {note}"
                        ))
                        session.commit()
                        st.success(f"Filled @ ${usd_p:.2f}")
                        st.cache_data.clear()
                        time.sleep(1); st.rerun()
                    else: st.error("Price not found")
                else:
                    session.add(Transaction(
                        user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                        status='PENDING', amount=amt, notes=note
                    ))
                    session.commit()
                    st.success("Pending Open")
                    time.sleep(1); st.rerun()

def pm_page(user):
    st.title("PM Dashboard")
    analysts = session.query(User).filter_by(role='analyst').all()
    
    summary = []
    for a in analysts:
        s = calculate_portfolio_state(a.id)
        summary.append({
            "Analyst": a.username, "Equity": s['equity'], 
            "Cash": s['cash'], "YTD PnL": s['equity'] - a.initial_capital
        })
    st.dataframe(pd.DataFrame(summary).style.format({"Equity":"${:,.0f}", "YTD PnL":"${:,.0f}"}), use_container_width=True)
    
    st.divider()
    sel = st.selectbox("Select Analyst", [a.username for a in analysts])
    if sel:
        target = session.query(User).filter_by(username=sel).first()
        analyst_page(target)

def admin_page():
    st.title("Admin")
    # Toggle Test Mode
    curr = is_test_mode()
    if st.toggle("Test Mode", value=curr) != curr:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        cfg.value = str(not curr); session.commit(); st.rerun()
    
    c1, c2 = st.columns([1,2])
    with c1:
        st.subheader("New User")
        with st.form("u"):
            u=st.text_input("User"); p=st.text_input("Pass",type="password"); r=st.selectbox("Role",["analyst","pm"])
            if st.form_submit_button("Create"):
                session.add(User(username=u,password_hash=hash_password(p),role=r));session.commit();st.rerun()
    with c2:
        st.subheader("Users")
        users = session.query(User).all()
        st.dataframe(pd.DataFrame([{"User":x.username,"Role":x.role} for x in users]), hide_index=True)
        to_del = st.selectbox("Delete", [u.username for u in users if u.username!='admin'])
        if st.button("Delete"):
            session.delete(session.query(User).filter_by(username=to_del).first()); session.commit(); st.rerun()

def main():
    st.set_page_config(layout="wide", page_title="AlphaTracker Pro")
    init_db()
    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    if not st.session_state.user_id:
        st.title("Login"); 
        u = st.text_input("User"); p = st.text_input("Pass", type="password")
        if st.button("Login"):
            user = session.query(User).filter_by(username=u).first()
            if user and check_password(p, user.password_hash):
                st.session_state.user_id = user.id; st.session_state.role = user.role; st.rerun()
    else:
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        if not user: st.session_state.user_id = None; st.rerun()
        
        with st.sidebar:
            st.write(f"User: {user.username}")
            if st.button("Logout"): st.session_state.user_id = None; st.rerun()
        
        if user.role=='admin': admin_page()
        elif user.role=='analyst': analyst_page(user)
        elif user.role=='pm': pm_page(user)

if __name__ == "__main__": main()
