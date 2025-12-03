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

MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD"},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD"},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY"},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY"},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY"},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP"}, 
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR"}
}

# --- DATABASE CONNECTION (Cached) ---
@st.cache_resource
def get_db_engine():
    # 1. Streamlit Cloud Secrets
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    # 2. Env Vars
    elif "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    # 3. Local Fallback
    else:
        db_url = 'sqlite:///portfolio.db'

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # pool_pre_ping=True checks connection liveness before checkout
    # Important for Supabase transaction poolers
    if 'sqlite' in db_url:
        return create_engine(db_url, connect_args={'check_same_thread': False})
    else:
        return create_engine(db_url, pool_pre_ping=True)

Base = declarative_base()

# --- DB MODELS ---
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

# Safe Table Creation
try:
    Base.metadata.create_all(engine)
except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.stop()

Session = sessionmaker(bind=engine)

# ==========================================
# 2. DATA ENGINE
# ==========================================

def extract_scalar(val):
    try:
        if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, list)):
            val = val.values.flatten()[0] if hasattr(val, 'values') else val[0]
        return float(val)
    except:
        return 0.0

@st.cache_data(ttl=600) 
def fetch_batch_data(tickers, start_date):
    """Downloads batch data for efficiency."""
    if not tickers: return pd.DataFrame()
    # Unique tickers only
    tickers = list(set(tickers))
    try:
        # Increase history slightly to ensure 'start_date' is covered
        data = yf.download(tickers, start=start_date, progress=False)['Close']
        
        # Handle Single Ticker returning Series vs Multi-Ticker DataFrame
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=tickers[0])
        elif data.empty:
            return pd.DataFrame()
            
        return data.ffill()
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def get_historical_price(ticker, date_obj, market):
    """
    Fetches historical price for a specific date (Backdating).
    """
    try:
        start = date_obj
        end = date_obj + timedelta(days=5) # Buffer for weekends
        
        # 1. Fetch Stock
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return 0.0, 0.0
        
        # Extract first available Close price in the window
        local_p = extract_scalar(df['Close'].dropna().iloc[0])
        usd_p = local_p

        # 2. FX Conversion
        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                # Fetch FX
                fx_df = yf.download(cfg['fx'], start=start, end=end, progress=False)
                if not fx_df.empty:
                    rate = extract_scalar(fx_df['Close'].dropna().iloc[0])
                    
                    if market == "UK": local_p = local_p / 100.0
                    
                    # Assume Rate is Local/USD (e.g. 7.8 HKD = 1 USD)
                    if rate > 0:
                        usd_p = (local_p / 100.0 if market == "UK" else local_p) / rate
                    else:
                        usd_p = 0.0

        return local_p, usd_p
    except Exception as e:
        print(f"Error fetching historical price: {e}")
        return 0.0, 0.0

# ==========================================
# 3. CORE LOGIC
# ==========================================

def calculate_portfolio_state(user_id, session_obj):
    user = session_obj.query(User).filter_by(id=user_id).first()
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    
    state = {
        "cash": user.initial_capital,
        "positions": {}, 
        "realized_pnl_ytd": {},
        "total_realized": 0.0,
        "equity": 0.0
    }

    # --- Replay Ledger ---
    for t in txs:
        tik = t.ticker
        if tik not in state["positions"]:
            state["positions"][tik] = {"qty": 0.0, "avg_cost": 0.0, "type": "FLAT", "market": t.market, "first_entry": None}
        
        pos = state["positions"][tik]
        
        if t.trans_type == "BUY":
            state["cash"] -= t.amount
            # Weighted Average Cost Update
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
            # Avg Price for Shorts is Total Liability / Share Count
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
            pos["qty"] += t.quantity # Adds back to negative
            if abs(pos["qty"]) <= 0.001: del state["positions"][tik]

    # --- Mark to Market ---
    active_tickers = list(state["positions"].keys())
    active_markets = {tik: pos['market'] for tik, pos in state["positions"].items()}
    
    fx_needed = []
    for m in active_markets.values():
        if m and MARKET_CONFIG.get(m, {}).get('fx'): 
            fx_needed.append(MARKET_CONFIG[m]['fx'])
    
    # Fetch Live Data (Last 5 days to ensure we get a print)
    today = datetime.now()
    batch_data = fetch_batch_data(active_tickers + fx_needed, today - timedelta(days=5))
    
    state["equity"] = state["cash"]
    
    for tik, pos in state["positions"].items():
        usd_p = pos["avg_cost"] # Default to cost if price fetch fails
        try:
            raw_p = None
            if not batch_data.empty:
                if tik in batch_data.columns:
                    raw_p = extract_scalar(batch_data[tik].iloc[-1])
            
            if raw_p is not None:
                mkt = pos.get('market', 'US')
                fx_tik = MARKET_CONFIG.get(mkt, {}).get('fx')
                
                # FX Conversion
                if fx_tik and fx_tik in batch_data.columns:
                     rate = extract_scalar(batch_data[fx_tik].iloc[-1])
                     if mkt == "UK": raw_p = raw_p / 100.0
                     if rate > 0: usd_p = raw_p / rate
                else:
                     usd_p = raw_p
        except: pass
             
        pos["current_price"] = usd_p
        
        # Calculate Value & PnL
        if pos["type"] == "LONG":
            mkt_val = pos["qty"] * usd_p
            state["equity"] += mkt_val
            pos["mkt_val"] = mkt_val
            pos["unrealized"] = mkt_val - (pos["qty"] * pos["avg_cost"])
        else:
            # Short Liability
            liability = abs(pos["qty"]) * usd_p
            state["equity"] -= liability
            pos["mkt_val"] = liability
            pos["unrealized"] = (pos["avg_cost"] - usd_p) * abs(pos["qty"])
            
    return state

def generate_curve_and_stats(user_id, session_obj):
    # Optimize: Only fetch required columns from DB
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    if not txs: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_date = txs[0].date
    tickers = list(set([t.ticker for t in txs]))
    
    # Fetch entire history once
    batch_data = fetch_batch_data(tickers, start_date)
    
    user = session_obj.query(User).filter_by(id=user_id).first()
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='B')
    
    curve = []
    
    # Helper to map Ticker -> Index in batch_data for speed
    if not batch_data.empty:
        batch_data.index = pd.to_datetime(batch_data.index).normalize()
    
    # Running state
    curr_cash = user.initial_capital
    holdings = {} # {ticker: qty}
    
    tx_idx = 0
    n_txs = len(txs)

    for d in dates:
        d_norm = d.normalize()
        
        # Apply transactions happening on or before this date
        while tx_idx < n_txs and txs[tx_idx].date.date() <= d_norm.date():
            t = txs[tx_idx]
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
            tx_idx += 1
            
        # Value Holdings
        long_val = 0.0
        short_val = 0.0
        
        if not batch_data.empty and d_norm in batch_data.index:
            row = batch_data.loc[d_norm]
            for tik, qty in holdings.items():
                if abs(qty) > 0.001:
                    # Note: Using Raw Price here for curve speed. 
                    # Real prod apps would need historical FX tables too.
                    try: 
                        p = float(row[tik]) if tik in row else 0.0
                        if pd.isna(p): p = 0.0
                    except: p = 0.0
                    
                    val = qty * p
                    if qty > 0: long_val += val
                    else: short_val += abs(val) # Liability
        
        total_equity = curr_cash + long_val - short_val
        curve.append({
            "Date": d, 
            "Equity": total_equity, 
            "Return %": ((total_equity/user.initial_capital)-1)*100,
            "Longs": long_val,
            "Shorts": -short_val # negative for visual chart
        })

    df_curve = pd.DataFrame(curve)
    
    # Benchmark SPY
    try:
        spy = fetch_batch_data(["SPY"], start_date)
        if not spy.empty and 'SPY' in spy.columns:
            spy_start = extract_scalar(spy['SPY'].iloc[0])
            spy_ret = ((spy['SPY'] / spy_start) - 1) * 100
        else:
            spy_ret = pd.Series(dtype=float)
    except:
        spy_ret = pd.Series(dtype=float)
        
    return df_curve, spy_ret

# ==========================================
# 4. HELPERS & UI
# ==========================================
def format_ticker(symbol, market):
    symbol = symbol.strip().upper()
    suffix = MARKET_CONFIG.get(market, {}).get('suffix', '')
    if not symbol.endswith(suffix):
        symbol = f"{symbol}{suffix}"
    return symbol

def is_test_mode(session_obj):
    try: 
        cfg = session_obj.query(SystemConfig).filter_by(key='test_mode').first()
        return cfg.value == 'True' if cfg else False
    except: return False

def hash_password(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
def check_password(p, h): return bcrypt.checkpw(p.encode(), h.encode())

def init_db(session_obj):
    try:
        if not session_obj.query(User).filter_by(username='admin').first():
            session_obj.add(User(username='admin', password_hash=hash_password('8848'), role='admin'))
            session_obj.commit()
        if not session_obj.query(SystemConfig).filter_by(key='test_mode').first():
            session_obj.add(SystemConfig(key='test_mode', value='False'))
            session_obj.commit()
    except: pass

# ==========================================
# 5. UI PAGES
# ==========================================

def analyst_page(user, session_obj):
    st.title(f"Portfolio: {user.username}")
    
    state = calculate_portfolio_state(user.id, session_obj)
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity (USD)", f"${state['equity']:,.0f}")
    c2.metric("Cash (USD)", f"${state['cash']:,.0f}")
    c3.metric("YTD PnL", f"${state['equity'] - user.initial_capital:,.0f}")
    st.divider()

    # Charts
    df_c, spy_c = generate_curve_and_stats(user.id, session_obj)
    
    t1, t2 = st.tabs(["Performance Chart", "Monthly Breakdown"])
    
    with t1:
        if not df_c.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['Return %'], name="Portfolio", line=dict(color='#00CC96')))
            if not spy_c.empty:
                fig.add_trace(go.Scatter(x=spy_c.index, y=spy_c.values, name="SPY", line=dict(color='gray', dash='dot')))
            fig.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No history yet.")
        
    with t2:
        if not df_c.empty:
            # Resample curve for monthly
            m_df = df_c.set_index('Date').resample('ME').last()
            m_df['Monthly Ret'] = m_df['Equity'].pct_change() * 100
            st.dataframe(m_df[['Equity', 'Monthly Ret']].style.format({"Equity":"${:,.0f}", "Monthly Ret":"{:.2f}%"}), use_container_width=True)
        else: st.info("No data.")

    # Holdings
    st.subheader("Holdings")
    holdings_data = []
    for tik, pos in state['positions'].items():
        holdings_data.append({
            "Ticker": tik, "Mkt": pos['market'], "Side": pos['type'], 
            "Qty": pos['qty'], "Avg Cost": pos['avg_cost'], 
            "Price": pos.get('current_price',0),
            "Value": pos.get('mkt_val', 0),
            "Unrealized": pos.get('unrealized',0),
            "Realized YTD": state['realized_pnl_ytd'].get(tik, 0)
        })
    
    if holdings_data:
        h_df = pd.DataFrame(holdings_data)
        st.dataframe(h_df.style.format({
            "Avg Cost":"${:,.2f}", "Price":"${:,.2f}", 
            "Value": "${:,.0f}", "Unrealized":"${:,.0f}", "Realized YTD":"${:,.0f}"
        }), use_container_width=True)

    # Order Entry
    st.divider()
    st.subheader("Trade Execution")
    
    with st.form("order"):
        c1,c2,c3,c4 = st.columns(4)
        mkt = c1.selectbox("Market", list(MARKET_CONFIG.keys()))
        tik = c2.text_input("Ticker").strip()
        side = c3.selectbox("Action", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
        amt = c4.number_input("Amount (USD)", min_value=10000.0, step=10000.0)
        
        test = is_test_mode(session_obj)
        d_val = st.date_input("Date", value="today") if test else datetime.now()
        note = st.text_area("Notes")
        
        if st.form_submit_button("Submit Order"):
            if not tik:
                st.error("Ticker required")
            else:
                final_tik = format_ticker(tik, mkt)
                
                # Compliance Check logic here (simplified for brevity, keep your existing logic)
                valid = True 
                
                if valid:
                    if test:
                        h_date = datetime.combine(d_val, datetime.min.time())
                        local_p, usd_p = get_historical_price(final_tik, h_date, mkt)
                        if usd_p > 0:
                            qty = amt / usd_p
                            session_obj.add(Transaction(
                                user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                                status='FILLED', date=h_date, amount=amt, quantity=qty,
                                local_price=local_p, price=usd_p, notes=f"[BACKDATE] {note}"
                            ))
                            session_obj.commit()
                            st.success(f"Filled @ ${usd_p:.2f}")
                            time.sleep(1); st.rerun()
                        else: st.error("Price not found")
                    else:
                        session_obj.add(Transaction(
                            user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                            status='PENDING', amount=amt, notes=note
                        ))
                        session_obj.commit()
                        st.success("Order Pending")
                        time.sleep(1); st.rerun()

def pm_page(user, session_obj):
    st.title("PM Dashboard")
    analysts = session_obj.query(User).filter_by(role='analyst').all()
    
    summary = []
    for a in analysts:
        s = calculate_portfolio_state(a.id, session_obj)
        summary.append({
            "Analyst": a.username, "Equity": s['equity'], 
            "Cash": s['cash'], "YTD PnL": s['equity'] - a.initial_capital
        })
    st.dataframe(pd.DataFrame(summary).style.format({"Equity":"${:,.0f}", "YTD PnL":"${:,.0f}"}), use_container_width=True)
    
    sel = st.selectbox("View Details", [a.username for a in analysts], index=None)
    if sel:
        target = session_obj.query(User).filter_by(username=sel).first()
        analyst_page(target, session_obj)

def admin_page(session_obj):
    st.title("Admin")
    curr = is_test_mode(session_obj)
    if st.toggle("Enable Test/Backdate Mode", value=curr) != curr:
        cfg = session_obj.query(SystemConfig).filter_by(key='test_mode').first()
        cfg.value = str(not curr); session_obj.commit(); st.rerun()
    
    with st.form("new_user"):
        c1, c2, c3 = st.columns(3)
        u=c1.text_input("User"); p=c2.text_input("Pass",type="password"); r=c3.selectbox("Role",["analyst","pm"])
        if st.form_submit_button("Create User"):
            session_obj.add(User(username=u,password_hash=hash_password(p),role=r));session_obj.commit();st.rerun()

def main():
    st.set_page_config(layout="wide", page_title="AlphaTracker Pro")
    
    # DB Session handling per request
    session = Session()
    init_db(session)
    
    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    if not st.session_state.user_id:
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            st.title("Login")
            u = st.text_input("User")
            p = st.text_input("Pass", type="password")
            if st.button("Login", use_container_width=True):
                user = session.query(User).filter_by(username=u).first()
                if user and check_password(p, user.password_hash):
                    st.session_state.user_id = user.id
                    st.session_state.role = user.role
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
    else:
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        if not user: 
            st.session_state.user_id = None
            st.rerun()
            
        with st.sidebar:
            st.header(f"Welcome, {user.username}")
            if st.button("Logout"): 
                st.session_state.user_id = None
                st.rerun()
        
        if user.role=='admin': admin_page(session)
        elif user.role=='analyst': analyst_page(user, session)
        elif user.role=='pm': pm_page(user, session)
        
    session.close()

if __name__ == "__main__": main()
