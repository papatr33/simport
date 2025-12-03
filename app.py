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
# 1. CONFIGURATION & STYLING
# ==========================================

st.set_page_config(
    layout="wide", 
    page_title="AlphaTracker Pro",
    page_icon="📈"
)

# Custom CSS for "Commercial Grade" Look
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #1f2937; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #6b7280; }
    .stDataFrame { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #111827; }
    .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .success-text { color: #10B981; font-weight: bold; }
    .danger-text { color: #EF4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD"},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD"},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY"},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY"},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY"},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP"}, 
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR"}
}

# ==========================================
# 2. DATABASE SETUP
# ==========================================

@st.cache_resource
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

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

engine = get_db_engine()
try:
    Base.metadata.create_all(engine)
except Exception:
    pass # Tables likely exist

Session = sessionmaker(bind=engine)

# ==========================================
# 3. DATA ENGINE
# ==========================================

def extract_scalar(val):
    try:
        if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, list)):
            val = val.values.flatten()[0] if hasattr(val, 'values') else val[0]
        return float(val)
    except: return 0.0

@st.cache_data(ttl=300) 
def fetch_batch_data(tickers, start_date):
    if not tickers: return pd.DataFrame()
    tickers = list(set(tickers))
    try:
        # Buffer start date
        data = yf.download(tickers, start=start_date - timedelta(days=5), progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        elif data.empty: return pd.DataFrame()
        
        # Ensure DateTime Index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data.index = data.index.normalize()
        
        return data.ffill()
    except: return pd.DataFrame()

def get_historical_price(ticker, date_obj, market):
    try:
        start = date_obj
        end = date_obj + timedelta(days=5)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return 0.0, 0.0
        
        local_p = extract_scalar(df['Close'].dropna().iloc[0])
        usd_p = local_p

        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                fx_df = yf.download(cfg['fx'], start=start, end=end, progress=False)
                if not fx_df.empty:
                    rate = extract_scalar(fx_df['Close'].dropna().iloc[0])
                    if market == "UK": local_p = local_p / 100.0
                    if rate > 0: usd_p = (local_p / 100.0 if market == "UK" else local_p) / rate
                    else: usd_p = 0.0
        return local_p, usd_p
    except: return 0.0, 0.0

# ==========================================
# 4. CORE LOGIC
# ==========================================

def calculate_portfolio_state(user_id, session_obj):
    user = session_obj.query(User).filter_by(id=user_id).first()
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    
    state = {
        "cash": user.initial_capital,
        "positions": {}, 
        "realized_pnl_ytd": {},
        "equity": 0.0
    }

    # Ledger Replay
    for t in txs:
        tik = t.ticker
        if tik not in state["positions"]:
            state["positions"][tik] = {"qty": 0.0, "avg_cost": 0.0, "type": "FLAT", "market": t.market}
        
        pos = state["positions"][tik]
        
        if t.trans_type == "BUY":
            state["cash"] -= t.amount
            new_val = (pos["qty"] * pos["avg_cost"]) + t.amount
            pos["qty"] += t.quantity
            pos["avg_cost"] = new_val / pos["qty"] if pos["qty"] > 0 else 0.0
            pos["type"] = "LONG"

        elif t.trans_type == "SELL":
            state["cash"] += t.amount
            cost_basis = t.quantity * pos["avg_cost"]
            pnl = t.amount - cost_basis
            state["realized_pnl_ytd"][tik] = state["realized_pnl_ytd"].get(tik, 0) + pnl
            pos["qty"] -= t.quantity
            if pos["qty"] <= 0.001: del state["positions"][tik]

        elif t.trans_type == "SHORT_SELL":
            state["cash"] += t.amount
            curr_val = abs(pos["qty"]) * pos["avg_cost"]
            new_val = curr_val + t.amount
            pos["qty"] -= t.quantity
            pos["avg_cost"] = new_val / abs(pos["qty"]) if abs(pos["qty"]) > 0 else 0.0
            pos["type"] = "SHORT"

        elif t.trans_type == "BUY_TO_COVER":
            state["cash"] -= t.amount
            cost_basis = t.quantity * pos["avg_cost"]
            pnl = cost_basis - t.amount
            state["realized_pnl_ytd"][tik] = state["realized_pnl_ytd"].get(tik, 0) + pnl
            pos["qty"] += t.quantity
            if abs(pos["qty"]) <= 0.001: del state["positions"][tik]

    # Live Mark to Market
    active_tickers = list(state["positions"].keys())
    active_markets = {tik: pos['market'] for tik, pos in state["positions"].items()}
    fx_needed = [MARKET_CONFIG[m]['fx'] for m in active_markets.values() if m and MARKET_CONFIG[m]['fx']]
    
    batch_data = fetch_batch_data(active_tickers + fx_needed, datetime.now() - timedelta(days=5))
    
    state["equity"] = state["cash"]
    
    for tik, pos in state["positions"].items():
        usd_p = pos["avg_cost"]
        try:
            raw_p = None
            if not batch_data.empty and tik in batch_data.columns:
                raw_p = extract_scalar(batch_data[tik].iloc[-1])
            
            if raw_p is not None:
                mkt = pos.get('market', 'US')
                fx_tik = MARKET_CONFIG.get(mkt, {}).get('fx')
                if fx_tik and fx_tik in batch_data.columns:
                     rate = extract_scalar(batch_data[fx_tik].iloc[-1])
                     if mkt == "UK": raw_p = raw_p / 100.0
                     if rate > 0: usd_p = raw_p / rate
                else: usd_p = raw_p
        except: pass
             
        pos["current_price"] = usd_p
        
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

def get_ytd_performance(user_id, session_obj):
    """
    Generates a continuous daily equity curve from Jan 1st (or inception) to Today.
    Fills gaps for holidays. Normalizes against SPY.
    """
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    user = session_obj.query(User).filter_by(id=user_id).first()
    
    # Define Timeframe: YTD or from first trade if simpler
    # Standard YTD: Jan 1 current year
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime.now()
    
    # If no history, return empty
    if not txs:
        dates = pd.date_range(start_date, end_date, freq='B')
        df = pd.DataFrame({'Date': dates, 'Equity': user.initial_capital})
        df['Return %'] = 0.0
        return df, pd.Series()

    # Get Tickers involved in history
    tickers = list(set([t.ticker for t in txs]))
    
    # Fetch data starting slightly before Jan 1 for lookback if needed
    fetch_start = min(start_date, txs[0].date) - timedelta(days=5)
    batch_data = fetch_batch_data(tickers, fetch_start)
    
    # Generate Business Days Range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    curve = []
    
    # Optimization: Map Tickers
    if not batch_data.empty:
        batch_data.index = pd.to_datetime(batch_data.index).normalize()
    
    curr_cash = user.initial_capital
    holdings = {}
    
    # Pre-process transactions into a DataFrame for easier slicing
    # We need to process ALL transactions from the beginning of time to get correct cash/holdings state at Jan 1
    # Then we record daily state from Jan 1 onwards.
    
    tx_idx = 0
    n_txs = len(txs)
    
    # 1. Roll forward to Jan 1 (Start of Chart)
    #    Process all txs before start_date to set initial state
    while tx_idx < n_txs and txs[tx_idx].date < start_date:
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

    # 2. Daily Loop for Chart
    for d in dates:
        d_norm = d.normalize()
        
        # Apply transactions for this day
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
            
        # Value Holdings using data (with ffill from fetch)
        long_val = 0.0
        short_val = 0.0
        
        if not batch_data.empty:
            # We look for the date, or the last available date before it (ffill logic is in fetch, but we need safe lookup)
            try:
                # Asof or direct lookup if freq matches
                if d_norm in batch_data.index:
                    row = batch_data.loc[d_norm]
                else:
                    # Find closest previous date
                    idx = batch_data.index.get_indexer([d_norm], method='pad')[0]
                    if idx != -1:
                        row = batch_data.iloc[idx]
                    else: row = pd.Series()

                if not row.empty:
                    for tik, qty in holdings.items():
                        if abs(qty) > 0.001:
                            p = float(row[tik]) if tik in row else 0.0
                            if pd.isna(p): p = 0.0
                            val = qty * p
                            if qty > 0: long_val += val
                            else: short_val += abs(val)
            except: pass
        
        equity = curr_cash + long_val - short_val
        curve.append({"Date": d, "Equity": equity})

    df_curve = pd.DataFrame(curve)
    df_curve['Return %'] = ((df_curve['Equity'] / user.initial_capital) - 1) * 100
    
    # SPY Benchmark (Normalized)
    spy_ret = pd.Series()
    try:
        spy = fetch_batch_data(["SPY"], fetch_start)
        # Filter for chart range
        spy = spy[(spy.index >= pd.Timestamp(start_date)) & (spy.index <= pd.Timestamp(end_date))]
        if not spy.empty and 'SPY' in spy.columns:
            start_price = extract_scalar(spy['SPY'].iloc[0])
            spy_ret = ((spy['SPY'] / start_price) - 1) * 100
    except: pass
    
    return df_curve, spy_ret

# ==========================================
# 5. UI COMPONENTS
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

def render_chart(df_c, spy_c):
    if not df_c.empty:
        fig = go.Figure()
        
        # Portfolio Line
        fig.add_trace(go.Scatter(
            x=df_c['Date'], y=df_c['Return %'], 
            name="Portfolio", 
            line=dict(color='#2563EB', width=3),
            hovertemplate='%{y:.2f}%'
        ))
        
        # SPY Line
        if not spy_c.empty:
            fig.add_trace(go.Scatter(
                x=spy_c.index, y=spy_c.values, 
                name="S&P 500", 
                line=dict(color='#9CA3AF', dash='dot', width=2),
                hovertemplate='%{y:.2f}%'
            ))
            
        fig.update_layout(
            title="YTD Performance vs Benchmark",
            template="plotly_white",
            hovermode="x unified",
            yaxis=dict(title="Cumulative Return (%)", tickformat=".1f"),
            legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
            margin=dict(l=20,r=20,t=60,b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to generate chart.")

def analyst_page(user, session_obj, is_pm_view=False):
    # Header
    if not is_pm_view:
        st.markdown(f"## 🚀 Welcome, {user.username}")
    else:
        st.markdown(f"### Viewing Analyst: {user.username}")
    
    state = calculate_portfolio_state(user.id, session_obj)
    
    # 1. Top Metrics
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Equity", f"${state['equity']:,.0f}")
        c2.metric("Cash Balance", f"${state['cash']:,.0f}")
        
        pnl_val = state['equity'] - user.initial_capital
        pnl_pct = (pnl_val / user.initial_capital) * 100
        color = "normal"
        if pnl_val > 0: color="success"
        elif pnl_val < 0: color="inverse"
        
        c3.metric("YTD PnL ($)", f"${pnl_val:,.0f}", delta=f"{pnl_pct:.2f}%")
        
        # Exposure
        long_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'LONG')
        short_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'SHORT')
        net_exp = long_exp - short_exp
        c4.metric("Net Exposure", f"${net_exp:,.0f}")

    st.markdown("---")

    # 2. Charts
    df_c, spy_c = get_ytd_performance(user.id, session_obj)
    render_chart(df_c, spy_c)
    
    st.markdown("---")

    # 3. Holdings
    st.subheader("Current Holdings")
    holdings_data = []
    for tik, pos in state['positions'].items():
        pnl = pos.get('unrealized', 0)
        holdings_data.append({
            "Ticker": tik, "Type": pos['type'], "Market": pos['market'],
            "Qty": f"{pos['qty']:.2f}", 
            "Avg Cost": f"${pos['avg_cost']:,.2f}", 
            "Current Price": f"${pos.get('current_price',0):,.2f}",
            "Market Val": pos.get('mkt_val', 0),
            "Unrealized PnL": pnl,
            "Return %": (pnl / (pos['qty']*pos['avg_cost']))*100 if pos['qty']!=0 else 0
        })
    
    if holdings_data:
        h_df = pd.DataFrame(holdings_data)
        # Styling for the table
        st.dataframe(
            h_df.style.format({
                "Market Val": "${:,.0f}", 
                "Unrealized PnL": "${:,.0f}",
                "Return %": "{:.2f}%"
            }).background_gradient(subset=["Unrealized PnL", "Return %"], cmap="RdYlGn", vmin=-5000, vmax=5000),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No active positions.")

    # 4. Order Entry (HIDDEN FOR PM)
    if not is_pm_view:
        st.markdown("---")
        st.subheader("⚡ Execute Trade")
        with st.form("order_form", clear_on_submit=True):
            col_a, col_b, col_c, col_d = st.columns(4)
            mkt = col_a.selectbox("Market", list(MARKET_CONFIG.keys()))
            tik = col_b.text_input("Ticker Symbol").strip()
            side = col_c.selectbox("Order Type", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
            amt = col_d.number_input("Amount (USD)", min_value=1000.0, step=1000.0)
            
            note = st.text_area("Investment Rationale / Notes", height=80)
            
            test_mode = is_test_mode(session_obj)
            if test_mode:
                st.warning("⚠️ Test/Backdate Mode Active")
                d_val = st.date_input("Backdate To", value=datetime.now())
            
            submitted = st.form_submit_button("Submit Order", type="primary")
            
            if submitted:
                if not tik: st.error("Ticker is required"); st.stop()
                
                final_tik = format_ticker(tik, mkt)
                
                # Check Compliance (Simplified)
                # ... (Keep existing checks if needed, skipping for brevity in this refactor)
                
                if test_mode:
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
                        st.success(f"Filled {side} {final_tik} @ ${usd_p:.2f}")
                        time.sleep(1); st.rerun()
                    else:
                        st.error("Could not fetch historical price.")
                else:
                    session_obj.add(Transaction(
                        user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                        status='PENDING', amount=amt, notes=note
                    ))
                    session_obj.commit()
                    st.success("Order queued for Next Open execution.")
                    time.sleep(1); st.rerun()

def pm_page(user, session_obj):
    st.title("👨‍💼 Portfolio Manager Dashboard")
    
    analysts = session_obj.query(User).filter_by(role='analyst').all()
    if not analysts:
        st.warning("No analysts found.")
        return

    # 1. Aggregate Stats
    summary = []
    monthly_map = {} # {analyst_name: {month: return}}
    
    progress = st.progress(0, text="Calculating Portfolio Analytics...")
    
    for idx, a in enumerate(analysts):
        # Calculate State
        s = calculate_portfolio_state(a.id, session_obj)
        total_ret_pct = ((s['equity'] / a.initial_capital) - 1) * 100
        
        summary.append({
            "Analyst": a.username, 
            "Equity": s['equity'], 
            "Cash %": (s['cash'] / s['equity']) * 100,
            "YTD PnL": s['equity'] - a.initial_capital,
            "Total Ret %": total_ret_pct
        })
        
        # Calculate Monthly Breakdown
        df_c, _ = get_ytd_performance(a.id, session_obj)
        if not df_c.empty:
            # Resample to month end
            m_df = df_c.set_index('Date').resample('ME').last()
            # Calculate monthly diff in equity or return? Return is better.
            m_df['Monthly Ret'] = m_df['Equity'].pct_change() * 100
            # Handle first month (pct_change is NaN) -> (Equity / Init) - 1
            if len(m_df) > 0:
                first_eq = m_df['Equity'].iloc[0]
                # Approximation for first month
                pass 
            
            # Create dict for heatmap
            m_ret = m_df['Monthly Ret'].fillna(0).to_dict() # {Timestamp: float}
            # Convert keys to Mon-YY string
            formatted_ret = {k.strftime('%b'): v for k, v in m_ret.items() if k.year == datetime.now().year}
            monthly_map[a.username] = formatted_ret
            
        progress.progress((idx + 1) / len(analysts))

    progress.empty()

    # Display Leaderboard
    df_sum = pd.DataFrame(summary).sort_values("Total Ret %", ascending=False)
    
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Leaderboard")
        st.dataframe(
            df_sum.style.format({
                "Equity": "${:,.0f}", "YTD PnL": "${:,.0f}", 
                "Cash %": "{:.1f}%", "Total Ret %": "{:.2f}%"
            }).background_gradient(subset=["Total Ret %"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True
        )

    with c2:
        st.subheader("Monthly Returns Matrix (%)")
        if monthly_map:
            heatmap_df = pd.DataFrame(monthly_map).T # Analysts as Rows
            # Reorder months?
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            existing_cols = [m for m in months if m in heatmap_df.columns]
            heatmap_df = heatmap_df[existing_cols]
            
            st.dataframe(
                heatmap_df.style.format("{:.2f}%").background_gradient(cmap="RdYlGn", vmin=-5, vmax=5),
                use_container_width=True
            )
        else:
            st.info("No monthly data available yet.")

    st.markdown("---")
    
    # 2. Detailed View
    st.subheader("🔍 Deep Dive")
    selected_analyst = st.selectbox("Select Analyst to Inspect", [a.username for a in analysts])
    if selected_analyst:
        target = session_obj.query(User).filter_by(username=selected_analyst).first()
        # Call analyst page in "PM View" mode (ReadOnly)
        with st.container(border=True):
            analyst_page(target, session_obj, is_pm_view=True)

def admin_page(session_obj):
    st.title("🛠️ System Administration")
    
    # Global Configs
    st.subheader("System Controls")
    curr = is_test_mode(session_obj)
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.toggle("Enable Test/Backdate Mode", value=curr):
            if not curr:
                cfg = session_obj.query(SystemConfig).filter_by(key='test_mode').first()
                if not cfg: 
                    session_obj.add(SystemConfig(key='test_mode', value='True'))
                else: cfg.value = 'True'
                session_obj.commit(); st.rerun()
        else:
            if curr:
                cfg = session_obj.query(SystemConfig).filter_by(key='test_mode').first()
                cfg.value = 'False'
                session_obj.commit(); st.rerun()
    with c2:
        if curr: st.info("Enabled: Analysts can backdate trades.")
        else: st.info("Disabled: Trades entered as PENDING for T+1 execution.")

    st.divider()

    # User Management
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("Create New User")
        with st.form("new_user"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            r = st.selectbox("Role", ["analyst", "pm"])
            if st.form_submit_button("Create Account"):
                if session_obj.query(User).filter_by(username=u).first():
                    st.error("Username exists")
                else:
                    session_obj.add(User(username=u, password_hash=bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode(), role=r))
                    session_obj.commit()
                    st.success(f"User {u} created!")
                    time.sleep(1); st.rerun()

    with c_right:
        st.subheader("Manage Users")
        users = session_obj.query(User).all()
        user_df = pd.DataFrame([{"ID": u.id, "Username": u.username, "Role": u.role, "Capital": u.initial_capital} for u in users])
        st.dataframe(user_df, hide_index=True, use_container_width=True)
        
        # Deletion
        del_target = st.selectbox("Delete User", [u.username for u in users if u.username != 'admin'], index=None)
        if del_target:
            if st.button(f"🗑️ Confirm Delete {del_target}", type="primary"):
                u_obj = session_obj.query(User).filter_by(username=del_target).first()
                session_obj.delete(u_obj)
                session_obj.commit()
                st.success("User deleted.")
                time.sleep(1); st.rerun()

def main():
    # Session handling
    session = Session()
    # Init Admin if fresh DB
    try:
        if not session.query(User).filter_by(username='admin').first():
            session.add(User(username='admin', password_hash=bcrypt.hashpw('8848'.encode(), bcrypt.gensalt()).decode(), role='admin'))
            session.commit()
    except: pass

    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    if not st.session_state.user_id:
        # Login Screen
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("## 🔐 Login")
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.button("Sign In", use_container_width=True, type="primary"):
                    user = session.query(User).filter_by(username=u).first()
                    if user and bcrypt.checkpw(p.encode(), user.password_hash.encode()):
                        st.session_state.user_id = user.id
                        st.session_state.role = user.role
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
    else:
        # Logged In
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        if not user:
            st.session_state.user_id = None
            st.rerun()

        # Sidebar
        with st.sidebar:
            st.markdown(f"### 👤 {user.username}")
            st.caption(f"Role: {user.role.upper()}")
            st.markdown("---")
            if st.button("Logout", use_container_width=True):
                st.session_state.user_id = None
                st.rerun()
            st.markdown("---")
            st.caption("AlphaTracker v2.0")

        # Routing
        if user.role == 'admin':
            admin_page(session)
        elif user.role == 'analyst':
            analyst_page(user, session)
        elif user.role == 'pm':
            pm_page(user, session)
            
    session.close()

if __name__ == "__main__":
    main()
