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
    pass 

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
        data = yf.download(tickers, start=start_date - timedelta(days=7), progress=False)['Close']
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
        "realized_pnl_by_side": {}, # Key: (ticker, 'LONG'/'SHORT') -> amount
        "equity": 0.0
    }

    # Ledger Replay
    for t in txs:
        tik = t.ticker
        if tik not in state["positions"]:
            state["positions"][tik] = {
                "qty": 0.0, 
                "avg_cost": 0.0, 
                "avg_cost_local": 0.0, # Added for local currency tracking
                "type": "FLAT", "market": t.market, 
                "first_entry": None
            }
        
        pos = state["positions"][tik]
        
        if t.trans_type == "BUY":
            state["cash"] -= t.amount
            new_val = (pos["qty"] * pos["avg_cost"]) + t.amount
            
            # Local Cost Calculation
            t_local_price = t.local_price if t.local_price else t.price
            new_val_local = (pos["qty"] * pos["avg_cost_local"]) + (t.quantity * t_local_price)
            
            pos["qty"] += t.quantity
            pos["avg_cost"] = new_val / pos["qty"] if pos["qty"] > 0 else 0.0
            pos["avg_cost_local"] = new_val_local / pos["qty"] if pos["qty"] > 0 else 0.0
            
            pos["type"] = "LONG"
            if not pos["first_entry"]: pos["first_entry"] = t.date

        elif t.trans_type == "SELL":
            state["cash"] += t.amount
            cost_basis = t.quantity * pos["avg_cost"]
            pnl = t.amount - cost_basis
            
            key = (tik, 'LONG')
            state["realized_pnl_by_side"][key] = state["realized_pnl_by_side"].get(key, 0) + pnl
            
            pos["qty"] -= t.quantity
            if pos["qty"] <= 0.001: 
                del state["positions"][tik]

        elif t.trans_type == "SHORT_SELL":
            state["cash"] += t.amount
            curr_val = abs(pos["qty"]) * pos["avg_cost"]
            new_val = curr_val + t.amount

            # Local Cost Calculation
            t_local_price = t.local_price if t.local_price else t.price
            curr_val_local = abs(pos["qty"]) * pos["avg_cost_local"]
            new_val_local = curr_val_local + (t.quantity * t_local_price)

            pos["qty"] -= t.quantity
            pos["avg_cost"] = new_val / abs(pos["qty"]) if abs(pos["qty"]) > 0 else 0.0
            pos["avg_cost_local"] = new_val_local / abs(pos["qty"]) if abs(pos["qty"]) > 0 else 0.0
            
            pos["type"] = "SHORT"
            if not pos["first_entry"]: pos["first_entry"] = t.date

        elif t.trans_type == "BUY_TO_COVER":
            state["cash"] -= t.amount
            cost_basis = t.quantity * pos["avg_cost"]
            pnl = cost_basis - t.amount
            
            key = (tik, 'SHORT')
            state["realized_pnl_by_side"][key] = state["realized_pnl_by_side"].get(key, 0) + pnl
            
            pos["qty"] += t.quantity
            if abs(pos["qty"]) <= 0.001: 
                del state["positions"][tik]

    # Live Mark to Market
    active_tickers = list(state["positions"].keys())
    active_markets = {tik: pos['market'] for tik, pos in state["positions"].items()}
    fx_needed = [MARKET_CONFIG[m]['fx'] for m in active_markets.values() if m and MARKET_CONFIG[m]['fx']]
    
    batch_data = fetch_batch_data(active_tickers + fx_needed, datetime.now() - timedelta(days=5))
    
    state["equity"] = state["cash"]
    
    for tik, pos in state["positions"].items():
        usd_p = pos["avg_cost"]
        pos["current_local_price"] = 0.0
        
        try:
            raw_p = None
            if not batch_data.empty and tik in batch_data.columns:
                raw_p = extract_scalar(batch_data[tik].iloc[-1])
            
            if raw_p is not None:
                pos["current_local_price"] = raw_p
                mkt = pos.get('market', 'US')
                fx_tik = MARKET_CONFIG.get(mkt, {}).get('fx')
                
                if fx_tik and fx_tik in batch_data.columns:
                     rate = extract_scalar(batch_data[fx_tik].iloc[-1])
                     if mkt == "UK": raw_p = raw_p / 100.0
                     if rate > 0: usd_p = raw_p / rate
                else: 
                     usd_p = raw_p
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
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    user = session_obj.query(User).filter_by(id=user_id).first()
    
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime.now()
    
    if not txs:
        # Return empty structures if no trades
        dates = pd.date_range(start_date, end_date, freq='B')
        df = pd.DataFrame({'Date': dates, 'Equity': user.initial_capital})
        df['Return %'] = 0.0
        return df, pd.Series(), pd.DataFrame()

    ticker_market_map = {}
    for t in txs:
        ticker_market_map[t.ticker] = t.market

    tickers = list(ticker_market_map.keys())
    
    fx_tickers = set()
    for m in ticker_market_map.values():
        if m and MARKET_CONFIG.get(m, {}).get('fx'):
            fx_tickers.add(MARKET_CONFIG[m]['fx'])
    
    all_tickers = tickers + list(fx_tickers)
    fetch_start = min(start_date, txs[0].date) - timedelta(days=5)
    batch_data = fetch_batch_data(all_tickers, fetch_start)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    curve = []
    
    if not batch_data.empty:
        batch_data.index = pd.to_datetime(batch_data.index).normalize()
    
    curr_cash = user.initial_capital
    holdings = {} # {ticker: quantity}
    
    # Track daily attribution
    long_pnl_tracker = 0.0
    short_pnl_tracker = 0.0

    tx_idx = 0
    n_txs = len(txs)
    
    # --- 1. PRE-ROLL (Before Start Date) ---
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

    # --- 2. DAILY LOOP ---
    for d in dates:
        d_norm = d.normalize()
        
        # Process transactions for this day
        # We also need to track realized pnl for attribution here if we want perfect daily pnl breakdown
        # For simplicity, we calculate daily change in Market Value + Flows to derive PnL
        
        daily_long_flow = 0.0
        daily_short_flow = 0.0
        
        while tx_idx < n_txs and txs[tx_idx].date.date() <= d_norm.date():
            t = txs[tx_idx]
            if t.trans_type == 'BUY':
                curr_cash -= t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) + t.quantity
                daily_long_flow += t.amount # Money went INTO longs
            elif t.trans_type == 'SELL':
                curr_cash += t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) - t.quantity
                daily_long_flow -= t.amount # Money came OUT of longs
            elif t.trans_type == 'SHORT_SELL':
                curr_cash += t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) - t.quantity
                daily_short_flow -= t.amount # Money came IN (liability increase, but cash inflow)
            elif t.trans_type == 'BUY_TO_COVER':
                curr_cash -= t.amount
                holdings[t.ticker] = holdings.get(t.ticker, 0) + t.quantity
                daily_short_flow += t.amount # Money went OUT (liability decrease)
            tx_idx += 1
            
        long_mv = 0.0
        short_mv = 0.0 # Absolute value of short liability
        
        if not batch_data.empty:
            try:
                row = pd.Series()
                if d_norm in batch_data.index:
                    row = batch_data.loc[d_norm]
                else:
                    idx = batch_data.index.get_indexer([d_norm], method='pad')[0]
                    if idx != -1: row = batch_data.iloc[idx]

                if not row.empty:
                    for tik, qty in holdings.items():
                        if abs(qty) > 0.001:
                            p_local = float(row[tik]) if tik in row else 0.0
                            if pd.isna(p_local): p_local = 0.0
                            
                            mkt = ticker_market_map.get(tik, 'US')
                            fx_sym = MARKET_CONFIG.get(mkt, {}).get('fx')
                            p_usd = p_local
                            
                            if fx_sym:
                                if fx_sym in row:
                                    rate = float(row[fx_sym])
                                    if mkt == "UK": p_local /= 100.0
                                    if rate > 0: p_usd = p_local / rate
                                    else: p_usd = 0.0
                                else:
                                    p_usd = p_local 

                            val = qty * p_usd
                            if qty > 0: long_mv += val
                            else: short_mv += abs(val) # Liability is positive magnitude
            except: pass
        
        # Equity = Cash + Longs - ShortLiability
        equity = curr_cash + long_mv - short_mv
        
        curve.append({
            "Date": d, 
            "Equity": equity,
            "LongMV": long_mv,
            "ShortMV": short_mv,
            "LongFlow": daily_long_flow,
            "ShortFlow": daily_short_flow
        })

    df_curve = pd.DataFrame(curve)
    
    # Calculate Daily PnL Attribution
    # PnL_Long = (LongMV_t - LongMV_t-1) - NetFlow_Long
    # PnL_Short = (ShortMV_t-1 - ShortMV_t) - NetFlow_Short (Short logic is inverted: Liab decreases is profit)
    # Note: ShortFlow is: ShortSell (+Cash), Cover (-Cash). 
    # Logic: NewShortMV = OldShortMV + SellAmt - CoverAmt - PnL_Short
    # => PnL_Short = OldShortMV - NewShortMV + SellAmt - CoverAmt
    # => PnL_Short = OldShortMV - NewShortMV - DailyShortFlow (where flow is -Sell + Cover)
    # Wait, in loop: ShortSell -> flow -= amount (negative), Cover -> flow += amount (positive)
    # So Flow = Cover - Sell
    # PnL_Short = (ShortMV_t-1 - ShortMV_t) - (Cover - Sell) 
    #           = ShortMV_t-1 - ShortMV_t - Flow
    
    df_curve['Long PnL'] = (df_curve['LongMV'].diff() - df_curve['LongFlow']).fillna(0)
    df_curve['Short PnL'] = (-(df_curve['ShortMV'].diff()) - df_curve['ShortFlow']).fillna(0)
    
    # Correction for first day attribution (assuming 0 previous) if needed, 
    # but strictly we care about returns in the window. 
    
    df_curve['Return %'] = ((df_curve['Equity'] / user.initial_capital) - 1) * 100
    
    spy_ret = pd.Series()
    try:
        spy = fetch_batch_data(["SPY"], fetch_start)
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
        
        fig.add_trace(go.Scatter(
            x=df_c['Date'], y=df_c['Return %'], 
            name="Portfolio", 
            line=dict(color='#2563EB', width=3),
            hovertemplate='%{y:.2f}%'
        ))
        
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
            yaxis=dict(title="Cumulative Return (%)", tickformat="+.1f"),
            legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
            margin=dict(l=20,r=20,t=60,b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to generate chart.")

def render_monthly_breakdown(df_curve, initial_capital):
    if df_curve.empty: return pd.DataFrame()
    
    # Resample to Monthly
    df_curve['Month'] = df_curve['Date'].dt.to_period('M')
    
    monthly_stats = []
    
    grouped = df_curve.groupby('Month')
    
    prev_equity = initial_capital
    
    for month, group in grouped:
        month_end_equity = group['Equity'].iloc[-1]
        
        # Monthly Return = (End Equity - Start Equity) / Start Equity
        # NOTE: Using previous month end equity as base
        if prev_equity == 0: prev_equity = initial_capital # safety
        
        total_ret = (month_end_equity - prev_equity) / prev_equity
        
        # Contribution Calculation: Sum of Daily PnL / Start Equity
        long_pnl_sum = group['Long PnL'].sum()
        short_pnl_sum = group['Short PnL'].sum()
        
        long_contrib = long_pnl_sum / prev_equity
        short_contrib = short_pnl_sum / prev_equity
        
        monthly_stats.append({
            "Month": month.strftime('%Y-%b'),
            "Long Ret %": long_contrib * 100,
            "Short Ret %": short_contrib * 100,
            "Total Ret %": total_ret * 100
        })
        
        prev_equity = month_end_equity
        
    return pd.DataFrame(monthly_stats)

def analyst_page(user, session_obj, is_pm_view=False):
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
        
        c3.metric("YTD PnL ($)", f"{pnl_val:+,.0f}", delta=f"{pnl_pct:+.2f}%")
        
        long_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'LONG')
        short_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'SHORT')
        net_exp = long_exp - short_exp
        c4.metric("Net Exposure", f"${net_exp:,.0f}")

    st.markdown("---")

    t1, t2, t3, t4, t5 = st.tabs(["Performance Chart", "Monthly Returns", "Current Holdings", "All Historical Positions", "Transaction Log"])
    
    df_c, spy_c = get_ytd_performance(user.id, session_obj)

    with t1:
        render_chart(df_c, spy_c)

    with t2:
        st.subheader("Monthly Return Attribution")
        m_df = render_monthly_breakdown(df_c, user.initial_capital)
        if not m_df.empty:
            st.dataframe(
                m_df.style.format({
                    "Long Ret %": "{:+.2f}%",
                    "Short Ret %": "{:+.2f}%",
                    "Total Ret %": "{:+.2f}%"
                }).background_gradient(subset=["Total Ret %"], cmap="RdYlGn", vmin=-5, vmax=5),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No data available.")

    with t3:
        st.subheader("Current Holdings")
        holdings_data = []
        for tik, pos in state['positions'].items():
            unrealized_pnl = pos.get('unrealized', 0)
            side_key = (tik, pos['type'])
            realized_pnl = state['realized_pnl_by_side'].get(side_key, 0.0)

            loc_p = pos.get('current_local_price', 0)
            avg_cost_loc = pos.get('avg_cost_local', 0)
            
            cur = MARKET_CONFIG.get(pos['market'], {}).get('currency', 'USD')
            
            if cur == 'USD':
                price_str = f"${loc_p:,.2f}"
                cost_str = f"${avg_cost_loc:,.2f}"
            else:
                price_str = f"{loc_p:,.2f} {cur}"
                cost_str = f"{avg_cost_loc:,.2f} {cur}"

            invested_capital = pos['qty'] * pos['avg_cost']
            ret_pct = (unrealized_pnl / abs(invested_capital)) * 100 if invested_capital != 0 else 0.0

            # Portfolio Percentage
            port_pct = (pos.get('mkt_val', 0) / state['equity']) * 100

            holdings_data.append({
                "Ticker": tik, "Type": pos['type'], "Market": pos['market'],
                "Qty": f"{pos['qty']:,.0f}", 
                "Avg Cost (USD)": f"${pos['avg_cost']:,.2f}", 
                "Avg Cost (Local)": cost_str,
                "Current Price": price_str,
                "Market Val (USD)": pos.get('mkt_val', 0),
                "Port %": port_pct,
                "Unrealized PnL": unrealized_pnl,
                "Realized PnL": realized_pnl, 
                "Return %": ret_pct,
                "Entry Date": pos.get('first_entry').strftime('%Y-%m-%d') if pos.get('first_entry') else '-'
            })
        
        if holdings_data:
            h_df = pd.DataFrame(holdings_data)
            st.dataframe(
                h_df.style.format({
                    "Market Val (USD)": "${:,.0f}", 
                    "Unrealized PnL": "{:+,.0f}",
                    "Realized PnL": "{:+,.0f}",
                    "Return %": "{:+.2f}%",
                    "Port %": "{:.1f}%"
                }).background_gradient(subset=["Return %"], cmap="RdYlGn", vmin=-20, vmax=20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active positions.")

    with t4:
        st.subheader("All Traded Positions (Historical)")
        all_keys = set(state['realized_pnl_by_side'].keys())
        for tik, pos in state['positions'].items():
            all_keys.add((tik, pos['type']))
        
        history_data = []
        for (tik, side) in all_keys:
            realized = state['realized_pnl_by_side'].get((tik, side), 0.0)
            unrealized = 0.0
            if tik in state['positions'] and state['positions'][tik]['type'] == side:
                unrealized = state['positions'][tik].get('unrealized', 0.0)
            
            total_pnl = realized + unrealized
            
            history_data.append({
                "Ticker": tik,
                "Side": side,
                "Total Realized PnL": realized,
                "Total Unrealized PnL": unrealized,
                "Total Net PnL": total_pnl
            })
            
        if history_data:
            hist_df = pd.DataFrame(history_data).sort_values("Total Net PnL", ascending=False)
            st.dataframe(
                hist_df.style.format({
                    "Total Realized PnL": "{:+,.0f}",
                    "Total Unrealized PnL": "{:+,.0f}",
                    "Total Net PnL": "{:+,.0f}"
                }).background_gradient(subset=["Total Net PnL"], cmap="RdYlGn", vmin=-5000, vmax=5000),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No trading history.")

    with t5:
        st.subheader("Transaction History")
        hist_txs = session_obj.query(Transaction).filter_by(user_id=user.id).order_by(Transaction.date.desc()).all()
        if hist_txs:
            hist_data = []
            for t in hist_txs:
                cur = MARKET_CONFIG.get(t.market, {}).get('currency', 'USD')
                p_display = t.local_price if t.local_price else t.price
                if not p_display: p_display = 0.0
                
                if cur == 'USD':
                    p_str = f"${p_display:,.2f}"
                else:
                    p_str = f"{p_display:,.2f} {cur}"

                hist_data.append({
                    "Date": t.date.strftime('%Y-%m-%d'),
                    "Ticker": t.ticker,
                    "Type": t.trans_type,
                    "Amount (USD)": t.amount,
                    "Fill Price (Local)": p_str,
                    "Quantity": t.quantity if t.quantity else 0,
                    "Status": t.status,
                    "Notes": t.notes
                })
            
            h_df = pd.DataFrame(hist_data)
            st.dataframe(
                h_df.style.format({"Amount (USD)": "${:,.0f}", "Quantity": "{:,.0f}"}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No transactions found.")

    if not is_pm_view:
        st.markdown("---")
        st.subheader("⚡ Execute Trade")
        
        with st.expander("Compliance Rules Summary"):
            st.markdown("""
            * **Longs:** Position: 10% - 40% of Equity. Total Longs: Must be > 90% of Equity. Max 5 Names.
            * **Shorts:** Position: 10% - 30% of Equity. Total Shorts: 30% - 50% of Equity. Max 3 Names.
            * **Lockup:** 30 Days. *Exception: Profit > 15% or Loss > 20% on Shorts.*
            """)

        with st.form("order_form", clear_on_submit=True):
            col_a, col_b, col_c, col_d = st.columns(4)
            mkt = col_a.selectbox("Market", list(MARKET_CONFIG.keys()))
            tik = col_b.text_input("Ticker Symbol").strip()
            side = col_c.selectbox("Order Type", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
            
            # Using fraction of equity might be easier, but sticking to shares for input
            qty_input = col_d.number_input("Quantity (Shares)", min_value=1.0, step=100.0)
            
            note = st.text_area("Investment Rationale / Notes", height=80)
            
            test_mode = is_test_mode(session_obj)
            if test_mode:
                st.warning("⚠️ Test/Backdate Mode Active")
                d_val = st.date_input("Backdate To", value=datetime.now())
            
            submitted = st.form_submit_button("Submit Order", type="primary")
            
            if submitted:
                if not tik: st.error("Ticker is required"); st.stop()
                final_tik = format_ticker(tik, mkt)
                
                if test_mode:
                    eval_date = datetime.combine(d_val, datetime.min.time())
                else:
                    eval_date = datetime.now()
                
                est_local_p, est_usd_p = get_historical_price(final_tik, eval_date, mkt)
                
                if est_usd_p <= 0:
                     st.error(f"Could not fetch price for {final_tik}. Cannot validate compliance.")
                     st.stop()
                
                est_amount = qty_input * est_usd_p
                total_equity = state['equity']
                
                # --- COMPLIANCE ENGINE ---
                error_msg = None
                warning_msg = None
                
                current_longs = [p for p in state['positions'].values() if p['type'] == 'LONG']
                current_shorts = [p for p in state['positions'].values() if p['type'] == 'SHORT']
                current_pos = state['positions'].get(final_tik)
                
                # Pre-calculate totals
                total_long_val = sum(p['mkt_val'] for p in current_longs)
                total_short_val = sum(p['mkt_val'] for p in current_shorts)
                
                # 1. LONG RULES
                if side == 'BUY':
                    # Rule: Max 5 positions
                    if not current_pos and len(current_longs) >= 5:
                        error_msg = "Compliance Violation: Max 5 Long positions allowed."
                    
                    # Rule: Size 10% - 40%
                    curr_val = current_pos['mkt_val'] if current_pos and current_pos['type']=='LONG' else 0
                    proj_val = curr_val + est_amount
                    pct_size = (proj_val / total_equity) * 100
                    
                    if not (10.0 <= pct_size <= 40.0):
                        # Relax lower bound for initial entry if building position? 
                        # User said "each long position size should be between", usually implies target.
                        # We will block if > 40%, warn if < 10%
                        if pct_size > 40.0:
                             error_msg = f"Compliance Violation: Position size {pct_size:.1f}% exceeds limit (40%)."
                        elif pct_size < 10.0:
                             warning_msg = f"⚠️ Warning: Position size {pct_size:.1f}% is below target range (10%)."

                    # Rule: Total Longs > 90%
                    proj_total_long = total_long_val + est_amount
                    if (proj_total_long / total_equity) < 0.90:
                         if warning_msg: warning_msg += " Also, Total Long Exposure is < 90%."
                         else: warning_msg = "⚠️ Warning: Total Long Exposure is below 90% target."

                # 2. SHORT RULES
                if side == 'SHORT_SELL':
                    # Rule: Max 3 positions
                    if not current_pos and len(current_shorts) >= 3:
                        error_msg = "Compliance Violation: Max 3 Short positions allowed."
                    
                    # Rule: Size 10% - 30%
                    curr_val = current_pos['mkt_val'] if current_pos and current_pos['type']=='SHORT' else 0
                    proj_val = curr_val + est_amount
                    pct_size = (proj_val / total_equity) * 100
                    
                    if not (10.0 <= pct_size <= 30.0):
                        if pct_size > 30.0:
                             error_msg = f"Compliance Violation: Short Position {pct_size:.1f}% exceeds limit (30%)."
                        elif pct_size < 10.0:
                             warning_msg = f"⚠️ Warning: Short Position {pct_size:.1f}% is below target range (10%)."
                    
                    # Rule: Total Short 30% - 50%
                    proj_total_short = total_short_val + est_amount
                    proj_total_pct = (proj_total_short / total_equity) * 100
                    
                    if proj_total_pct > 50.0:
                        error_msg = f"Compliance Violation: Total Short Exposure {proj_total_pct:.1f}% exceeds limit (50%)."
                    elif proj_total_pct < 30.0:
                         if warning_msg: warning_msg += " Total Short < 30%."
                         else: warning_msg = "⚠️ Warning: Total Short Exposure is below 30% target."

                # 3. LOCKUP & EXCEPTION (Short Squeeze / Stop Loss)
                if side in ['SELL', 'BUY_TO_COVER']:
                    curr_date = datetime.combine(d_val, datetime.min.time()) if test_mode else datetime.now()
                    
                    if not current_pos:
                         error_msg = "Cannot close a position you don't hold."
                    else:
                        days_held = (curr_date - current_pos['first_entry']).days
                        
                        # Default strict rule
                        is_violation = days_held < 30
                        
                        # Check Exception for Shorts
                        if is_violation and side == 'BUY_TO_COVER':
                            # Calculate PnL %
                            # Short PnL % = (Entry Price - Current Price) / Entry Price
                            # Note: This is simplified. Using averages.
                            entry_p = current_pos['avg_cost']
                            curr_p = est_usd_p
                            pnl_pct = (entry_p - curr_p) / entry_p
                            
                            # Exception: > 15% Profit OR > 20% Loss (pnl < -0.20)
                            if pnl_pct > 0.15 or pnl_pct < -0.20:
                                is_violation = False
                                if warning_msg: warning_msg += f" (Lockup bypassed: PnL {pnl_pct*100:.1f}%)"
                                else: warning_msg = f"⚠️ Lockup bypassed due to PnL trigger ({pnl_pct*100:.1f}%)."
                        
                        if is_violation:
                             error_msg = f"Compliance Violation: Position held for {days_held} days. Min holding 30 days."

                if error_msg:
                    st.error(error_msg)
                else:
                    if warning_msg: st.warning(warning_msg)
                    
                    if test_mode:
                        h_date = datetime.combine(d_val, datetime.min.time())
                        session_obj.add(Transaction(
                            user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                            status='FILLED', date=h_date, 
                            amount=est_amount, quantity=qty_input,
                            local_price=est_local_p, price=est_usd_p, 
                            notes=f"[BACKDATE] {note}"
                        ))
                        session_obj.commit()
                        st.success(f"Filled {qty_input:,.0f} shares of {final_tik} @ {est_usd_p:.2f} USD")
                        time.sleep(1); st.rerun()
                    else:
                        session_obj.add(Transaction(
                            user_id=user.id, ticker=final_tik, market=mkt, trans_type=side,
                            status='PENDING', 
                            amount=est_amount, 
                            quantity=qty_input, 
                            notes=f"{note} (Est Price: ${est_usd_p:.2f})"
                        ))
                        session_obj.commit()
                        st.success(f"Order for {qty_input:,.0f} shares queued.")
                        time.sleep(1); st.rerun()

def pm_page(user, session_obj):
    st.title("👨‍💼 Portfolio Manager Dashboard")
    
    analysts = session_obj.query(User).filter_by(role='analyst').all()
    if not analysts:
        st.warning("No analysts found.")
        return

    summary = []
    
    # Placeholders for breakdown tables
    monthly_data_frames = {} 
    
    progress = st.progress(0, text="Calculating Portfolio Analytics...")
    
    for idx, a in enumerate(analysts):
        s = calculate_portfolio_state(a.id, session_obj)
        total_ret_pct = ((s['equity'] / a.initial_capital) - 1) * 100
        
        summary.append({
            "Analyst": a.username, 
            "Equity": s['equity'], 
            "Cash %": (s['cash'] / s['equity']) * 100,
            "YTD PnL": s['equity'] - a.initial_capital,
            "Total Ret %": total_ret_pct
        })
        
        df_c, _ = get_ytd_performance(a.id, session_obj)
        m_df = render_monthly_breakdown(df_c, a.initial_capital)
        if not m_df.empty:
            monthly_data_frames[a.username] = m_df
            
        progress.progress((idx + 1) / len(analysts))

    progress.empty()

    df_sum = pd.DataFrame(summary).sort_values("Total Ret %", ascending=False)
    
    st.subheader("Leaderboard")
    st.dataframe(
        df_sum.style.format({
            "Equity": "${:,.0f}", "YTD PnL": "{:+,.0f}", 
            "Cash %": "{:.1f}%", "Total Ret %": "{:+.2f}%"
        }).background_gradient(subset=["Total Ret %"], cmap="RdYlGn", vmin=-10, vmax=10),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.subheader("Monthly Returns Breakdown")
    
    if monthly_data_frames:
        tabs = st.tabs(list(monthly_data_frames.keys()))
        for i, (name, df) in enumerate(monthly_data_frames.items()):
            with tabs[i]:
                st.dataframe(
                    df.style.format({
                        "Long Ret %": "{:+.2f}%",
                        "Short Ret %": "{:+.2f}%",
                        "Total Ret %": "{:+.2f}%"
                    }).background_gradient(subset=["Total Ret %"], cmap="RdYlGn", vmin=-5, vmax=5),
                    use_container_width=True, hide_index=True
                )
    else:
        st.info("No monthly data available yet.")

    st.markdown("---")
    
    st.subheader("🔍 Deep Dive")
    selected_analyst = st.selectbox("Select Analyst to Inspect", [a.username for a in analysts])
    if selected_analyst:
        target = session_obj.query(User).filter_by(username=selected_analyst).first()
        with st.container(border=True):
            analyst_page(target, session_obj, is_pm_view=True)

def admin_page(session_obj):
    st.title("🛠️ System Administration")
    
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
        
        del_target = st.selectbox("Delete User", [u.username for u in users if u.username != 'admin'], index=None)
        if del_target:
            if st.button(f"🗑️ Confirm Delete {del_target}", type="primary"):
                u_obj = session_obj.query(User).filter_by(username=del_target).first()
                session_obj.delete(u_obj)
                session_obj.commit()
                st.success("User deleted.")
                time.sleep(1); st.rerun()

def main():
    session = Session()
    try:
        if not session.query(User).filter_by(username='admin').first():
            session.add(User(username='admin', password_hash=bcrypt.hashpw('8848'.encode(), bcrypt.gensalt()).decode(), role='admin'))
            session.commit()
    except: pass

    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    if not st.session_state.user_id:
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
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        if not user:
            st.session_state.user_id = None
            st.rerun()

        with st.sidebar:
            st.markdown(f"### 👤 {user.username}")
            st.caption(f"Role: {user.role.upper()}")
            st.markdown("---")
            if st.button("Logout", use_container_width=True):
                st.session_state.user_id = None
                st.rerun()
            st.markdown("---")
            st.caption("AlphaTracker v2.0")

        if user.role == 'admin':
            admin_page(session)
        elif user.role == 'analyst':
            analyst_page(user, session)
        elif user.role == 'pm':
            pm_page(user, session)
            
    session.close()

if __name__ == "__main__":
    main()
