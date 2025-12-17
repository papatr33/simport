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

from core_logic import calculate_portfolio_state, get_ytd_performance, fetch_batch_data, extract_scalar, MARKET_CONFIG

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
# 3. DATA ENGINE (OPTIMIZED)
# ==========================================



@st.cache_data(ttl=600) 
# Wrappers to maintain Streamlit Caching
def fetch_batch_data_cached(tickers, start_date):
    return fetch_batch_data(tickers, start_date)

@st.cache_data(ttl=60, show_spinner=False)
def calculate_portfolio_state_cached(txs_data, initial_capital):
    return calculate_portfolio_state(txs_data, initial_capital)

@st.cache_data(ttl=300, show_spinner=False)
def get_ytd_performance_cached(txs_data, initial_capital):
    return get_ytd_performance(txs_data, initial_capital)



def get_historical_price(ticker, date_obj, market):
    # This is only used for new order entry/validation
    try:
        # 1. Try Forward/Current Window (Standard for T+1 / Live)
        start = date_obj
        end = date_obj + timedelta(days=5)
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        local_p = 0.0
        
        if not df.empty:
            # Use 'Open' if available for the requested date/future
            local_p = extract_scalar(df['Open'].dropna().iloc[0])
        else:
            # 2. Fallback: If no price (e.g., Market Closed/Pre-market), check last 7 days
            end_back = date_obj
            start_back = date_obj - timedelta(days=7)
            df_back = yf.download(ticker, start=start_back, end=end_back, progress=False)
            
            if not df_back.empty:
                # Use the last known 'Close' for compliance validation
                local_p = extract_scalar(df_back['Close'].dropna().iloc[-1])
            else:
                return 0.0, 0.0

        usd_p = local_p

        if market != "US":
            cfg = MARKET_CONFIG.get(market)
            if cfg and cfg['fx']:
                # Same fallback logic for FX
                fx_df = yf.download(cfg['fx'], start=start, end=end, progress=False)
                rate = 0.0
                if not fx_df.empty:
                    rate = extract_scalar(fx_df['Open'].dropna().iloc[0])
                else:
                    end_back = date_obj
                    start_back = date_obj - timedelta(days=7)
                    fx_df_back = yf.download(cfg['fx'], start=start_back, end=end_back, progress=False)
                    if not fx_df_back.empty:
                        rate = extract_scalar(fx_df_back['Close'].dropna().iloc[-1])
                
                if rate > 0:
                    if market == "UK": local_p = local_p / 100.0
                    usd_p = (local_p / 100.0 if market == "UK" else local_p) / rate
                else:
                    # If we have price but no FX, return 0 (safer than wrong price)
                    return 0.0, 0.0
                    
        return local_p, usd_p
    except: return 0.0, 0.0

# --- NEW HELPER: Fetch raw dicts to enable caching ---
def fetch_user_transactions(user_id, session_obj):
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    return [
        {
            'ticker': t.ticker,
            'market': t.market,
            'trans_type': t.trans_type,
            'date': t.date,
            'amount': t.amount,
            'quantity': t.quantity,
            'local_price': t.local_price,
            'price': t.price
        }
        for t in txs
    ]

# ==========================================
# 4. CORE LOGIC (CACHED)
# ==========================================

@st.cache_data(ttl=60, show_spinner=False)
def calculate_portfolio_state_cached(txs_data, initial_capital):
    state = {
        "cash": initial_capital,
        "positions": {}, 
        "realized_pnl_by_side": {}, 
        "equity": 0.0
    }

    # Ledger Replay
    for t in txs_data:
        tik = t['ticker']
        if tik not in state["positions"]:
            state["positions"][tik] = {
                "qty": 0.0, 
                "avg_cost": 0.0, 
                "avg_cost_local": 0.0,
                "type": "FLAT", "market": t['market'], 
                "first_entry": None
            }
        
        pos = state["positions"][tik]
        
        if t['trans_type'] == "BUY":
            state["cash"] -= t['amount']
            new_val = (pos["qty"] * pos["avg_cost"]) + t['amount']
            
            t_local_price = t['local_price'] if t['local_price'] else t['price']
            new_val_local = (pos["qty"] * pos["avg_cost_local"]) + (t['quantity'] * t_local_price)
            
            pos["qty"] += t['quantity']
            pos["avg_cost"] = new_val / pos["qty"] if pos["qty"] > 0 else 0.0
            pos["avg_cost_local"] = new_val_local / pos["qty"] if pos["qty"] > 0 else 0.0
            
            pos["type"] = "LONG"
            if not pos["first_entry"]: pos["first_entry"] = t['date']

        elif t['trans_type'] == "SELL":
            state["cash"] += t['amount']
            cost_basis = t['quantity'] * pos["avg_cost"]
            pnl = t['amount'] - cost_basis
            
            key = (tik, 'LONG')
            state["realized_pnl_by_side"][key] = state["realized_pnl_by_side"].get(key, 0) + pnl
            
            pos["qty"] -= t['quantity']
            if pos["qty"] <= 0.001: 
                del state["positions"][tik]

        elif t['trans_type'] == "SHORT_SELL":
            state["cash"] += t['amount']
            curr_val = abs(pos["qty"]) * pos["avg_cost"]
            new_val = curr_val + t['amount']

            t_local_price = t['local_price'] if t['local_price'] else t['price']
            curr_val_local = abs(pos["qty"]) * pos["avg_cost_local"]
            new_val_local = curr_val_local + (t['quantity'] * t_local_price)

            pos["qty"] -= t['quantity']
            pos["avg_cost"] = new_val / abs(pos["qty"]) if abs(pos["qty"]) > 0 else 0.0
            pos["avg_cost_local"] = new_val_local / abs(pos["qty"]) if abs(pos["qty"]) > 0 else 0.0
            
            pos["type"] = "SHORT"
            if not pos["first_entry"]: pos["first_entry"] = t['date']

        elif t['trans_type'] == "BUY_TO_COVER":
            state["cash"] -= t['amount']
            cost_basis = t['quantity'] * pos["avg_cost"]
            pnl = cost_basis - t['amount']
            
            key = (tik, 'SHORT')
            state["realized_pnl_by_side"][key] = state["realized_pnl_by_side"].get(key, 0) + pnl
            
            pos["qty"] += t['quantity']
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

@st.cache_data(ttl=300, show_spinner=False)
def get_ytd_performance_cached(txs_data, initial_capital):
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime.now()
    
    if not txs_data:
        dates = pd.date_range(start_date, end_date, freq='B')
        df = pd.DataFrame({'Date': dates, 'Equity': initial_capital})
        df['Return %'] = 0.0
        # FIX: Initialize PnL columns for empty DataFrame
        df['Long PnL'] = 0.0
        df['Short PnL'] = 0.0
        return df, pd.Series()

    ticker_market_map = {}
    for t in txs_data:
        ticker_market_map[t['ticker']] = t['market']

    tickers = list(ticker_market_map.keys())
    
    fx_tickers = set()
    for m in ticker_market_map.values():
        if m and MARKET_CONFIG.get(m, {}).get('fx'):
            fx_tickers.add(MARKET_CONFIG[m]['fx'])
    
    all_tickers = tickers + list(fx_tickers)
    
    # Safely find min date
    first_tx_date = min(t['date'] for t in txs_data)
    fetch_start = min(start_date, first_tx_date) - timedelta(days=5)
    
    batch_data = fetch_batch_data(all_tickers, fetch_start)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    curve = []
    
    if not batch_data.empty:
        batch_data.index = pd.to_datetime(batch_data.index).normalize()
    
    curr_cash = initial_capital
    holdings = {} # {ticker: quantity}
    
    tx_idx = 0
    n_txs = len(txs_data)
    
    # --- 1. PRE-ROLL (Before Start Date) ---
    while tx_idx < n_txs and txs_data[tx_idx]['date'] < start_date:
        t = txs_data[tx_idx]
        if t['trans_type'] == 'BUY':
            curr_cash -= t['amount']
            holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']
        elif t['trans_type'] == 'SELL':
            curr_cash += t['amount']
            holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']
        elif t['trans_type'] == 'SHORT_SELL':
            curr_cash += t['amount']
            holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']
        elif t['trans_type'] == 'BUY_TO_COVER':
            curr_cash -= t['amount']
            holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']
        tx_idx += 1

    # --- 2. DAILY LOOP ---
    for d in dates:
        d_norm = d.normalize()
        
        daily_long_flow = 0.0
        daily_short_flow = 0.0
        
        while tx_idx < n_txs and txs_data[tx_idx]['date'].date() <= d_norm.date():
            t = txs_data[tx_idx]
            if t['trans_type'] == 'BUY':
                curr_cash -= t['amount']
                holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']
                daily_long_flow += t['amount'] 
            elif t['trans_type'] == 'SELL':
                curr_cash += t['amount']
                holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']
                daily_long_flow -= t['amount']
            elif t['trans_type'] == 'SHORT_SELL':
                curr_cash += t['amount']
                holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']
                daily_short_flow -= t['amount']
            elif t['trans_type'] == 'BUY_TO_COVER':
                curr_cash -= t['amount']
                holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']
                daily_short_flow += t['amount']
            tx_idx += 1
            
        long_mv = 0.0
        short_mv = 0.0 
        
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
                            else: short_mv += abs(val) 
            except: pass
        
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
    
    df_curve['Long PnL'] = (df_curve['LongMV'].diff() - df_curve['LongFlow']).fillna(0)
    df_curve['Short PnL'] = (-(df_curve['ShortMV'].diff()) - df_curve['ShortFlow']).fillna(0)
    
    df_curve['Return %'] = ((df_curve['Equity'] / initial_capital) - 1) * 100
    
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

def color_pnl(val):
    """Styles negative values red and positive values green."""
    if isinstance(val, (int, float)):
        color = '#10B981' if val > 0 else '#EF4444' if val < 0 else 'black'
        return f'color: {color}'
    return ''

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
        if prev_equity == 0: prev_equity = initial_capital 
        
        total_ret = (month_end_equity - prev_equity) / prev_equity
        
        long_pnl_sum = group['Long PnL'].sum()
        short_pnl_sum = group['Short PnL'].sum()
        
        long_contrib = long_pnl_sum / prev_equity
        short_contrib = short_pnl_sum / prev_equity
        
        monthly_stats.append({
            "Metric": month.strftime('%Y-%b'), # Will be transposed
            "Long Ret %": long_contrib * 100,
            "Short Ret %": short_contrib * 100,
            "Total Ret %": total_ret * 100
        })
        
        prev_equity = month_end_equity
    
    # Transpose logic
    df = pd.DataFrame(monthly_stats)
    if not df.empty:
        df = df.set_index("Metric").T
        return df
        
    return pd.DataFrame()

def analyst_page(user, session_obj, is_pm_view=False):
    # --- ADDED: Refresh Button ---
    col_top, col_refresh = st.columns([6,1])
    with col_top:
        if not is_pm_view:
            st.markdown(f"## 🚀 Welcome, {user.username} ({user.role.capitalize()})")
        else:
            st.markdown(f"### Viewing: {user.username}")
    with col_refresh:
        if st.button("🔄 Refresh Data", key=f"refresh_{user.id}"):
            st.cache_data.clear()
            st.rerun()

    txs_data = fetch_user_transactions(user.id, session_obj)
    state = calculate_portfolio_state_cached(txs_data, user.initial_capital)
    
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

    # --- EXPOSURE WARNINGS ---
    long_pct = (long_exp / state['equity']) * 100
    short_pct = (short_exp / state['equity']) * 100
    
    warnings = []
    if long_pct < 90.0:
        warnings.append(f"⚠️ **Long Exposure Low:** Current {long_pct:.1f}% (Target > 90%)")
    if short_pct < 30.0:
        warnings.append(f"⚠️ **Short Exposure Low:** Current {short_pct:.1f}% (Target 30% - 50%)")
        
    if warnings:
        with st.container():
            for w in warnings:
                st.warning(w)

    st.markdown("---")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Performance Chart", "Monthly Returns", "Current Holdings", "Pending Orders", "All Historical Positions", "Transaction Log"])
    
    df_c, spy_c = get_ytd_performance_cached(txs_data, user.initial_capital)

    with t1:
        render_chart(df_c, spy_c)

    with t2:
        st.subheader("Monthly Return Attribution")
        m_df = render_monthly_breakdown(df_c, user.initial_capital)
        if not m_df.empty:
            # Columns are now Months. Apply color map to all.
            st.dataframe(
                m_df.style.format("{:+.2f}%").map(color_pnl),
                use_container_width=True
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
                }).map(color_pnl, subset=["Return %", "Unrealized PnL", "Realized PnL"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active positions.")

    with t4:
        st.subheader("Queued (Pending) Orders")
        # Query database directly for pending orders
        pending_txs = session_obj.query(Transaction).filter_by(user_id=user.id, status='PENDING').order_by(Transaction.date).all()
        
        if pending_txs:
            # Table View
            p_data = []
            for pt in pending_txs:
                p_data.append({
                    "ID": pt.id,
                    "Date": pt.date.strftime('%Y-%m-%d %H:%M'),
                    "Market": pt.market,
                    "Type": pt.trans_type,
                    "Ticker": pt.ticker,
                    "Shares": f"{pt.quantity:,.0f}",
                    "Est. Amount ($)": f"${pt.amount:,.0f}"
                })
            
            st.table(pd.DataFrame(p_data))
            
            if not is_pm_view:
                st.caption("Select an Order ID to cancel:")
                col_del_1, col_del_2 = st.columns([3, 1])
                with col_del_1:
                    to_delete = st.selectbox("Order ID", options=[p["ID"] for p in p_data], label_visibility="collapsed")
                with col_del_2:
                    if st.button("Cancel Order", type="primary"):
                        if to_delete:
                            t_to_del = session_obj.query(Transaction).filter_by(id=to_delete).first()
                            if t_to_del:
                                session_obj.delete(t_to_del)
                                session_obj.commit()
                                st.success(f"Order {to_delete} cancelled.")
                                time.sleep(1); st.rerun()
            
            st.caption("Orders will be picked up by the execution engine shortly.")
        else:
            st.info("No pending orders in queue.")

    with t5:
        st.subheader("All Traded Positions (Historical)")
        all_keys = set(state['realized_pnl_by_side'].keys())
        for tik, pos in state['positions'].items():
            all_keys.add((tik, pos['type']))
        
        history_data = []
        
        # Stats for Hit Ratio
        stats = {"LONG": {"win": 0, "total": 0}, "SHORT": {"win": 0, "total": 0}}
        
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
            
            if side in stats:
                stats[side]["total"] += 1
                if total_pnl > 0: stats[side]["win"] += 1
        
        # Calculate Hit Ratios
        col_s1, col_s2, col_s3 = st.columns(3)
        
        long_rate = (stats["LONG"]["win"] / stats["LONG"]["total"] * 100) if stats["LONG"]["total"] > 0 else 0
        short_rate = (stats["SHORT"]["win"] / stats["SHORT"]["total"] * 100) if stats["SHORT"]["total"] > 0 else 0
        
        total_wins = stats["LONG"]["win"] + stats["SHORT"]["win"]
        total_pos = stats["LONG"]["total"] + stats["SHORT"]["total"]
        total_rate = (total_wins / total_pos * 100) if total_pos > 0 else 0
        
        col_s1.metric("Long Hit Ratio", f"{long_rate:.1f}%", f"{stats['LONG']['win']}/{stats['LONG']['total']}")
        col_s2.metric("Short Hit Ratio", f"{short_rate:.1f}%", f"{stats['SHORT']['win']}/{stats['SHORT']['total']}")
        col_s3.metric("Total Hit Ratio", f"{total_rate:.1f}%", f"{total_wins}/{total_pos}")
            
        if history_data:
            hist_df = pd.DataFrame(history_data).sort_values("Total Net PnL", ascending=False)
            st.dataframe(
                hist_df.style.format({
                    "Total Realized PnL": "{:+,.0f}",
                    "Total Unrealized PnL": "{:+,.0f}",
                    "Total Net PnL": "{:+,.0f}"
                }).map(color_pnl, subset=["Total Realized PnL", "Total Unrealized PnL", "Total Net PnL"]),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No trading history.")

    with t6:
        st.subheader("Transaction History")
        if txs_data:
            hist_data = []
            sorted_txs = sorted(txs_data, key=lambda x: x['date'], reverse=True)
            for t in sorted_txs:
                cur = MARKET_CONFIG.get(t['market'], {}).get('currency', 'USD')
                p_display = t['local_price'] if t['local_price'] else t['price']
                if not p_display: p_display = 0.0
                
                if cur == 'USD':
                    p_str = f"${p_display:,.2f}"
                else:
                    p_str = f"{p_display:,.2f} {cur}"

                hist_data.append({
                    "Date": t['date'].strftime('%Y-%m-%d'),
                    "Ticker": t['ticker'],
                    "Type": t['trans_type'],
                    "Amount (USD)": t['amount'],
                    "Fill Price (Local)": p_str,
                    "Quantity": t['quantity'] if t['quantity'] else 0
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
            if user.role == 'trader':
                st.info("ℹ️ **Trader Mode Active**: Compliance rules (position limits, frequency checks, lockups) are bypassed.")
            else:
                st.markdown("""
                * **Longs:** Position: 10% - 40% of Equity. Total Longs: Min 90% - Max 100% of Equity. Max 5 Names.
                * **Shorts:** Position: 10% - 30% of Equity. Total Shorts: Max 50% of Equity. Max 3 Names.
                * **Lockup:** 30 Days. *Exception: Profit > 15% or Loss > 20% on Shorts.*
                * **Frequency:** Cannot repeat an unwind trade (Sell/Cover) on the same stock within 5 days. Entries are allowed.
                * **Execution:** Trades placed during market hours (plus delay) execute near current price. Trades placed *after* market close will execute at the next trading day's available price.
                """)

        with st.form("order_form", clear_on_submit=True):
            col_a, col_b, col_c, col_d = st.columns(4)
            mkt = col_a.selectbox("Market", list(MARKET_CONFIG.keys()))
            tik = col_b.text_input("Ticker Symbol").strip()
            side = col_c.selectbox("Order Type", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
            
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
                
                # --- NEW COMPLIANCE ENGINE (POST-TRADE SIMULATION) ---
                error_msg = None
                warning_msg = None
                
                # --- TRADER ROLE BYPASS LOGIC ---
                if user.role != 'trader':
                    current_longs = {t: p for t, p in state['positions'].items() if p['type'] == 'LONG'}
                    current_shorts = {t: p for t, p in state['positions'].items() if p['type'] == 'SHORT'}
                    current_pos = state['positions'].get(final_tik)
                    
                    est_amount = qty_input * est_usd_p
                    
                    # Simulate Equity (Assume Cash Swap = No Immediate Equity Change)
                    sim_equity = state['equity'] 
                    
                    # Simulate Positions
                    sim_longs = current_longs.copy()
                    sim_shorts = current_shorts.copy()
                    
                    # -----------------------------------------------
                    # 1. FULL UNWIND DETECTION (OVERRIDE)
                    # -----------------------------------------------
                    is_full_unwind = False
                    current_qty = current_pos['qty'] if current_pos else 0
                    
                    if side == 'SELL' and current_pos and current_pos['type'] == 'LONG':
                        if abs(current_qty - qty_input) < 0.001: is_full_unwind = True
                    elif side == 'BUY_TO_COVER' and current_pos and current_pos['type'] == 'SHORT':
                        if abs(abs(current_qty) - qty_input) < 0.001: is_full_unwind = True

                    if is_full_unwind:
                        warning_msg = "ℹ️ Full position unwind detected. Compliance checks (Frequency, Lockup) bypassed."

                    # Update specific position
                    if side == 'BUY':
                        if final_tik in sim_longs:
                            old_val = sim_longs[final_tik]['mkt_val']
                            sim_longs[final_tik] = {'mkt_val': old_val + est_amount} # Simply update val
                        else:
                            sim_longs[final_tik] = {'mkt_val': est_amount}
                            
                    elif side == 'SELL':
                        if is_full_unwind:
                            if final_tik in sim_longs: del sim_longs[final_tik]
                        elif final_tik in sim_longs:
                            old_val = sim_longs[final_tik]['mkt_val']
                            new_val = old_val - est_amount
                            if new_val < 100: # Assuming closed
                                del sim_longs[final_tik]
                            else:
                                sim_longs[final_tik] = {'mkt_val': new_val}

                    elif side == 'SHORT_SELL':
                        if final_tik in sim_shorts:
                            old_val = sim_shorts[final_tik]['mkt_val']
                            sim_shorts[final_tik] = {'mkt_val': old_val + est_amount}
                        else:
                            sim_shorts[final_tik] = {'mkt_val': est_amount}
                    
                    elif side == 'BUY_TO_COVER':
                        if is_full_unwind:
                            if final_tik in sim_shorts: del sim_shorts[final_tik]
                        elif final_tik in sim_shorts:
                            old_val = sim_shorts[final_tik]['mkt_val']
                            new_val = old_val - est_amount
                            if new_val < 100:
                                del sim_shorts[final_tik]
                            else:
                                sim_shorts[final_tik] = {'mkt_val': new_val}

                    # --- CHECK RULES ON SIMULATED STATE ---
                    sim_long_total = sum(p['mkt_val'] for p in sim_longs.values())
                    sim_short_total = sum(p['mkt_val'] for p in sim_shorts.values())

                    # 1. LONG RULES (BUY or SELL/TRIM)
                    if side in ['BUY', 'SELL']:
                        if len(sim_longs) > 5:
                            error_msg = f"Compliance Violation: Max 5 Long positions allowed (Projected: {len(sim_longs)})."
                        
                        # Check SIZE of THIS position (if it still exists/wasn't closed)
                        if final_tik in sim_longs:
                            this_val = sim_longs[final_tik]['mkt_val']
                            pct = (this_val / sim_equity) * 100
                            if not (10.0 <= pct <= 40.0):
                                if pct > 40.0: 
                                    error_msg = f"Compliance Violation: Projected Long Position {pct:.1f}% exceeds max limit (40%)."
                                elif pct < 10.0: 
                                    error_msg = f"Compliance Violation: Projected Long Position {pct:.1f}% is below min limit (10%)."
                        
                        # Total Long Exposure Checks
                        long_exp_pct = (sim_long_total / sim_equity) * 100
                        if long_exp_pct < 90.0:
                             w = f"Total Long Exposure {long_exp_pct:.1f}% below target 90%."
                             warning_msg = f"{warning_msg} {w}" if warning_msg else f"⚠️ Warning: {w}"
                        
                        if long_exp_pct > 100.0:
                            error_msg = f"Compliance Violation: Total Long Exposure {long_exp_pct:.1f}% exceeds max limit (100%)."

                    # 2. SHORT RULES (SHORT_SELL or COVER/TRIM)
                    if side in ['SHORT_SELL', 'BUY_TO_COVER']:
                        if len(sim_shorts) > 3:
                            error_msg = f"Compliance Violation: Max 3 Short positions allowed (Projected: {len(sim_shorts)})."
                        
                        if final_tik in sim_shorts:
                            this_val = sim_shorts[final_tik]['mkt_val']
                            pct = (this_val / sim_equity) * 100
                            if not (10.0 <= pct <= 30.0):
                                 if pct > 30.0: 
                                     error_msg = f"Compliance Violation: Projected Short Position {pct:.1f}% exceeds max limit (30%)."
                                 elif pct < 10.0: 
                                     error_msg = f"Compliance Violation: Projected Short Position {pct:.1f}% is below min limit (10%)."
                        
                        total_pct = (sim_short_total / sim_equity) * 100
                        if total_pct > 50.0:
                            error_msg = f"Compliance Violation: Total Short Exposure {total_pct:.1f}% exceeds max limit (50%)."
                        elif total_pct < 30.0:
                             w = "Total Short Exposure below 30%."
                             warning_msg = f"{warning_msg} {w}" if warning_msg else f"⚠️ Warning: {w}"

                    # 3. FREQUENCY LIMIT (Modified: Block repeated unwinds only)
                    # Bypass if is_full_unwind
                    if side in ['SELL', 'BUY_TO_COVER'] and not is_full_unwind:
                        check_start_date = eval_date - timedelta(days=5)
                        recent_unwinds = session_obj.query(Transaction).filter(
                            Transaction.user_id == user.id,
                            Transaction.ticker == final_tik,
                            Transaction.status == 'FILLED',
                            Transaction.trans_type == side,
                            Transaction.date >= check_start_date
                        ).count()
                        
                        if recent_unwinds > 0:
                            error_msg = f"Compliance Violation: Unwind Frequency. You have already executed a {side} on {final_tik} in the last 5 days. Repeated unwinding is not permitted."

                    # 4. LOCKUP (Existing logic remains valid for closing trades)
                    # Bypass if is_full_unwind
                    if side in ['SELL', 'BUY_TO_COVER'] and not is_full_unwind:
                        curr_date = datetime.combine(d_val, datetime.min.time()) if test_mode else datetime.now()
                        if not current_pos:
                             error_msg = "Cannot close a position you don't hold."
                        else:
                            days_held = (curr_date - current_pos['first_entry']).days
                            lockup_days = 2
                            is_violation = days_held < lockup_days
                            
                            if is_violation and side == 'BUY_TO_COVER':
                                entry_p = current_pos['avg_cost']
                                curr_p = est_usd_p
                                pnl_pct = (entry_p - curr_p) / entry_p
                                if pnl_pct > 0.15 or pnl_pct < -0.20:
                                    is_violation = False
                                    warning_msg = f"⚠️ Lockup bypassed due to PnL trigger ({pnl_pct*100:.1f}%)."
                            
                            if is_violation:
                                 error_msg = f"Compliance Violation: Position held for {days_held} days. Min holding {lockup_days} days."
                else:
                    # Trader Role: Bypass all checks
                    warning_msg = "⚠️ Trader Role: Compliance checks bypassed."
                    est_amount = qty_input * est_usd_p
                    
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
                        st.cache_data.clear()
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
    # --- ADDED: Refresh Button ---
    col_top, col_refresh = st.columns([6,1])
    with col_top:
        st.title("📖 Portfolio Manager Dashboard")
    with col_refresh:
        if st.button("🔄 Refresh Data", key="pm_refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # MODIFIED: Include both Analysts and Traders in PM view
    analysts = session_obj.query(User).filter(User.role.in_(['analyst', 'trader'])).all()
    if not analysts:
        st.warning("No analysts or traders found.")
        return

    summary = []
    
    monthly_data_frames = {} 
    
    progress = st.progress(0, text="Calculating Portfolio Analytics...")
    
    # New: Collect just the Total Ret % row for the summary table
    analyst_monthly_summary = {}

    for idx, a in enumerate(analysts):
        txs_data = fetch_user_transactions(a.id, session_obj)
        s = calculate_portfolio_state_cached(txs_data, a.initial_capital)
        
        total_ret_pct = ((s['equity'] / a.initial_capital) - 1) * 100
        
        long_exp = sum(p['mkt_val'] for p in s['positions'].values() if p['type'] == 'LONG')
        short_exp = sum(p['mkt_val'] for p in s['positions'].values() if p['type'] == 'SHORT')
        
        gross_exp_pct = ((long_exp + short_exp) / s['equity']) * 100
        net_exp_pct = ((long_exp - short_exp) / s['equity']) * 100
        
        summary.append({
            "Analyst": a.username,
            "Role": a.role.capitalize(), # Show role in table
            "Equity": s['equity'], 
            "Cash %": (s['cash'] / s['equity']) * 100,
            "Gross Exp": gross_exp_pct,
            "Net Exp": net_exp_pct,
            "Total Ret %": total_ret_pct
        })
        
        df_c, _ = get_ytd_performance_cached(txs_data, a.initial_capital)
        m_df = render_monthly_breakdown(df_c, a.initial_capital)
        if not m_df.empty:
            monthly_data_frames[a.username] = m_df
            # Extract Total Ret row for the horizontal summary
            if "Total Ret %" in m_df.index:
                analyst_monthly_summary[a.username] = m_df.loc["Total Ret %"]
            
        progress.progress((idx + 1) / len(analysts))

    progress.empty()

    df_sum = pd.DataFrame(summary).sort_values("Total Ret %", ascending=False)
    
    st.subheader("Leaderboard")
    st.dataframe(
        df_sum.style.format({
            "Equity": "${:,.0f}", 
            "Cash %": "{:.1f}%", 
            "Gross Exp": "{:.1f}%",
            "Net Exp": "{:.1f}%",
            "Total Ret %": "{:+.2f}%"
        }).map(color_pnl, subset=["Total Ret %"]),
        use_container_width=True, hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Analyst Monthly Total Returns")
    if analyst_monthly_summary:
        # Create DataFrame from dict where keys are indices (Analysts) and values are Series (Months)
        monthly_comp_df = pd.DataFrame(analyst_monthly_summary).T # Analysts as rows, Months as cols
        st.dataframe(
            monthly_comp_df.style.format("{:+.2f}%").map(color_pnl),
            use_container_width=True
        )
    else:
        st.info("No monthly data.")

    st.markdown("---")
    st.subheader("Monthly Returns Breakdown (Detailed)")
    
    if monthly_data_frames:
        tabs = st.tabs(list(monthly_data_frames.keys()))
        for i, (name, df) in enumerate(monthly_data_frames.items()):
            with tabs[i]:
                # Already transposed in helper
                st.dataframe(
                    df.style.format("{:+.2f}%").map(color_pnl),
                    use_container_width=True
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
            # MODIFIED: Added 'trader' to role selection
            r = st.selectbox("Role", ["analyst", "pm", "trader"])
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
        elif user.role == 'trader':
            # Traders use the analyst page but bypass compliance
            analyst_page(user, session)
        elif user.role == 'pm':
            pm_page(user, session)
            
    session.close()

if __name__ == "__main__":
    main()
