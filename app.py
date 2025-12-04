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

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #1f2937; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #6b7280; }
    .stDataFrame { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #111827; }
    .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }
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
        data = yf.download(tickers, start=start_date - timedelta(days=7), progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        elif data.empty: return pd.DataFrame()
        
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

def get_latest_price_snapshot(ticker, market):
    return get_historical_price(ticker, datetime.now() - timedelta(days=5), market)

# ==========================================
# 4. CORE LOGIC
# ==========================================

def calculate_portfolio_state(user_id, session_obj):
    user = session_obj.query(User).filter_by(id=user_id).first()
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    
    state = {
        "cash": user.initial_capital,
        "positions": {}, 
        "realized_pnl_by_side": {}, 
        "equity": 0.0
    }

    # Ledger Replay
    for t in txs:
        tik = t.ticker
        if tik not in state["positions"]:
            state["positions"][tik] = {
                "qty": 0.0, "avg_cost": 0.0, "avg_cost_local": 0.0,
                "type": "FLAT", "market": t.market, 
                "first_entry": None
            }
        
        pos = state["positions"][tik]
        
        if t.trans_type == "BUY":
            state["cash"] -= t.amount
            new_val = (pos["qty"] * pos["avg_cost"]) + t.amount
            
            # Weighted Average Cost Local
            t_local_price = t.local_price if t.local_price else t.price # Fallback
            new_val_local = (pos["qty"] * pos["avg_cost_local"]) + (t.quantity * t_local_price)
            
            pos["qty"] += t.quantity
            
            if pos["qty"] > 0:
                pos["avg_cost"] = new_val / pos["qty"]
                pos["avg_cost_local"] = new_val_local / pos["qty"]
            else:
                pos["avg_cost"] = 0.0
                pos["avg_cost_local"] = 0.0
                
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
            
            t_local_price = t.local_price if t.local_price else t.price
            curr_val_local = abs(pos["qty"]) * pos["avg_cost_local"]
            new_val_local = curr_val_local + (t.quantity * t_local_price)
            
            pos["qty"] -= t.quantity
            
            if abs(pos["qty"]) > 0:
                pos["avg_cost"] = new_val / abs(pos["qty"])
                pos["avg_cost_local"] = new_val_local / abs(pos["qty"])
            else:
                pos["avg_cost"] = 0.0
                pos["avg_cost_local"] = 0.0

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

def get_detailed_performance(user_id, session_obj):
    """
    Returns daily curve AND monthly breakdown (Long/Short/Total).
    """
    txs = session_obj.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    user = session_obj.query(User).filter_by(id=user_id).first()
    
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime.now()
    
    if not txs:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

    ticker_market_map = {t.ticker: t.market for t in txs}
    tickers = list(ticker_market_map.keys())
    
    fx_tickers = set()
    for m in ticker_market_map.values():
        if m and MARKET_CONFIG.get(m, {}).get('fx'):
            fx_tickers.add(MARKET_CONFIG[m]['fx'])
    
    fetch_start = min(start_date, txs[0].date) - timedelta(days=5)
    batch_data = fetch_batch_data(tickers + list(fx_tickers), fetch_start)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    curve = []
    
    if not batch_data.empty:
        batch_data.index = pd.to_datetime(batch_data.index).normalize()
    
    curr_cash = user.initial_capital
    holdings = {} # {ticker: qty}
    
    tx_idx = 0
    n_txs = len(txs)
    
    # 1. Init Holdings before Start Date
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

    # 2. Daily Loop to track Long/Short PnL
    prev_prices = {} # {ticker: usd_price}
    
    for d in dates:
        d_norm = d.normalize()
        
        # Process Day's Trades
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
            
        long_val = 0.0
        short_val = 0.0
        day_long_pnl = 0.0
        day_short_pnl = 0.0
        
        if not batch_data.empty:
            try:
                row = pd.Series()
                if d_norm in batch_data.index: row = batch_data.loc[d_norm]
                else:
                    idx = batch_data.index.get_indexer([d_norm], method='pad')[0]
                    if idx != -1: row = batch_data.iloc[idx]

                if not row.empty:
                    for tik, qty in holdings.items():
                        if abs(qty) > 0.001:
                            # Pricing Logic
                            p_local = float(row[tik]) if tik in row else 0.0
                            if pd.isna(p_local): p_local = 0.0
                            
                            mkt = ticker_market_map.get(tik, 'US')
                            fx_sym = MARKET_CONFIG.get(mkt, {}).get('fx')
                            p_usd = p_local
                            if fx_sym and fx_sym in row:
                                rate = float(row[fx_sym])
                                if mkt == "UK": p_local /= 100.0
                                if rate > 0: p_usd = p_local / rate
                                else: p_usd = 0.0
                            elif mkt == "UK": p_usd = p_local / 100.0 # Fallback UK
                            
                            val = qty * p_usd
                            if qty > 0: long_val += val
                            else: short_val += abs(val)
                            
                            # PnL Calculation (Change in Price * Qty)
                            # Approximate daily PnL attribution
                            prev_p = prev_prices.get(tik, p_usd) 
                            # If new position today, prev_p is essentially entry price, 
                            # but simpler to rely on Equity change for Total and just track attribution roughly
                            price_change = p_usd - prev_p
                            
                            if qty > 0: day_long_pnl += (qty * price_change)
                            else: day_short_pnl += (qty * price_change) # Short qty is neg, price up = neg pnl
                            
                            prev_prices[tik] = p_usd
            except: pass
        
        equity = curr_cash + long_val - short_val
        curve.append({
            "Date": d, "Equity": equity, 
            "Long_PnL": day_long_pnl, "Short_PnL": day_short_pnl
        })

    df_curve = pd.DataFrame(curve)
    df_curve['Return %'] = ((df_curve['Equity'] / user.initial_capital) - 1) * 100
    
    # Calculate Monthly Breakdown
    # Resample Daily PnL
    m_stats = df_curve.set_index('Date').resample('ME').agg({
        'Equity': 'last',
        'Long_PnL': 'sum',
        'Short_PnL': 'sum'
    })
    
    m_stats['Prev Equity'] = m_stats['Equity'].shift(1).fillna(user.initial_capital)
    m_stats['Long Ret %'] = (m_stats['Long_PnL'] / m_stats['Prev Equity']) * 100
    m_stats['Short Ret %'] = (m_stats['Short_PnL'] / m_stats['Prev Equity']) * 100
    m_stats['Total Ret %'] = ((m_stats['Equity'] / m_stats['Prev Equity']) - 1) * 100
    
    # SPY
    spy_ret = pd.Series()
    try:
        spy = fetch_batch_data(["SPY"], fetch_start)
        spy = spy[(spy.index >= pd.Timestamp(start_date)) & (spy.index <= pd.Timestamp(end_date))]
        if not spy.empty and 'SPY' in spy.columns:
            start_price = extract_scalar(spy['SPY'].iloc[0])
            spy_ret = ((spy['SPY'] / start_price) - 1) * 100
    except: pass
    
    return df_curve, m_stats, spy_ret

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
        fig.add_trace(go.Scatter(x=df_c['Date'], y=df_c['Return %'], name="Portfolio", line=dict(color='#2563EB', width=3)))
        if not spy_c.empty:
            fig.add_trace(go.Scatter(x=spy_c.index, y=spy_c.values, name="S&P 500", line=dict(color='#9CA3AF', dash='dot', width=2)))
        fig.update_layout(title="YTD Performance", template="plotly_white", hovermode="x unified", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data.")

def analyst_page(user, session_obj, is_pm_view=False):
    if not is_pm_view:
        st.markdown(f"## 🚀 Welcome, {user.username}")
    else:
        st.markdown(f"### Viewing Analyst: {user.username}")
    
    state = calculate_portfolio_state(user.id, session_obj)
    
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Equity", f"${state['equity']:,.0f}")
        c2.metric("Cash Balance", f"${state['cash']:,.0f}")
        pnl_val = state['equity'] - user.initial_capital
        pnl_pct = (pnl_val / user.initial_capital) * 100
        c3.metric("YTD PnL", f"{pnl_val:+,.0f}", delta=f"{pnl_pct:+.2f}%")
        
        long_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'LONG')
        short_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'SHORT')
        net_exp = long_exp - short_exp
        c4.metric("Net Exposure", f"${net_exp:,.0f}")

    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["Performance", "Current Holdings", "Historical", "Transactions"])
    
    df_c, m_stats, spy_c = get_detailed_performance(user.id, session_obj)
    
    with t1:
        render_chart(df_c, spy_c)
        st.subheader("Monthly Attribution")
        if not m_stats.empty:
            # Reformat for display
            disp_m = m_stats[['Long Ret %', 'Short Ret %', 'Total Ret %']].copy()
            disp_m.index = disp_m.index.strftime('%b %Y')
            st.dataframe(disp_m.style.format("{:+.2f}%").background_gradient(cmap="RdYlGn", vmin=-5, vmax=5), use_container_width=True)
        else:
            st.info("No history yet.")
        
    with t2:
        st.subheader("Current Holdings")
        holdings_data = []
        for tik, pos in state['positions'].items():
            unrealized_pnl = pos.get('unrealized', 0)
            side_key = (tik, pos['type'])
            realized_pnl = state['realized_pnl_by_side'].get(side_key, 0.0)

            loc_p = pos.get('current_local_price', 0)
            avg_loc = pos.get('avg_cost_local', 0)
            cur = MARKET_CONFIG.get(pos['market'], {}).get('currency', 'USD')
            
            if cur == 'USD':
                p_str = f"${loc_p:,.2f}"
                avg_str = f"${avg_loc:,.2f}"
            else:
                p_str = f"{loc_p:,.2f} {cur}"
                avg_str = f"{avg_loc:,.2f} {cur}"

            invested_capital = pos['qty'] * pos['avg_cost']
            ret_pct = (unrealized_pnl / abs(invested_capital)) * 100 if invested_capital != 0 else 0.0
            
            # % of Portfolio
            pct_port = (pos['mkt_val'] / state['equity']) * 100 if state['equity'] > 0 else 0

            holdings_data.append({
                "Ticker": tik, "Type": pos['type'], "Market": pos['market'],
                "Qty": f"{pos['qty']:,.0f}", 
                "Avg Cost (Local)": avg_str, 
                "Current Price": p_str,
                "Market Val (USD)": pos.get('mkt_val', 0),
                "% Port": pct_port,
                "Unrealized PnL": unrealized_pnl,
                "Realized PnL": realized_pnl,
                "Return %": ret_pct,
            })
        
        if holdings_data:
            h_df = pd.DataFrame(holdings_data)
            st.dataframe(
                h_df.style.format({
                    "Market Val (USD)": "${:,.0f}", 
                    "% Port": "{:.1f}%",
                    "Unrealized PnL": "{:+,.0f}",
                    "Realized PnL": "{:+,.0f}",
                    "Return %": "{:+.2f}%"
                }).background_gradient(subset=["Return %"], cmap="RdYlGn", vmin=-20, vmax=20),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No active positions.")

    with t3:
        st.subheader("All Traded Positions")
        all_keys = set(state['realized_pnl_by_side'].keys())
        for tik, pos in state['positions'].items(): all_keys.add((tik, pos['type']))
        
        history_data = []
        for (tik, side) in all_keys:
            realized = state['realized_pnl_by_side'].get((tik, side), 0.0)
            unrealized = 0.0
            if tik in state['positions'] and state['positions'][tik]['type'] == side:
                unrealized = state['positions'][tik].get('unrealized', 0.0)
            
            history_data.append({
                "Ticker": tik, "Side": side,
                "Realized": realized, "Unrealized": unrealized, "Total Net": realized + unrealized
            })
            
        if history_data:
            st.dataframe(pd.DataFrame(history_data).sort_values("Total Net", ascending=False).style.format("{:+,.0f}"), use_container_width=True, hide_index=True)

    with t4:
        st.subheader("Transactions")
        hist_txs = session_obj.query(Transaction).filter_by(user_id=user.id).order_by(Transaction.date.desc()).all()
        if hist_txs:
            st.dataframe(pd.DataFrame([{
                "Date": t.date.strftime('%Y-%m-%d'), "Ticker": t.ticker, "Type": t.trans_type, 
                "Qty": t.quantity, "Price": t.price, "Amt": t.amount
            } for t in hist_txs]).style.format({"Amt": "${:,.0f}", "Price": "${:,.2f}", "Qty":"{:,.0f}"}), use_container_width=True, hide_index=True)

    if not is_pm_view:
        st.markdown("---")
        st.subheader("⚡ Execute Trade")
        
        with st.expander("Mandate Rules"):
            st.markdown("""
            * **Longs:** Position 10-40% of Equity. Total > 90%. Max 5 Names.
            * **Shorts:** Position 10-30% of Equity. Total 30-50%. Max 3 Names.
            * **Lockup:** 30 Days (Unless Short Profit > 15% or Loss > 20%).
            """)

        with st.form("order_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            mkt = c1.selectbox("Market", list(MARKET_CONFIG.keys()))
            tik = c2.text_input("Ticker").strip()
            side = c3.selectbox("Action", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
            qty = c4.number_input("Shares", min_value=1.0, step=100.0)
            note = st.text_area("Notes")
            
            test_mode = is_test_mode(session_obj)
            d_val = st.date_input("Backdate", value=datetime.now()) if test_mode else None
            
            if st.form_submit_button("Submit", type="primary"):
                if not tik: st.error("Ticker required"); st.stop()
                final_tik = format_ticker(tik, mkt)
                eval_date = datetime.combine(d_val, datetime.min.time()) if test_mode else datetime.now()
                
                est_loc, est_usd = get_historical_price(final_tik, eval_date, mkt)
                if est_usd <= 0: st.error("Price not found"); st.stop()
                
                est_amt = qty * est_usd
                equity = state['equity']
                
                # --- COMPLIANCE ---
                err = None
                
                cur_pos = state['positions'].get(final_tik)
                
                # Check Lockup Exception for Shorts
                can_skip_lockup = False
                if side == 'BUY_TO_COVER' and cur_pos:
                    # Return Formula: (AvgCost - Curr) / AvgCost -> 1 - Curr/AvgCost
                    # For Lockup we want > 15% profit or > 20% loss (Return < -20%)
                    # Note: Using Current Price for check
                    short_ret = (1 - (est_usd / cur_pos['avg_cost']))
                    if short_ret > 0.15 or short_ret < -0.20:
                        can_skip_lockup = True
                        st.success(f"Short Lockup Exception Triggered (Return: {short_ret*100:.1f}%)")

                if side in ['SELL', 'BUY_TO_COVER']:
                    if cur_pos and cur_pos['first_entry']:
                        days = (eval_date - cur_pos['first_entry']).days
                        if days < 30 and not can_skip_lockup:
                            err = f"Lockup Violation: Held {days} days."
                
                # Size Checks (New % Rules)
                if side == 'BUY':
                    # Proj Long Exposure
                    cur_long_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'LONG')
                    proj_long_exp = cur_long_exp + est_amt
                    if cur_pos: proj_pos_size = cur_pos['mkt_val'] + est_amt
                    else: proj_pos_size = est_amt
                    
                    if not (0.10 * equity <= proj_pos_size <= 0.40 * equity):
                        err = f"Long Size Violation: {proj_pos_size/equity*100:.1f}% (Limit 10-40%)"
                    
                    # Note: We don't block BUY if total < 90%, we block SELL if it drops below
                    
                    if not cur_pos:
                        l_count = len([p for p in state['positions'].values() if p['type'] == 'LONG'])
                        if l_count >= 5: err = "Max 5 Longs"

                if side == 'SELL':
                     cur_long_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'LONG')
                     proj_long_exp = cur_long_exp - est_amt
                     if proj_long_exp < 0.90 * equity:
                         st.warning(f"⚠️ Warning: Long Exposure dropping to {proj_long_exp/equity*100:.1f}% (Target > 90%)")

                if side == 'SHORT_SELL':
                    cur_short_exp = sum(p['mkt_val'] for p in state['positions'].values() if p['type'] == 'SHORT')
                    proj_short_exp = cur_short_exp + est_amt
                    
                    if cur_pos: proj_pos_size = cur_pos['mkt_val'] + est_amt
                    else: proj_pos_size = est_amt
                    
                    if not (0.10 * equity <= proj_pos_size <= 0.30 * equity):
                         err = f"Short Size Violation: {proj_pos_size/equity*100:.1f}% (Limit 10-30%)"
                    
                    if not (0.30 * equity <= proj_short_exp <= 0.50 * equity):
                         # If it's the first short, maybe allow building up, but ideally block excessive
                         if proj_short_exp > 0.50 * equity:
                             err = f"Total Short Exposure Violation: {proj_short_exp/equity*100:.1f}% (Limit 50%)"
                    
                    if not cur_pos:
                         s_count = len([p for p in state['positions'].values() if p['type'] == 'SHORT'])
                         if s_count >= 3: err = "Max 3 Shorts"

                if err: st.error(err)
                else:
                    if test_mode:
                        session_obj.add(Transaction(user_id=user.id, ticker=final_tik, market=mkt, trans_type=side, status='FILLED', date=eval_date, amount=est_amt, quantity=qty, local_price=est_loc, price=est_usd, notes=f"[BACKDATE] {note}"))
                        session_obj.commit(); st.success("Filled"); time.sleep(1); st.rerun()
                    else:
                        session_obj.add(Transaction(user_id=user.id, ticker=final_tik, market=mkt, trans_type=side, status='PENDING', amount=est_amt, quantity=qty, notes=note))
                        session_obj.commit(); st.success("Queued"); time.sleep(1); st.rerun()

def pm_page(user, session_obj):
    st.title("👨‍💼 Portfolio Manager")
    analysts = session_obj.query(User).filter_by(role='analyst').all()
    
    summary = []
    
    for a in analysts:
        s = calculate_portfolio_state(a.id, session_obj)
        _, m_stats, _ = get_detailed_performance(a.id, session_obj)
        
        last_m_ret = m_stats['Total Ret %'].iloc[-1] if not m_stats.empty else 0.0
        
        summary.append({
            "Analyst": a.username, 
            "Equity": s['equity'], 
            "Total Ret %": ((s['equity']/a.initial_capital)-1)*100,
            "Last Month %": last_m_ret
        })

    st.subheader("Summary")
    st.dataframe(pd.DataFrame(summary).style.format({"Equity": "${:,.0f}", "Total Ret %": "{:+.2f}%", "Last Month %": "{:+.2f}%"}).background_gradient(subset=["Total Ret %"], cmap="RdYlGn"), use_container_width=True, hide_index=True)
    
    st.divider()
    sel = st.selectbox("Inspect Analyst", [a.username for a in analysts])
    if sel:
        target = session_obj.query(User).filter_by(username=sel).first()
        with st.container(border=True):
            analyst_page(target, session_obj, is_pm_view=True)

def admin_page(session_obj):
    st.title("Admin")
    # Toggles and User Management logic here (Same as previous)
    curr = is_test_mode(session_obj)
    if st.toggle("Test Mode", value=curr) != curr:
        cfg = session_obj.query(SystemConfig).filter_by(key='test_mode').first()
        if not cfg: session_obj.add(SystemConfig(key='test_mode', value='True'))
        else: cfg.value = 'True' if not curr else 'False'
        session_obj.commit(); st.rerun()
        
    with st.form("new"):
        u = st.text_input("User"); p = st.text_input("Pass", type="password"); r = st.selectbox("Role", ["analyst", "pm"])
        if st.form_submit_button("Create"):
            session_obj.add(User(username=u, password_hash=bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode(), role=r)); session_obj.commit(); st.rerun()

def main():
    session = Session()
    try:
        if not session.query(User).filter_by(username='admin').first():
            session.add(User(username='admin', password_hash=bcrypt.hashpw('8848'.encode(), bcrypt.gensalt()).decode(), role='admin'))
            session.commit()
    except: pass
    
    if 'user_id' not in st.session_state: st.session_state.user_id = None
    if not st.session_state.user_id:
        u = st.text_input("User"); p = st.text_input("Pass", type="password")
        if st.button("Login"):
            user = session.query(User).filter_by(username=u).first()
            if user and bcrypt.checkpw(p.encode(), user.password_hash.encode()):
                st.session_state.user_id = user.id; st.session_state.role = user.role; st.rerun()
    else:
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        with st.sidebar:
            st.title(user.username); 
            if st.button("Logout"): st.session_state.user_id = None; st.rerun()
        
        if user.role == 'admin': admin_page(session)
        elif user.role == 'analyst': analyst_page(user, session)
        elif user.role == 'pm': pm_page(user, session)
    session.close()

if __name__ == "__main__": main()
