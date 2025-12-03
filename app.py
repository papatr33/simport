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
# 1. DATABASE MODELS (LEDGER STYLE)
# ==========================================
Base = declarative_base()

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
    # Type: BUY, SELL (for Longs), SHORT_SELL, BUY_TO_COVER (for Shorts)
    trans_type = Column(String, nullable=False) 
    
    date = Column(DateTime, default=datetime.now)
    price = Column(Float, nullable=True) # None if PENDING
    quantity = Column(Float, nullable=True)
    amount = Column(Float, nullable=True) # Dollar value (price * qty)
    
    status = Column(String, default='PENDING') # PENDING, FILLED
    notes = Column(Text, nullable=True)
    
    user = relationship("User", back_populates="transactions")

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

# --- DB Connection ---
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
try: Base.metadata.create_all(engine)
except: pass
Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# 2. CORE LOGIC (PORTFOLIO ENGINE)
# ==========================================

def get_live_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 0.0
    except: return 0.0

def get_historical_price(ticker, date_obj):
    try:
        start = date_obj
        end = date_obj + timedelta(days=5)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty:
            # Handle Multi-index
            if 'Close' in df.columns:
                val = df['Close']
                if isinstance(val, pd.DataFrame): val = val.iloc[:, 0]
                return float(val.iloc[0])
            return float(df.iloc[0, 0])
        return 0.0
    except: return 0.0

def calculate_portfolio_state(user_id):
    """
    Iterates through the Transaction Ledger to build the current state.
    Calculates Avg Cost, Realized PnL, Net Quantity, etc.
    """
    user = session.query(User).filter_by(id=user_id).first()
    # Fetch FILLED transactions sorted by date
    txs = session.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    pending = session.query(Transaction).filter_by(user_id=user_id, status='PENDING').all()

    state = {
        "cash": user.initial_capital,
        "positions": {}, # {ticker: {qty, avg_price, unrealized_pnl, type, first_entry_date}}
        "realized_pnl_ytd": {}, # {ticker: amount}
        "history": [], # For curve generation
        "short_notional_used": 0.0,
        "long_capital_used": 0.0,
        "total_realized": 0.0
    }

    for t in txs:
        tik = t.ticker
        if tik not in state["positions"]:
            state["positions"][tik] = {"qty": 0.0, "avg_cost": 0.0, "type": "FLAT", "first_entry": None}
        
        pos = state["positions"][tik]
        
        # --- LONG LOGIC ---
        if t.trans_type == "BUY":
            # Cash Out
            cost = t.amount
            state["cash"] -= cost
            
            # Update Avg Cost
            total_val = (pos["qty"] * pos["avg_cost"]) + cost
            pos["qty"] += t.quantity
            pos["avg_cost"] = total_val / pos["qty"] if pos["qty"] > 0 else 0.0
            pos["type"] = "LONG"
            if pos["first_entry"] is None: pos["first_entry"] = t.date

        elif t.trans_type == "SELL":
            # Cash In
            proceeds = t.amount
            state["cash"] += proceeds
            
            # Realized PnL (FIFO/Avg Cost assumption: realized against avg cost)
            cost_basis_of_sale = t.quantity * pos["avg_cost"]
            pnl = proceeds - cost_basis_of_sale
            state["total_realized"] += pnl
            state["realized_pnl_ytd"][tik] = state["realized_pnl_ytd"].get(tik, 0) + pnl
            
            pos["qty"] -= t.quantity
            if pos["qty"] <= 0.001: # Closed
                del state["positions"][tik]

        # --- SHORT LOGIC ---
        elif t.trans_type == "SHORT_SELL":
            # Cash Effect: Usually shorts add cash (proceeds), but we reserve it as collateral.
            # Simplified: Cash stays flat (collateral locked), we track Notional separately.
            # Or to simulate "Buying Power": Cash reduces by margin requirement. 
            # Prompt says "$3M notional to short". 
            # Let's model it: Cash increases by sale proceeds, but Liability increases.
            
            proceeds = t.amount
            state["cash"] += proceeds 
            
            # Update Avg Price (Entry Price for Short)
            # Existing Short + New Short
            current_short_val = abs(pos["qty"]) * pos["avg_cost"]
            new_short_val = proceeds
            total_short_qty = abs(pos["qty"]) + t.quantity
            
            pos["avg_cost"] = (current_short_val + new_short_val) / total_short_qty
            pos["qty"] -= t.quantity # Negative Qty for Shorts
            pos["type"] = "SHORT"
            if pos["first_entry"] is None: pos["first_entry"] = t.date

        elif t.trans_type == "BUY_TO_COVER":
            # Cash Out to buy back
            cost = t.amount
            state["cash"] -= cost
            
            # Realized PnL
            # Entry (Avg Cost) - Exit (Current Cost)
            cost_basis_of_cover = t.quantity * pos["avg_cost"]
            pnl = cost_basis_of_cover - cost
            state["total_realized"] += pnl
            state["realized_pnl_ytd"][tik] = state["realized_pnl_ytd"].get(tik, 0) + pnl
            
            pos["qty"] += t.quantity # Add back to negative number
            if abs(pos["qty"]) <= 0.001:
                del state["positions"][tik]

    # --- MARK TO MARKET (Current State) ---
    state["equity"] = state["cash"]
    state["long_mkt_val"] = 0.0
    state["short_mkt_val"] = 0.0
    
    # Pre-fetch live prices
    tickers = list(state["positions"].keys())
    live_prices = {}
    if tickers:
        try:
            # Batch download for speed
            # Use 'download' instead of Ticker for batch
            df = yf.download(tickers, period="1d", progress=False)
            if 'Close' in df.columns:
                closes = df['Close'].iloc[-1]
                if isinstance(closes, pd.Series):
                    for tk in tickers:
                        try: live_prices[tk] = float(closes[tk])
                        except: live_prices[tk] = 0.0
                else:
                    live_prices[tickers[0]] = float(closes)
        except: pass

    # Calculate Current Values
    for tik, pos in state["positions"].items():
        curr_price = live_prices.get(tik, get_live_price(tik))
        pos["current_price"] = curr_price
        
        if pos["type"] == "LONG":
            mkt_val = pos["qty"] * curr_price
            state["equity"] += mkt_val # Cash + Stock Value
            pos["mkt_value"] = mkt_val
            state["long_mkt_val"] += mkt_val
            pos["unrealized_pnl"] = mkt_val - (pos["qty"] * pos["avg_cost"])
            
        elif pos["type"] == "SHORT":
            # Equity = Cash - Liability
            # Liability = abs(Qty) * Current Price
            liability = abs(pos["qty"]) * curr_price
            state["equity"] -= liability
            pos["mkt_value"] = liability
            state["short_notional_used"] += (abs(pos["qty"]) * pos["avg_cost"]) # Based on entry
            state["short_mkt_val"] += liability
            # PnL = (Entry Price - Current Price) * Qty
            pos["unrealized_pnl"] = (pos["avg_cost"] - curr_price) * abs(pos["qty"])

    # Pending impacts cash availability but not equity yet
    state["pending_impact"] = 0.0
    for p in pending:
        state["pending_impact"] += p.amount if p.amount else 0.0

    return state, pending

# ==========================================
# 3. REPORTING & CHARTS
# ==========================================

def get_spy_benchmark(start_date):
    """Downloads SPY data and normalizes to % return."""
    try:
        spy = yf.download("SPY", start=start_date, progress=False)['Close']
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:,0]
        # Normalize: (Price / Start_Price) - 1
        start_price = float(spy.iloc[0])
        spy_ret = ((spy / start_price) - 1) * 100
        return spy_ret
    except:
        return pd.Series()

def generate_performance_data(user_id):
    # This rebuilds the daily equity curve using the same logic as 'calculate_portfolio_state'
    # but for every single day in the past. 
    # NOTE: For speed/simplicity in this file-based app, we will estimate it 
    # by taking the "Transaction Date" snapshots.
    
    # 1. Get all transactions
    txs = session.query(Transaction).filter_by(user_id=user_id, status='FILLED').order_by(Transaction.date).all()
    if not txs: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    start_date = txs[0].date
    end_date = datetime.now()
    
    # 2. Get Universe Data
    tickers = list(set([t.ticker for t in txs]))
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    
    # 3. Build Daily Equity
    user = session.query(User).filter_by(id=user_id).first()
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    equity_curve = []
    daily_breakdown = [] # For monthly table
    
    for d in dates:
        # Filter trades that happened before or on date 'd'
        # We need to reconstruct the portfolio state at end of day 'd'
        
        # Optimization: We can't run the full loop efficiently for 365 days. 
        # Simplified: We calculate Equity = Cash + Sum(Qty * Price_on_Day_D)
        
        current_cash = user.initial_capital
        current_holdings = {} # {ticker: qty}
        
        relevant_txs = [t for t in txs if t.date <= d]
        
        for t in relevant_txs:
            if t.trans_type in ['BUY', 'BUY_TO_COVER']:
                current_cash -= t.amount
                sign = 1 if t.trans_type == 'BUY' else 1 # Qty is positive in DB, we handle logic
                # Actually, BUY adds qty, COVER adds qty (reduces negative)
                # Wait, earlier logic: Short Qty is negative.
                qty_change = t.quantity
                if t.trans_type == 'BUY_TO_COVER': 
                     # If we stored Short as negative, adding positive reduces it.
                     pass 
                
            elif t.trans_type in ['SELL', 'SHORT_SELL']:
                current_cash += t.amount
                qty_change = -t.quantity

            # Update Net Qty
            current_holdings[t.ticker] = current_holdings.get(t.ticker, 0.0)
            
            if t.trans_type == 'BUY': current_holdings[t.ticker] += t.quantity
            elif t.trans_type == 'SELL': current_holdings[t.ticker] -= t.quantity
            elif t.trans_type == 'SHORT_SELL': current_holdings[t.ticker] -= t.quantity
            elif t.trans_type == 'BUY_TO_COVER': current_holdings[t.ticker] += t.quantity
            
        # Calculate Equity
        daily_long_val = 0.0
        daily_short_val = 0.0
        
        for tik, qty in current_holdings.items():
            if abs(qty) > 0.001:
                # Get price
                try:
                    if isinstance(prices, pd.DataFrame):
                        # Handle multi-column vs single column
                        if tik in prices.columns:
                            p_series = prices[tik]
                        else:
                            p_series = prices.iloc[:, 0] # Fallback
                    else:
                        p_series = prices
                    
                    # Get price at date d
                    idx = p_series.index.get_indexer([d], method='pad')[0]
                    if idx != -1:
                        p = float(p_series.iloc[idx])
                    else:
                        p = 0.0
                except: p = 0.0
                
                mkt_val = qty * p
                
                if qty > 0: daily_long_val += mkt_val
                else: daily_short_val += mkt_val # This is negative (Liability)
        
        # Equity = Cash + Longs + Shorts (Shorts are negative value here? No.)
        # Logic Recap:
        # Short Sell: Cash increases (+100). Position Qty (-1). Price (100).
        # Equity = 100 + (-1 * 100) = 0. Correct.
        # Price drops to 90.
        # Equity = 100 + (-1 * 90) = 10. Correct.
        
        total_equity = current_cash + daily_long_val + daily_short_val
        equity_curve.append({"Date": d, "Equity": total_equity, "Return %": ((total_equity/user.initial_capital)-1)*100})
        
        # For Monthly Table
        daily_breakdown.append({
            "Date": d, 
            "Long PnL": daily_long_val, # Approx
            "Short PnL": daily_short_val, # Approx
            "Total Equity": total_equity
        })
        
    df_curve = pd.DataFrame(equity_curve)
    df_breakdown = pd.DataFrame(daily_breakdown)
    
    # Fetch SPY
    spy_curve = get_spy_benchmark(start_date)
    
    return df_curve, spy_curve, df_breakdown

def get_monthly_table(df_breakdown):
    if df_breakdown.empty: return pd.DataFrame()
    
    # Resample to Monthly End
    df_breakdown.set_index('Date', inplace=True)
    monthly = df_breakdown.resample('ME').last()
    
    # Calculate % Change
    monthly['Total Return'] = monthly['Total Equity'].pct_change()
    
    # Format for display
    display_df = monthly[['Total Equity', 'Total Return']].copy()
    display_df['Total Return'] = display_df['Total Return'] * 100
    display_df.index = display_df.index.strftime('%Y-%B')
    
    return display_df

# ==========================================
# 4. HELPERS
# ==========================================
def format_ticker(symbol, market):
    symbol = symbol.strip().upper()
    suffixes = {"US": "", "Hong Kong": ".HK", "China (Shanghai)": ".SS", "China (Shenzhen)": ".SZ", "Japan": ".T", "UK": ".L", "France": ".PA"}
    return f"{symbol}{suffixes.get(market, '')}"

def is_test_mode():
    try:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        return cfg.value == 'True' if cfg else False
    except: return False

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

def analyst_page(user):
    st.title(f"Portfolio: {user.username}")
    
    # --- 1. Top Metrics ---
    state, pending = calculate_portfolio_state(user.id)
    
    # Constraints Check
    long_count = len([k for k,v in state['positions'].items() if v['type']=='LONG'])
    short_count = len([k for k,v in state['positions'].items() if v['type']=='SHORT'])
    cash_ok = state['cash'] < 1500000 
    
    # Status Banner
    status_cols = st.columns(4)
    status_cols[0].metric("Equity", f"${state['equity']:,.0f}")
    status_cols[1].metric("Cash", f"${state['cash']:,.0f}", delta="Too High" if not cash_ok else "Compliant", delta_color="inverse")
    status_cols[2].metric("Longs", f"{long_count}/5")
    status_cols[3].metric("Shorts", f"{short_count}/3")
    
    # --- 2. Charts (PnL vs SPY) ---
    st.divider()
    df_curve, spy_curve, df_monthly = generate_performance_data(user.id)
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Performance vs SPY")
        if not df_curve.empty:
            fig = go.Figure()
            # User Curve
            fig.add_trace(go.Scatter(x=df_curve['Date'], y=df_curve['Return %'], name='Portfolio', line=dict(color='blue')))
            # SPY Curve
            if not spy_curve.empty:
                # Align dates
                fig.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve.values, name='S&P 500', line=dict(color='gray', dash='dot')))
            
            fig.update_layout(yaxis_title="Return (%)", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history available.")

    with col_stats:
        st.subheader("Monthly Returns")
        monthly_df = get_monthly_table(df_monthly)
        if not monthly_df.empty:
            st.dataframe(monthly_df.style.format({"Total Equity": "${:,.0f}", "Total Return": "{:+.2f}%"}))
        else:
            st.write("No data.")

    # --- 3. Holdings & YTD Table ---
    st.divider()
    
    # YTD Table preparation
    ytd_data = []
    for tik, pos in state['positions'].items():
        ytd_data.append({
            "Ticker": tik,
            "Side": pos['type'],
            "Qty": pos['qty'],
            "Avg Cost": pos['avg_cost'],
            "Current": pos.get('current_price', 0),
            "Unrealized PnL": pos.get('unrealized_pnl', 0),
            "Realized PnL": state['realized_pnl_ytd'].get(tik, 0),
            "Total PnL": pos.get('unrealized_pnl', 0) + state['realized_pnl_ytd'].get(tik, 0)
        })
    # Add fully closed positions
    for tik, r_pnl in state['realized_pnl_ytd'].items():
        if tik not in state['positions']:
             ytd_data.append({
                "Ticker": tik, "Side": "-", "Qty": 0, "Avg Cost": 0, "Current": 0,
                "Unrealized PnL": 0, "Realized PnL": r_pnl, "Total PnL": r_pnl
             })

    t1, t2 = st.tabs(["Current Holdings", "YTD PnL Breakdown"])
    
    with t1:
        if state['positions']:
            h_df = pd.DataFrame([
                {"Ticker": k, "Side": v['type'], "Entry Date": v['first_entry'].date(), "Qty": v['qty'], "Mkt Value": v.get('mkt_value',0), "Unrealized": v.get('unrealized_pnl',0)}
                for k,v in state['positions'].items()
            ])
            st.dataframe(h_df.style.format({"Mkt Value":"${:,.0f}", "Unrealized":"${:,.0f}"}), use_container_width=True)
        else: st.write("Cash only.")

    with t2:
        if ytd_data:
            ytd_df = pd.DataFrame(ytd_data)
            st.dataframe(ytd_df.style.format({
                "Avg Cost":"${:,.2f}", "Current":"${:,.2f}", 
                "Unrealized PnL":"${:,.0f}", "Realized PnL":"${:,.0f}", "Total PnL":"${:,.0f}"
            }), use_container_width=True)

    # --- 4. Trade Entry (Compliance Rules) ---
    st.divider()
    st.subheader("Trade Execution")
    
    with st.expander("Compliance Rules", expanded=False):
        st.markdown("""
        1. **Longs:** $500k - $2M per name. Max 5 names.
        2. **Shorts:** $300k - $1.2M per name. Max 3 names.
        3. **Hold:** New positions must be held for > 30 days.
        4. **Cash:** Target < $1.5M.
        """)

    test_mode = is_test_mode()
    if test_mode: st.warning("TEST MODE: Backdating Enabled")

    with st.form("order_ticket"):
        c1,c2,c3,c4 = st.columns(4)
        mkt = c1.selectbox("Market", ["US", "Hong Kong", "China (Shanghai)", "China (Shenzhen)", "Japan", "UK", "France"])
        tik = c2.text_input("Ticker").strip()
        action = c3.selectbox("Action", ["BUY", "SELL", "SHORT_SELL", "BUY_TO_COVER"])
        amt = c4.number_input("Amount ($)", min_value=10000.0, step=10000.0)
        
        t_date = datetime.now()
        if test_mode: t_date = st.date_input("Date", value="today")
        notes = st.text_area("Notes")

        if st.form_submit_button("Submit Order"):
            final_tik = format_ticker(tik, mkt)
            
            # --- COMPLIANCE CHECKS ---
            valid = True
            err_msg = ""
            
            # Check 1: Count Limits (Only for Opening trades)
            if action == 'BUY' and final_tik not in state['positions'] and long_count >= 5:
                valid = False; err_msg = "Max Long count (5) reached."
            if action == 'SHORT_SELL' and final_tik not in state['positions'] and short_count >= 3:
                valid = False; err_msg = "Max Short count (3) reached."
                
            # Check 2: Size Limits
            if action in ['BUY', 'SHORT_SELL']:
                # Existing pos size
                curr_size = state['positions'].get(final_tik, {}).get('mkt_value', 0)
                proj_size = abs(curr_size) + amt
                
                if action == 'BUY':
                    if not (500000 <= proj_size <= 2000000):
                        valid = False; err_msg = f"Long size must be \$500k - \$2M. Projected: \${proj_size:,.0f}"
                else: # Short
                    if not (300000 <= proj_size <= 1200000):
                        valid = False; err_msg = f"Short size must be \$300k - \$1.2M. Projected: \${proj_size:,.0f}"
            
            # Check 3: 1 Month Lockup (For Closing)
            if action in ['SELL', 'BUY_TO_COVER']:
                if final_tik in state['positions']:
                    entry_dt = state['positions'][final_tik]['first_entry']
                    # Use current date or backdated date for check
                    check_date = datetime.combine(t_date, datetime.min.time()) if test_mode else datetime.now()
                    
                    if entry_dt and (check_date - entry_dt).days < 30:
                        valid = False; err_msg = f"Position held < 30 days (Entry: {entry_dt.date()}). Cannot unwind yet."
                else:
                    valid = False; err_msg = "No position to close."

            if valid:
                # Execution Logic
                if test_mode:
                    h_date = datetime.combine(t_date, datetime.min.time())
                    price = get_historical_price(final_tik, h_date)
                    if price > 0:
                        qty = amt / price
                        t = Transaction(user_id=user.id, ticker=final_tik, trans_type=action,
                                        status='FILLED', price=price, quantity=qty, amount=amt, date=h_date, notes=notes)
                        session.add(t)
                        session.commit()
                        st.success(f"Executed at ${price:.2f}")
                        time.sleep(1)
                        st.rerun()
                    else: st.error("Price not found")
                else:
                    # Live Pending
                    t = Transaction(user_id=user.id, ticker=final_tik, trans_type=action,
                                    status='PENDING', amount=amt, notes=notes)
                    session.add(t)
                    session.commit()
                    st.success("Order Pending Next Open")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error(f"Compliance Error: {err_msg}")

def pm_page(user):
    st.title("PM Overview")
    analysts = session.query(User).filter_by(role='analyst').all()
    
    # Master Summary
    summary = []
    for a in analysts:
        s, _ = calculate_portfolio_state(a.id)
        pnl = s['equity'] - a.initial_capital
        summary.append({
            "Analyst": a.username,
            "Equity": s['equity'],
            "Cash": s['cash'],
            "YTD PnL": pnl,
            "Compliance": "⚠️ High Cash" if s['cash'] > 1500000 else "✅ OK"
        })
    
    st.dataframe(pd.DataFrame(summary).style.format({"Equity":"${:,.0f}", "Cash":"${:,.0f}", "YTD PnL":"${:,.0f}"}), use_container_width=True)
    
    st.divider()
    sel = st.selectbox("Select Analyst", [a.username for a in analysts])
    if sel:
        target = session.query(User).filter_by(username=sel).first()
        analyst_page(target) # Reuse the analyst view

def admin_page():
    st.title("Admin")
    # Toggle Test Mode
    curr = is_test_mode()
    if st.toggle("Test Mode", value=curr) != curr:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        cfg.value = str(not curr); session.commit(); st.rerun()
    
    # User Mgmt
    with st.form("new_user"):
        u = st.text_input("User"); p = st.text_input("Pass", type="password"); r = st.selectbox("Role", ["analyst", "pm"])
        if st.form_submit_button("Create"):
            session.add(User(username=u, password_hash=hash_password(p), role=r)); session.commit(); st.rerun()
    
    users = session.query(User).all()
    to_del = st.selectbox("Delete", [u.username for u in users if u.username!='admin'])
    if st.button("Delete"):
        session.delete(session.query(User).filter_by(username=to_del).first()); session.commit(); st.rerun()

# --- HELPERS ---
def format_ticker(symbol, market):
    symbol = symbol.strip().upper()
    suffixes = {"US": "", "Hong Kong": ".HK", "China (Shanghai)": ".SS", "China (Shenzhen)": ".SZ", "Japan": ".T", "UK": ".L", "France": ".PA"}
    return f"{symbol}{suffixes.get(market, '')}"
def is_test_mode():
    try: cfg = session.query(SystemConfig).filter_by(key='test_mode').first(); return cfg.value == 'True' if cfg else False
    except: return False
def hash_password(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
def init_db():
    try:
        if not session.query(User).filter_by(username='admin').first():
            session.add(User(username='admin', password_hash=hash_password('8848'), role='admin'))
            session.commit()
        if not session.query(SystemConfig).filter_by(key='test_mode').first():
            session.add(SystemConfig(key='test_mode', value='False'))
            session.commit()
    except: pass

def main():
    st.set_page_config(layout="wide", page_title="AlphaTracker Pro")
    init_db()
    if 'user_id' not in st.session_state: st.session_state.user_id = None
    
    if not st.session_state.user_id:
        st.title("Login"); 
        u = st.text_input("User"); p = st.text_input("Pass", type="password")
        if st.button("Login"):
            user = session.query(User).filter_by(username=u).first()
            if user and bcrypt.checkpw(p.encode(), user.password_hash.encode()):
                st.session_state.user_id = user.id; st.session_state.role = user.role; st.rerun()
    else:
        user = session.query(User).filter_by(id=st.session_state.user_id).first()
        with st.sidebar:
            if st.button("Logout"): st.session_state.user_id = None; st.rerun()
        if user.role=='admin': admin_page()
        elif user.role=='analyst': analyst_page(user)
        elif user.role=='pm': pm_page(user)

if __name__ == "__main__": main()
