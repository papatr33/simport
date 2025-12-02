import streamlit as st
import pandas as pd
import yfinance as yf
import bcrypt
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import time

# ==========================================
# 1. DATABASE CONFIGURATION (CLOUD AWARE)
# ==========================================
Base = declarative_base()

# --- Define Models ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False) # 'admin', 'pm', 'analyst'
    email = Column(String, nullable=True)
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False) # 'Long' or 'Short'
    entry_date = Column(DateTime, default=datetime.now)
    status = Column(String, default='PENDING') # 'PENDING', 'OPEN', 'CLOSED'
    cost_basis = Column(Float, nullable=True)
    notes = Column(String, nullable=True)
    user = relationship("User", back_populates="trades")

class SystemConfig(Base):
    __tablename__ = 'system_config'
    key = Column(String, primary_key=True)
    value = Column(String) 

# --- Connect to DB ---
def get_db_engine():
    # 1. Try Streamlit Cloud Secrets
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    # 2. Try Environment Variable (Local or GitHub Actions)
    elif "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    # 3. Fallback to Local SQLite (For testing without Supabase)
    else:
        db_url = 'sqlite:///portfolio.db'

    # Fix for Supabase/SQLAlchemy protocol mismatch
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # specific args for SQLite vs Postgres
    if 'sqlite' in db_url:
        return create_engine(db_url, connect_args={'check_same_thread': False})
    else:
        return create_engine(db_url)

engine = get_db_engine()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def init_db():
    """Creates default admin and config if empty."""
    try:
        # Create Admin if table is empty
        if not session.query(User).filter_by(username='admin').first():
            pw_hash = hash_password('8848')
            new_admin = User(username='admin', password_hash=pw_hash, role='admin', email='admin@fund.com')
            session.add(new_admin)
            session.commit()
            
        # Create Default Config
        if not session.query(SystemConfig).filter_by(key='test_mode').first():
            session.add(SystemConfig(key='test_mode', value='False'))
            session.commit()
            
    except Exception as e:
        st.error(f"DB Init Error: {e}")

def is_test_mode():
    cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
    return cfg.value == 'True' if cfg else False

def get_live_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0.0
    except:
        return 0.0

def format_ticker(symbol, market):
    symbol = symbol.strip().upper()
    suffixes = {
        "US": "",
        "Hong Kong": ".HK",
        "China (Shanghai)": ".SS",
        "China (Shenzhen)": ".SZ",
        "Japan": ".T",
        "UK": ".L",
        "France": ".PA"
    }
    return f"{symbol}{suffixes.get(market, '')}"

# ==========================================
# 3. PAGE VIEWS
# ==========================================

def admin_page():
    st.header("Admin Dashboard 🛠️")
    
    # Settings
    st.subheader("⚙️ System Settings")
    current_mode = is_test_mode()
    test_mode_toggle = st.toggle("Enable Test Mode", value=current_mode, help="Trades fill immediately at last close price.")
    
    if test_mode_toggle != current_mode:
        cfg = session.query(SystemConfig).filter_by(key='test_mode').first()
        cfg.value = str(test_mode_toggle)
        session.commit()
        st.success(f"Test Mode set to: {test_mode_toggle}")
        time.sleep(1)
        st.rerun()

    st.divider()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Create User")
        with st.form("create_user"):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            new_email = st.text_input("Email Address")
            new_role = st.selectbox("Role", ["analyst", "pm"])
            submitted = st.form_submit_button("Create User")
            
            if submitted:
                if not new_user or not new_pass:
                    st.error("Username and Password are required.")
                elif session.query(User).filter_by(username=new_user).first():
                    st.error("User already exists.")
                else:
                    u = User(username=new_user, password_hash=hash_password(new_pass), role=new_role, email=new_email)
                    session.add(u)
                    session.commit()
                    st.success(f"User {new_user} created successfully!")
                    time.sleep(1)
                    st.rerun()

    with col2:
        st.subheader("Manage Users")
        users = session.query(User).filter(User.username != 'admin').all()
        
        if users:
            user_df = pd.DataFrame([{"ID": u.id, "Username": u.username, "Role": u.role, "Email": u.email} for u in users])
            st.dataframe(user_df, use_container_width=True, hide_index=True)
            
            c_del_1, c_del_2 = st.columns([3, 1])
            with c_del_1:
                user_to_delete = st.selectbox("Select User to Delete", [u.username for u in users], key="del_select")
            with c_del_2:
                st.write("") 
                st.write("") 
                if st.button("Delete User", type="primary"):
                    u_del = session.query(User).filter_by(username=user_to_delete).first()
                    session.delete(u_del)
                    session.commit()
                    st.warning(f"User {user_to_delete} deleted.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No other users found.")

def analyst_page(user):
    st.header(f"Analyst Dashboard: {user.username} 📈")
    
    if is_test_mode():
        st.warning("⚠️ TEST MODE ACTIVE: Trades will fill immediately.")

    # Data
    my_trades = session.query(Trade).filter_by(user_id=user.id).all()
    active_trades = [t for t in my_trades if t.status == 'OPEN']
    pending_trades = [t for t in my_trades if t.status == 'PENDING']
    
    longs_count = len([t for t in my_trades if t.direction == 'Long' and t.status != 'CLOSED'])
    shorts_count = len([t for t in my_trades if t.direction == 'Short' and t.status != 'CLOSED'])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Long Positions", f"{longs_count} / 5")
    c2.metric("Short Positions", f"{shorts_count} / 3")
    c3.metric("Pending", len(pending_trades))
    
    st.divider()

    # Entry
    st.subheader("Enter New Trade")
    with st.form("trade_form"):
        c_mkt, c_tick, c_dir = st.columns([1, 1, 1])
        with c_mkt:
            market = st.selectbox("Market", ["US", "Hong Kong", "China (Shanghai)", "China (Shenzhen)", "Japan", "UK", "France"])
        with c_tick:
            raw_ticker = st.text_input("Ticker (e.g., AAPL, 700, MC)").strip()
        with c_dir:
            direction = st.selectbox("Direction", ["Long", "Short"])
        
        notes = st.text_area("Trade Notes / Thesis")
        submit = st.form_submit_button("Submit Order")
        
        if submit:
            if not raw_ticker:
                st.error("Please enter a ticker.")
            else:
                if direction == "Long" and longs_count >= 5:
                    st.error("🚫 Max Long positions (5) reached.")
                elif direction == "Short" and shorts_count >= 3:
                    st.error("🚫 Max Short positions (3) reached.")
                else:
                    final_ticker = format_ticker(raw_ticker, market)
                    
                    if is_test_mode():
                        fill_price = get_live_price(final_ticker)
                        if fill_price > 0:
                            new_trade = Trade(
                                user_id=user.id, ticker=final_ticker, direction=direction,
                                status='OPEN', cost_basis=fill_price, notes=f"[TEST FILL] {notes}"
                            )
                            msg = f"✅ TEST MODE: Filled {final_ticker} immediately at ${fill_price:.2f}"
                        else:
                            st.error(f"Could not fetch price for {final_ticker}.")
                            return
                    else:
                        new_trade = Trade(
                            user_id=user.id, ticker=final_ticker, direction=direction,
                            status='PENDING', cost_basis=0.0, notes=notes
                        )
                        msg = f"✅ Order for {final_ticker} submitted! Pending next open."

                    session.add(new_trade)
                    session.commit()
                    st.success(msg)
                    time.sleep(1.5)
                    st.rerun()

    st.divider()

    # Active
    st.subheader("Active Portfolio")
    if active_trades:
        data = []
        for t in active_trades:
            curr_price = get_live_price(t.ticker)
            if t.cost_basis > 0:
                if t.direction == 'Long':
                    pnl_pct = ((curr_price - t.cost_basis) / t.cost_basis) * 100
                else:
                    pnl_pct = ((t.cost_basis - curr_price) / t.cost_basis) * 100
            else:
                pnl_pct = 0.0

            data.append({
                "Ticker": t.ticker,
                "Side": t.direction,
                "Fill Date": t.entry_date.strftime("%Y-%m-%d"),
                "Cost": f"${t.cost_basis:.2f}",
                "Current": f"${curr_price:.2f}",
                "Unrealized PnL": f"{pnl_pct:.2f}%",
                "Notes": t.notes
            })
        st.table(pd.DataFrame(data))
    else:
        st.info("No active positions.")

    # Pending
    if pending_trades:
        st.subheader("⏳ Pending Orders")
        p_data = [{"Ticker": t.ticker, "Side": t.direction, "Submitted": t.entry_date.strftime("%Y-%m-%d %H:%M"), "Notes": t.notes} for t in pending_trades]
        st.dataframe(pd.DataFrame(p_data))

def pm_page(user):
    st.header("Portfolio Manager Overview 🏦")
    analysts = session.query(User).filter_by(role='analyst').all()
    selected_analyst_name = st.selectbox("Filter by Analyst", ["All Analysts"] + [u.username for u in analysts])
    
    query = session.query(Trade).join(User).filter(Trade.status == 'OPEN')
    if selected_analyst_name != "All Analysts":
        query = query.filter(User.username == selected_analyst_name)
    
    trades = query.all()
    if not trades:
        st.warning("No active trades found.")
        return

    rows = []
    progress_bar = st.progress(0)
    for idx, t in enumerate(trades):
        current_price = get_live_price(t.ticker)
        if t.cost_basis and t.cost_basis > 0:
            if t.direction == 'Long':
                pnl = (current_price - t.cost_basis) / t.cost_basis
            else:
                pnl = (t.cost_basis - current_price) / t.cost_basis
        else:
            pnl = 0.0
            
        rows.append({
            "Analyst": t.user.username,
            "Ticker": t.ticker,
            "Side": t.direction,
            "Cost": t.cost_basis,
            "Current": current_price,
            "PnL %": pnl * 100,
            "Notes": t.notes
        })
        progress_bar.progress((idx + 1) / len(trades))
    progress_bar.empty()
    
    if rows:
        df = pd.DataFrame(rows)
        def color_pnl(val):
            color = 'red' if val < 0 else 'green'
            if val <= -20: color = 'darkred'
            return f'color: {color}'

        st.dataframe(df.style.format({"Cost": "${:.2f}", "Current": "${:.2f}", "PnL %": "{:.2f}%"}).map(color_pnl, subset=['PnL %']), use_container_width=True)

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
def main():
    st.set_page_config(page_title="AlphaTracker", layout="wide")
    init_db()

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        st.session_state.role = None
        st.session_state.username = None

    if st.session_state.user_id is None:
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            st.title("AlphaTracker 🔒")
            with st.form("login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    user = session.query(User).filter_by(username=username).first()
                    if user and check_password(password, user.password_hash):
                        st.session_state.user_id = user.id
                        st.session_state.role = user.role
                        st.session_state.username = user.username
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
    else:
        with st.sidebar:
            st.write(f"👤 **{st.session_state.username}** ({st.session_state.role.upper()})")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.role = None
                st.session_state.username = None
                st.rerun()

        current_user = session.query(User).filter_by(id=st.session_state.user_id).first()
        # Handle case where user was deleted while logged in
        if not current_user:
            st.session_state.user_id = None
            st.rerun()

        if current_user.role == 'admin':
            admin_page()
        elif current_user.role == 'analyst':
            analyst_page(current_user)
        elif current_user.role == 'pm':
            pm_page(current_user)

if __name__ == "__main__":
    main()