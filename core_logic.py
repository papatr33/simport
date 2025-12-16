import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Re-use your Market Config
MARKET_CONFIG = {
    "US": {"suffix": "", "fx": None, "currency": "USD"},
    "Hong Kong": {"suffix": ".HK", "fx": "HKD=X", "currency": "HKD"},
    "China (Shanghai)": {"suffix": ".SS", "fx": "CNY=X", "currency": "CNY"},
    "China (Shenzhen)": {"suffix": ".SZ", "fx": "CNY=X", "currency": "CNY"},
    "Japan": {"suffix": ".T", "fx": "JPY=X", "currency": "JPY"},
    "UK": {"suffix": ".L", "fx": "GBP=X", "currency": "GBP"}, 
    "France": {"suffix": ".PA", "fx": "EUR=X", "currency": "EUR"},
    "Netherlands": {"suffix": ".AS", "fx": "EUR=X", "currency": "EUR"}
}

def extract_scalar(val):
    try:
        if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, list)):
            val = val.values.flatten()[0] if hasattr(val, 'values') else val[0]
        return float(val)
    except: return 0.0

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

def calculate_portfolio_state(txs_data, initial_capital, evaluation_date=None):
    """
    Calculates portfolio state. 
    If evaluation_date is provided, ignores transactions after that date.
    """
    if evaluation_date is None:
        evaluation_date = datetime.now()

    # Filter transactions based on evaluation date
    txs_data = [t for t in txs_data if t['date'] <= evaluation_date]

    state = {
        "cash": initial_capital,
        "positions": {}, 
        "realized_pnl_by_side": {}, 
        "equity": 0.0
    }

    # 1. Ledger Replay
    for t in txs_data:
        tik = t['ticker']
        if tik not in state["positions"]:
            state["positions"][tik] = {
                "qty": 0.0, "avg_cost": 0.0, "avg_cost_local": 0.0,
                "type": "FLAT", "market": t['market'], "first_entry": None
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
            if pos["qty"] <= 0.001: del state["positions"][tik]

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
            if abs(pos["qty"]) <= 0.001: del state["positions"][tik]

    # 2. Mark to Market (Live Prices)
    active_tickers = list(state["positions"].keys())
    active_markets = {tik: pos['market'] for tik, pos in state["positions"].items()}
    fx_needed = [MARKET_CONFIG[m]['fx'] for m in active_markets.values() if m and MARKET_CONFIG[m]['fx']]
    
    # We fetch data up to the evaluation date
    batch_data = fetch_batch_data(active_tickers + fx_needed, evaluation_date)
    
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

def get_ytd_performance(txs_data, initial_capital):
    # (Exact copy of the logic from app.py, just removing the wrapper)
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1)
    end_date = datetime.now()
    
    if not txs_data:
        dates = pd.date_range(start_date, end_date, freq='B')
        df = pd.DataFrame({'Date': dates, 'Equity': initial_capital})
        df['Return %'] = 0.0
        df['Long PnL'] = 0.0
        df['Short PnL'] = 0.0
        return df, pd.Series()

    ticker_market_map = {}
    for t in txs_data: ticker_market_map[t['ticker']] = t['market']
    tickers = list(ticker_market_map.keys())
    
    fx_tickers = set()
    for m in ticker_market_map.values():
        if m and MARKET_CONFIG.get(m, {}).get('fx'):
            fx_tickers.add(MARKET_CONFIG[m]['fx'])
    all_tickers = tickers + list(fx_tickers)
    
    first_tx_date = min(t['date'] for t in txs_data)
    fetch_start = min(start_date, first_tx_date) - timedelta(days=5)
    batch_data = fetch_batch_data(all_tickers, fetch_start)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    curve = []
    
    if not batch_data.empty:
        batch_data.index = pd.to_datetime(batch_data.index).normalize()
    
    curr_cash = initial_capital
    holdings = {} 
    
    tx_idx = 0
    n_txs = len(txs_data)
    
    # 1. PRE-ROLL
    while tx_idx < n_txs and txs_data[tx_idx]['date'] < start_date:
        t = txs_data[tx_idx]
        if t['trans_type'] == 'BUY':
            curr_cash -= t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']
        elif t['trans_type'] == 'SELL':
            curr_cash += t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']
        elif t['trans_type'] == 'SHORT_SELL':
            curr_cash += t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']
        elif t['trans_type'] == 'BUY_TO_COVER':
            curr_cash -= t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']
        tx_idx += 1

    # 2. DAILY LOOP
    for d in dates:
        d_norm = d.normalize()
        daily_long_flow, daily_short_flow = 0.0, 0.0
        
        while tx_idx < n_txs and txs_data[tx_idx]['date'].date() <= d_norm.date():
            t = txs_data[tx_idx]
            if t['trans_type'] == 'BUY':
                curr_cash -= t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']; daily_long_flow += t['amount']
            elif t['trans_type'] == 'SELL':
                curr_cash += t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']; daily_long_flow -= t['amount']
            elif t['trans_type'] == 'SHORT_SELL':
                curr_cash += t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) - t['quantity']; daily_short_flow -= t['amount']
            elif t['trans_type'] == 'BUY_TO_COVER':
                curr_cash -= t['amount']; holdings[t['ticker']] = holdings.get(t['ticker'], 0) + t['quantity']; daily_short_flow += t['amount']
            tx_idx += 1
            
        long_mv, short_mv = 0.0, 0.0 
        
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
                            p_local = float(row[tik]) if tik in row else 0.0
                            mkt = ticker_market_map.get(tik, 'US')
                            fx_sym = MARKET_CONFIG.get(mkt, {}).get('fx')
                            p_usd = p_local
                            if fx_sym and fx_sym in row:
                                rate = float(row[fx_sym])
                                if mkt == "UK": p_local /= 100.0
                                if rate > 0: p_usd = p_local / rate
                            val = qty * p_usd
                            if qty > 0: long_mv += val
                            else: short_mv += abs(val) 
            except: pass
        
        equity = curr_cash + long_mv - short_mv
        curve.append({
            "Date": d, "Equity": equity, "LongMV": long_mv, "ShortMV": short_mv,
            "LongFlow": daily_long_flow, "ShortFlow": daily_short_flow
        })

    df_curve = pd.DataFrame(curve)
    if df_curve.empty: return df_curve, pd.Series()

    df_curve['Long PnL'] = (df_curve['LongMV'].diff() - df_curve['LongFlow']).fillna(0)
    df_curve['Short PnL'] = (-(df_curve['ShortMV'].diff()) - df_curve['ShortFlow']).fillna(0)
    df_curve['Return %'] = ((df_curve['Equity'] / initial_capital) - 1) * 100
    
    return df_curve, pd.Series()
