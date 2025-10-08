# ==============================
# SilverFoxFlow — Market UOA 2.0
# ==============================
# - Live scan from Unusual Whales API (via Secrets)
# - Endpoint auto-discovery (no guessing paths)
# - Per-ticker UOA 2.0 aggregation
# - Ranked picks + simple underlying backtest

import time
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# -----------------------------
# Page / Layout
# -----------------------------
st.set_page_config(page_title="SilverFoxFlow — Market UOA Scanner", page_icon="🦊", layout="wide")
st.title("🦊 SilverFoxFlow — Market UOA Scanner")
st.caption("We scan institutional options flow across the market and surface the strongest, cleanest signals.")

# -----------------------------
# Presets (UOA 2.0 profiles)
# -----------------------------
PRESETS = {
    "Strict": dict(MIN_PREM=3_000_000, MIN_PRINTS=8, EXP_MIN_W=1, EXP_MAX_W=8,
                   REQ_VOL_GT_OI=True, DOMINANCE=1.40, MIN_TOTAL_PREM=5_000_000,
                   MIN_AVG_AGGR_CALL=0.60, MAX_AVG_AGGR_PUT=0.40),
    "Balanced": dict(MIN_PREM=2_000_000, MIN_PRINTS=6, EXP_MIN_W=1, EXP_MAX_W=10,
                     REQ_VOL_GT_OI=False, DOMINANCE=1.30, MIN_TOTAL_PREM=3_000_000,
                     MIN_AVG_AGGR_CALL=0.58, MAX_AVG_AGGR_PUT=0.42),
    "Explorer": dict(MIN_PREM=1_000_000, MIN_PRINTS=4, EXP_MIN_W=1, EXP_MAX_W=16,
                     REQ_VOL_GT_OI=False, DOMINANCE=1.20, MIN_TOTAL_PREM=2_000_000,
                     MIN_AVG_AGGR_CALL=0.55, MAX_AVG_AGGR_PUT=0.45),
}
REQUIRED_COLS = [
    "timestamp","ticker","expiry_weeks","notional","prints","aggr_ratio","volume","open_interest"
]

# -----------------------------
# Controls
# -----------------------------
left, right = st.columns([3,2], gap="large")
with left:
    mode = st.radio("Scan profile", list(PRESETS.keys()), index=1, horizontal=True)
cfg = PRESETS[mode]
with right:
    window_min = st.select_slider("Lookback window (minutes)", options=[15,30,45,60,90,120], value=60)
    auto = st.toggle("Auto-refresh every 60s", value=False)

# -----------------------------
# Secrets / Provider
# -----------------------------
UOA = st.secrets.get("uoa", {})
PROVIDER = UOA.get("provider")
API_URL  = UOA.get("api_url")       # base host only, e.g. https://api.unusualwhales.com
API_KEY  = UOA.get("api_key")

status = "Demo Mode"
if PROVIDER and API_URL and API_KEY:
    status = f"Live: {PROVIDER.upper()}"
st.caption(f"Data source: **{status}** • Window: **{window_min}m** • Profile: **{mode}**")
if "uw_endpoint_used" in st.session_state:
    st.caption(f"UW endpoint: {st.session_state['uw_endpoint_used']}")

# -----------------------------
# Helpers (stable recency)
# -----------------------------
def _stable_minutes_old(ts: pd.Series) -> pd.Series:
    as_of = pd.to_datetime(ts).max()
    mins = (as_of - pd.to_datetime(ts)).dt.total_seconds() / 60.0
    return mins.clip(lower=0).fillna(60)

def _recency_weight(minutes_old: pd.Series) -> pd.Series:
    return pd.Series([1.0 if m < 10 else 0.7 if m < 30 else 0.5 for m in minutes_old], index=minutes_old.index)

# -----------------------------
# Data fetchers
# -----------------------------
def _demo_flow(n: int = 400, max_window: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.utcnow()
    tickers = ["AAPL","MSFT","NVDA","META","TSLA","AMZN","AMD","NFLX","NOC","MRVL","CAT","MCD",
               "DDOG","SHOP","SMCI","PANW","ORCL","GE","AVGO","LMT","BA","UNH","CVS","LLY","MDB"]
    ts = [now - timedelta(minutes=int(rng.integers(0, max_window))) for _ in range(n)]
    df = pd.DataFrame({
        "timestamp": [t.isoformat(sep=" ") for t in ts],
        "ticker": rng.choice(tickers, size=n),
        "expiry_weeks": rng.integers(1, 13, size=n),
        "notional": rng.choice([1.2e6, 2.5e6, 3e6, 4.5e6, 6e6, 8e6, 10e6], size=n),
        "prints": rng.integers(3, 21, size=n),
        "aggr_ratio": np.round(rng.uniform(0.3, 0.8, size=n), 2),
        "volume": rng.integers(1500, 24000, size=n),
        "open_interest": rng.integers(1200, 20000, size=n),
    })
    return df

def fetch_flow_generic(api_url: str, api_key: str, window_min: int) -> pd.DataFrame:
    """
    Unusual Whales adapter with endpoint auto-discovery.
    Provide base host in secrets (e.g., https://api.unusualwhales.com).
    Tries multiple likely paths until one returns 200.
    """
    candidate_paths = [
        "/option-trades/flow-alerts",
        "/option-trades/flow",
        "/options/flow-alerts",
        "/options/flow",
        "/flow-alerts",
        "/flow",
    ]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params  = {"minutes": window_min, "window_min": window_min}  # prefer 'minutes'
    base = api_url.rstrip("/")

    last_err = None
    for path in candidate_paths:
        url = f"{base}{path}"
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 404:
                last_err = f"404 at {url}"
                continue
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", payload) if isinstance(payload, dict) else payload

            def first(d, *keys, default=None):
                for k in keys:
                    if k in d and d[k] is not None:
                        return d[k]
                return default

            rows = []
            for x in data:
                ts   = first(x, "ts","timestamp","time","created_at")
                sym  = first(x, "symbol","sym","ticker","underlying","underlying_symbol")
                dte  = first(x, "dte","days_to_expiry","days_til_expiration")
                exp  = first(x, "expiry","expiration","exp_date","exp")
                wks  = first(x, "expiry_weeks","weeks_to_expiry","w","weeks")
                if wks is None:
                    if dte is not None:
                        try: wks = float(dte)/7.0
                        except: wks = None
                    elif exp:
                        try: wks = max(0.0, (pd.to_datetime(exp) - pd.Timestamp.utcnow()).days/7.0)
                        except: wks = None

                prem = first(x, "premium","prem","notional","usd_value","dollar_value")
                prts = first(x, "prints","nprints","count","num_trades","sweeps","blocks")
                aggr = first(x, "aggressor_ratio","aggr_ratio","aggr","ask_hit_ratio","at_ask_ratio")
                vol  = first(x, "volume","vol","contracts_traded","contracts")
                oi   = first(x, "open_interest","open_int","oi")

                rows.append({
                    "timestamp": ts,
                    "ticker": sym,
                    "expiry_weeks": wks,
                    "notional": prem,
                    "prints": prts,
                    "aggr_ratio": aggr,
                    "volume": vol,
                    "open_interest": oi,
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                for c in ["expiry_weeks","notional","prints","aggr_ratio","volume","open_interest"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            st.session_state["uw_endpoint_used"] = url
            return df

        except requests.HTTPError as e:
            last_err = f"{e.response.status_code} at {url}"
            if e.response is not None and e.response.status_code in (401, 403):
                raise  # auth/plan issue — bubble up
            continue
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(f"Could not find a working UW endpoint. Last error: {last_err}")

def get_market_flow(window_min: int) -> pd.DataFrame:
    try:
        if PROVIDER and API_URL and API_KEY:
            df = fetch_flow_generic(API_URL, API_KEY, window_min)
        else:
            df = _demo_flow(max_window=window_min)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        df = _demo_flow(max_window=window_min)

    df = df.copy()
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df[REQUIRED_COLS].dropna(subset=["timestamp","ticker"])

# -----------------------------
# UOA 2.0 logic
# -----------------------------
def eligible_mask(df: pd.DataFrame) -> pd.Series:
    near = df["expiry_weeks"].fillna(0).between(cfg["EXP_MIN_W"], cfg["EXP_MAX_W"])
    prem = df["notional"].fillna(0) >= cfg["MIN_PREM"]
    prints = df["prints"].fillna(0) >= cfg["MIN_PRINTS"]
    aggr = df["aggr_ratio"].fillna(0).between(0,1)
    if cfg["REQ_VOL_GT_OI"]:
        v_ok = df["volume"].fillna(0) > df["open_interest"].fillna(0)
    else:
        v_ok = True
    return near & prem & prints & aggr & v_ok

def summarize_ticker(g: pd.DataFrame) -> Dict[str, Any]:
    if g.empty:
        return dict(decision="NO TRADE", reason="No qualifying prints.", total_premium=0, bull=0, bear=0,
                    dominance=1.0, avg_aggr=0.0, prints=0)
    g = g.copy()
    g["minutes_old"] = _stable_minutes_old(g["timestamp"])
    g["w"] = _recency_weight(g["minutes_old"])
    g["bull_$"] = g["notional"] * g["aggr_ratio"] * g["w"]
    g["bear_$"] = g["notional"] * (1 - g["aggr_ratio"]) * g["w"]

    bull = float(g["bull_$"].sum())
    bear = float(g["bear_$"].sum())
    total = float(g["notional"].sum())
    prints = int(g["prints"].sum())
    avg_aggr = float(g["aggr_ratio"].mean())
    dominance = (bull / max(bear, 1e-9)) if bear > 0 else float("inf")

    if total < cfg["MIN_TOTAL_PREM"] or prints < cfg["MIN_PRINTS"]:
        return dict(decision="NO TRADE", reason="Insufficient premium/prints.", total_premium=total,
                    bull=bull, bear=bear, dominance=dominance, avg_aggr=avg_aggr, prints=prints)
    if dominance >= cfg["DOMINANCE"] and avg_aggr >= cfg["MIN_AVG_AGGR_CALL"]:
        return dict(decision="BUY CALLS", reason="Bullish dollars dominate; buyers aggressive.",
                    total_premium=total, bull=bull, bear=bear, dominance=dominance, avg_aggr=avg_aggr, prints=prints)
    if dominance <= 1.0/cfg["DOMINANCE"] and avg_aggr <= cfg["MAX_AVG_AGGR_PUT"]:
        return dict(decision="BUY PUTS", reason="Bearish dollars dominate; sellers aggressive.",
                    total_premium=total, bull=bull, bear=bear, dominance=dominance, avg_aggr=avg_aggr, prints=prints)
    return dict(decision="NO TRADE", reason="Mixed/indecisive.", total_premium=total,
                bull=bull, bear=bear, dominance=dominance, avg_aggr=avg_aggr, prints=prints)

# -----------------------------
# Backtest helpers (underlying)
# -----------------------------
BT_HOLD_DAYS_DEFAULT = 5
BT_TP_DEFAULT = 0.12   # +12%
BT_SL_DEFAULT = -0.07  # -7%

def fetch_prices_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower).reset_index().rename(columns={"index":"date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date","open","high","low","close","volume"]]

def build_daily_signals(edf: pd.DataFrame) -> pd.DataFrame:
    edf = edf.copy()
    edf["date"] = edf["timestamp"].dt.tz_localize(None).dt.date
    rows = []
    for (tkr, d), g in edf.groupby(["ticker","date"]):
        s = summarize_ticker(g)
        rows.append(dict(date=pd.to_datetime(d), ticker=tkr, decision=s["decision"],
                         total_premium=s["total_premium"], prints=s["prints"]))
    sig = pd.DataFrame(rows)
    if sig.empty:
        return sig
    sig = (sig.sort_values(["ticker","date","total_premium"], ascending=[True,True,False])
             .drop_duplicates(subset=["ticker","date"], keep="first"))
    return sig

def simulate_trade_path(prices: pd.DataFrame, entry_idx: int, side: str,
                        hold_days: int, sl: float, tp: float) -> float:
    if entry_idx+1 >= len(prices):
        return 0.0
    entry_px = float(prices.iloc[entry_idx+1]["open"])
    if entry_px <= 0:
        return 0.0
    end_idx = min(entry_idx+1+hold_days, len(prices)-1)
    direction = 1 if side == "BUY CALLS" else -1
    for i in range(entry_idx+1, end_idx+1):
        hi = float(prices.iloc[i]["high"]); lo = float(prices.iloc[i]["low"])
        r_tp = direction * ((hi - entry_px)/entry_px)
        r_sl = direction * ((lo - entry_px)/entry_px)
        if r_tp >= tp: return tp
        if r_sl <= sl: return sl
    exit_px = float(prices.iloc[end_idx]["close"])
    return direction * ((exit_px - entry_px)/entry_px)

def backtest_signals(sig: pd.DataFrame, start: str, end: str,
                     hold_days: int, sl: float, tp: float) -> pd.DataFrame:
    trades = []
    cache: Dict[str, pd.DataFrame] = {}
    for tkr, g in sig.groupby("ticker"):
        if tkr not in cache:
            cache[tkr] = fetch_prices_yf(tkr, start, end)
        px = cache[tkr]
        if px.empty: continue
        px = px.sort_values("date").reset_index(drop=True)
        px["d"] = px["date"].dt.date
        idx_by_date = {d:i for i,d in enumerate(px["d"].tolist())}
        for _, row in g.sort_values("date").iterrows():
            d = row["date"].date()
            if d not in idx_by_date: continue
            entry_idx = idx_by_date[d]
            ret = simulate_trade_path(px, entry_idx, row["decision"], hold_days, sl, tp)
            trades.append(dict(date=row["date"], ticker=tkr, decision=row["decision"], ret=ret))
    return pd.DataFrame(trades)

# -----------------------------
# Scan trigger
# -----------------------------
scan_col, _ = st.columns([1,6])
with scan_col:
    run_scan = st.button("🔎 Scan Market", type="primary")

if auto and int(time.time()) % 60 == 0:
    run_scan = True

# -----------------------------
# Run scan
# -----------------------------
if run_scan:
    flow = get_market_flow(window_min)
    msk = eligible_mask(flow)
    edf = flow.loc[msk].copy()

    results = []
    for tkr, g in (edf.groupby("ticker") if not edf.empty else []):
        s = summarize_ticker(g)
        results.append(dict(
            ticker=tkr, decision=s["decision"], reason=s["reason"],
            total_premium=round(s["total_premium"],2),
            bull_dollars=round(s["bull"],2), bear_dollars=round(s["bear"],2),
            dominance=("∞" if s["dominance"]==float("inf") else round(s["dominance"],2)),
            avg_aggr=round(s["avg_aggr"],3), prints=s["prints"]
        ))

    res = (pd.DataFrame(results)
           .sort_values(["decision","total_premium"], ascending=[True, False])
           if results else pd.DataFrame(columns=[
               "ticker","decision","reason","total_premium","bull_dollars","bear_dollars",
               "dominance","avg_aggr","prints"
           ]))

    st.session_state["scan_ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.session_state["res"] = res
    st.session_state["eligible_rows"] = int(len(edf))
    st.session_state["unique_tickers"] = int(edf["ticker"].nunique()) if not edf.empty else 0
    st.session_state["flow_for_bt"] = edf

# -----------------------------
# Render results
# -----------------------------
if "res" in st.session_state:
    res = st.session_state["res"]
    st.markdown(f"**Last scan:** {st.session_state['scan_ts']} • Mode **{mode}** • Window **{window_min}m**")

    st.subheader("Top Picks")
    if res.empty:
        st.warning("No qualifying tickers. Try a wider window or Explorer mode.")
    else:
        top = res.sort_values("total_premium", ascending=False).head(10)
        cols = st.columns(min(5, len(top)))
        for i, (_, r) in enumerate(top.iterrows()):
            with cols[i % len(cols)]:
                st.metric(f"{r['ticker']}", r["decision"], help=r["reason"])

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligible prints", f"{st.session_state.get('eligible_rows',0):,}")
    c2.metric("Unique tickers", f"{st.session_state.get('unique_tickers',0):,}")
    c3.metric("BUY CALLS", int((res["decision"]=="BUY CALLS").sum()))
    c4.metric("BUY PUTS", int((res["decision"]=="BUY PUTS").sum()))

    st.subheader("All Signals")
    st.caption("Sorted by total premium — highest conviction first.")
    st.dataframe(res.reset_index(drop=True), use_container_width=True, hide_index=True)

    # -------------------------
    # Backtest UI
    # -------------------------
    st.divider()
    st.subheader("🧪 Backtest (underlying returns)")
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_days = st.slider("Hold days", 2, 15, value=BT_HOLD_DAYS_DEFAULT)
    with col2:
        bt_tp = st.slider("Take-profit (%)", 3, 30, value=int(abs(BT_TP_DEFAULT)*100)) / 100.0
    with col3:
        bt_sl = - st.slider("Stop-loss (%)", 2, 20, value=int(abs(BT_SL_DEFAULT)*100)) / 100.0

    run_bt = st.button("▶️ Run Backtest", type="secondary")

    if run_bt:
        edf = st.session_state.get("flow_for_bt")
        if edf is None or edf.empty:
            st.warning("No eligible flow to backtest. Scan first.")
        else:
            edf = edf.copy()
            edf["timestamp"] = pd.to_datetime(edf["timestamp"])
            signals = build_daily_signals(edf)
            if signals.empty:
                st.warning("No signals produced for backtest with current filters.")
            else:
                start = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d")
                end   = datetime.utcnow().strftime("%Y-%m-%d")
                trades = backtest_signals(signals, start, end, bt_days, bt_sl, bt_tp)
                if trades.empty:
                    st.warning("No trades could be simulated (price data missing).")
                else:
                    wins = (trades["ret"] > 0).sum()
                    total = len(trades)
                    win_rate = (wins/total)*100 if total else 0.0
                    avg_ret = float(trades["ret"].mean()) if total else 0.0
                    best = float(trades["ret"].max()); worst = float(trades["ret"].min())

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Trades", f"{total}")
                    m2.metric("Win %", f"{win_rate:.1f}%")
                    m3.metric("Avg Return", f"{avg_ret*100:.2f}%")
                    m4.metric("Best", f"{best*100:.1f}%")
                    m5.metric("Worst", f"{worst*100:.1f}%")
                    st.caption("Rule: Enter next day open after a signal; exit on TP/SL intraday or after hold-days at close.")

                    by_tkr = (trades.assign(pct=lambda d: d["ret"]*100)
                                    .groupby("ticker")
                                    .agg(trades=("ret","count"),
                                         win_pct=("ret", lambda s: (s>0).mean()*100),
                                         avg_ret=("ret","mean"))
                                    .reset_index()
                                    .sort_values(["trades","win_pct"], ascending=[False, False]))
                    by_tkr["avg_ret"] = (by_tkr["avg_ret"]*100).round(2)

                    st.subheader("Per-Ticker Performance")
                    st.dataframe(by_tkr, use_container_width=True, hide_index=True)

                    st.subheader("Sample Trades")
                    st.dataframe(trades.assign(pct=lambda d: (d["ret"]*100).round(2))
                                         .sort_values("date", ascending=False)
                                         .head(50)
                                         .reset_index(drop=True),
                                 use_container_width=True, hide_index=True)
else:
    st.info("Click **Scan Market** to fetch institutional flow and generate signals.")
