# SilverFoxFlow — Market-Wide UOA Scanner
# - Live scan via provider API (set in st.secrets)
# - Per-ticker aggregation + UOA 2.0 verdicts
# - Clean, simple UI (no CSVs)

import os, time, hashlib, random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import requests
import streamlit as st

# =========================
# PAGE / THEME
# =========================
st.set_page_config(page_title="SilverFoxFlow — Market UOA Scanner", page_icon="🦊", layout="wide")
st.title("🦊 SilverFoxFlow — Market UOA Scanner")
st.caption("We scan institutional options flow across the market and surface the strongest, cleanest signals.")

# =========================
# CONFIG PRESETS (UOA 2.0)
# =========================
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

left, right = st.columns([3,2], gap="large")
with left:
    mode = st.radio("Scan profile", list(PRESETS.keys()), index=1, horizontal=True)
cfg = PRESETS[mode]
with right:
    window_min = st.select_slider("Lookback window (minutes)", options=[15, 30, 45, 60, 90, 120], value=60)
    autorefresh = st.toggle("Auto-refresh every 60s", value=False)

# =========================
# PROVIDER / SECRETS
# Put secrets in .streamlit/secrets.toml:
# [uoa]
# provider = "generic"        # or "custom"
# api_url  = "https://YOUR_ENDPOINT/path"
# api_key  = "YOUR_KEY"
# =========================
UOA = st.secrets.get("uoa", {})
PROVIDER = UOA.get("provider")
API_URL  = UOA.get("api_url")
API_KEY  = UOA.get("api_key")

# Columns we expect after normalization
REQUIRED_COLS = ["timestamp","ticker","expiry_weeks","notional","prints","aggr_ratio","volume","open_interest"]

def _stable_minutes_old(ts: pd.Series) -> pd.Series:
    as_of = pd.to_datetime(ts).max()
    mins = (as_of - pd.to_datetime(ts)).dt.total_seconds() / 60.0
    return mins.clip(lower=0).fillna(60)

def _recency_weight(minutes_old: pd.Series) -> pd.Series:
    return pd.Series([1.0 if m < 10 else 0.7 if m < 30 else 0.5 for m in minutes_old], index=minutes_old.index)

# =========================
# DATA FETCHERS
# Implement one adapter. If no secrets, run Demo Mode.
# =========================
def fetch_flow_generic(api_url: str, api_key: str, window_min: int) -> pd.DataFrame:
    """
    Expected provider response -> list of prints with fields we can map to REQUIRED_COLS.
    Adjust mapping here to your provider.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"window_min": window_min}
    r = requests.get(api_url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # ---- MAP YOUR FIELDS HERE ----
    # Example mapping; change keys to match your provider:
    # ts -> 'timestamp', sym -> 'ticker', w -> 'expiry_weeks', prem -> 'notional',
    # nprints -> 'prints', aggressor -> 'aggr_ratio', vol -> 'volume', oi -> 'open_interest'
    rows = []
    for x in data:
        rows.append(dict(
            timestamp   = x.get("ts") or x.get("timestamp"),
            ticker      = x.get("sym") or x.get("ticker"),
            expiry_weeks= x.get("w")   or x.get("expiry_weeks"),
            notional    = x.get("prem") or x.get("notional"),
            prints      = x.get("nprints") or x.get("prints"),
            aggr_ratio  = x.get("aggressor") or x.get("aggr_ratio"),
            volume      = x.get("vol") or x.get("volume"),
            open_interest = x.get("oi") or x.get("open_interest"),
        ))
    df = pd.DataFrame(rows)
    return df

def demo_flow(window_min: int) -> pd.DataFrame:
    """Demo data so the app runs without keys — multiple tickers, varied flow."""
    random.seed(42)
    now = datetime.utcnow()
    tickers = ["AAPL","MSFT","NVDA","META","TSLA","AMZN","AMD","NFLX","NOC","MRVL","CAT","MCD","DDOG","SHOP","SMCI","PANW","ORCL","GE","AVGO","LMT"]
    rows = []
    for _ in range(300):
        tkr = random.choice(tickers)
        mins_ago = random.randint(0, window_min)
        ts = now - timedelta(minutes=mins_ago)
        prem = random.choice([1.2e6, 2.5e6, 3e6, 4.5e6, 6e6, 8e6])
        prints = random.randint(3, 20)
        w = random.randint(1, 12)
        aggr = round(random.uniform(0.3, 0.8), 2)
        vol = random.randint(2000, 20000)
        oi = random.randint(1500, 18000)
        rows.append(dict(timestamp=ts.isoformat(sep=' '), ticker=tkr, expiry_weeks=w, notional=prem,
                         prints=prints, aggr_ratio=aggr, volume=vol, open_interest=oi))
    return pd.DataFrame(rows)

def get_market_flow(window_min: int) -> pd.DataFrame:
    try:
        if PROVIDER and API_URL and API_KEY:
            df = fetch_flow_generic(API_URL, API_KEY, window_min)
        else:
            df = demo_flow(window_min)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        df = demo_flow(window_min)
    # normalize types
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_COLS].dropna(subset=["timestamp","ticker"])

# =========================
# ELIGIBILITY + VERDICT
# =========================
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
    prints = int(g["prints"].sum())  # cluster size proxy
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

# =========================
# SCAN BUTTON + REFRESH
# =========================
scan_col, time_col = st.columns([1,3])
with scan_col:
    run = st.button("🔎 Scan Market", type="primary")
with time_col:
    st.write("")

if autorefresh:
    st.experimental_singleton.clear()  # keep memory light between auto runs
    st.experimental_rerun() if int(time.time()) % 60 == 0 else None

if run:
    df = get_market_flow(window_min)
    mask = eligible_mask(df)
    edf = df.loc[mask].copy()
    by_ticker = []
    for tkr, g in (edf.groupby("ticker") if not edf.empty else []):
        s = summarize_ticker(g)
        by_ticker.append(dict(
            ticker=tkr,
            decision=s["decision"],
            reason=s["reason"],
            total_premium=round(s["total_premium"],2),
            bull_dollars=round(s["bull"],2),
            bear_dollars=round(s["bear"],2),
            dominance=("∞" if s["dominance"]==float("inf") else round(s["dominance"],2)),
            avg_aggr=round(s["avg_aggr"],3),
            prints=s["prints"]
        ))

    result = pd.DataFrame(by_ticker).sort_values(
        ["decision","total_premium"], ascending=[True, False]
    ) if by_ticker else pd.DataFrame(columns=[
        "ticker","decision","reason","total_premium","bull_dollars","bear_dollars","dominance","avg_aggr","prints"
    ])

    st.session_state["scan_ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.session_state["result"] = result
    st.session_state["eligible_rows"] = int(len(edf))
    st.session_state["unique_tickers"] = int(edf["ticker"].nunique()) if not edf.empty else 0

# =========================
# RENDER RESULTS
# =========================
if "result" in st.session_state:
    st.markdown(f"**Last scan:** {st.session_state['scan_ts']} • Mode: **{mode}** • Window: **{window_min}m**")
    res = st.session_state["result"]
    snap = dict(eligible=st.session_state["eligible_rows"], names=st.session_state["unique_tickers"])

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
    c1.metric("Eligible prints", f"{snap['eligible']:,}")
    c2.metric("Unique tickers", f"{snap['names']:,}")
    c3.metric("BUY CALLS", int((res["decision"]=="BUY CALLS").sum()))
    c4.metric("BUY PUTS", int((res["decision"]=="BUY PUTS").sum()))

    st.subheader("All Signals")
    st.caption("Sorted by total premium — highest conviction first.")
    st.dataframe(res.reset_index(drop=True), use_container_width=True, hide_index=True)
else:
    st.info("Click **Scan Market** to fetch live flow and generate signals.")
