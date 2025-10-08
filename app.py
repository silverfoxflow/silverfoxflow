# ==============================
# SilverFoxFlow — Market UOA Scanner (UW API Integrated + Secrets Debug)
# ==============================
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="SilverFoxFlow — Market UOA Scanner", page_icon="🦊", layout="wide")
st.title("🦊 SilverFoxFlow — Market UOA Scanner")
st.caption("We scan institutional options flow across the market and surface the strongest, cleanest signals.")

# ---- Load Secrets ----
UOA = st.secrets.get("uoa", {})
PROVIDER = UOA.get("provider")
API_BASE = (UOA.get("api_base") or "").rstrip("/")
API_KEY = UOA.get("api_key")
UW_ENDPOINT = (UOA.get("endpoint") or "").strip()
UW_MIN_PARAM = (UOA.get("minutes_param") or "minutes").strip()

# ---- Debug Info Sidebar ----
dbg = {
    "provider": PROVIDER,
    "api_base": API_BASE,
    "endpoint": UW_ENDPOINT,
    "minutes_param": UW_MIN_PARAM,
    "has_api_key": bool(API_KEY and len(API_KEY) > 10),
}
st.sidebar.info(f"Secrets loaded: {dbg}")

# ---- Determine Live vs Demo Mode ----
status = "Demo Mode"
if PROVIDER and API_BASE and API_KEY:
    status = f"Live: {PROVIDER.upper()}"
st.caption(f"Data source: **{status}**")

# -----------------------------
# Scan Configuration Presets
# -----------------------------
PRESETS = {
    "Strict": dict(MIN_PREM=3_000_000, MIN_PRINTS=8, EXP_MIN_W=1, EXP_MAX_W=8),
    "Balanced": dict(MIN_PREM=2_000_000, MIN_PRINTS=6, EXP_MIN_W=1, EXP_MAX_W=10),
    "Explorer": dict(MIN_PREM=1_000_000, MIN_PRINTS=4, EXP_MIN_W=1, EXP_MAX_W=16),
}

# -----------------------------
# Controls
# -----------------------------
left, right = st.columns([3, 2], gap="large")
with left:
    mode = st.radio("Scan profile", list(PRESETS.keys()), index=1, horizontal=True)
cfg = PRESETS[mode]
with right:
    window_min = st.select_slider("Lookback window (minutes)", options=[15, 30, 45, 60, 90, 120], value=60)
    auto = st.toggle("Auto-refresh every 60s", value=False)

# -----------------------------
# Helper functions
# -----------------------------
def _demo_flow(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.utcnow()
    tickers = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "AMD", "NFLX", "MRVL", "CAT", "MCD", "DDOG"]
    ts = [now - timedelta(minutes=int(rng.integers(0, 120))) for _ in range(n)]
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in ts],
        "ticker": rng.choice(tickers, size=n),
        "expiry_weeks": rng.integers(1, 12, size=n),
        "notional": rng.choice([1.2e6, 2.5e6, 4e6, 6e6, 8e6], size=n),
        "prints": rng.integers(3, 20, size=n),
        "aggr_ratio": np.round(rng.uniform(0.3, 0.8, size=n), 2),
        "volume": rng.integers(1000, 20000, size=n),
        "open_interest": rng.integers(800, 18000, size=n),
    })
    return df

def _map_uw_rows_to_df(data):
    def first(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None
    rows = []
    for x in data:
        rows.append({
            "timestamp": first(x, "timestamp", "ts", "time", "created_at"),
            "ticker": first(x, "ticker", "symbol", "underlying_symbol"),
            "expiry_weeks": first(x, "expiry_weeks", "weeks_to_expiry"),
            "notional": first(x, "premium", "notional", "usd_value"),
            "prints": first(x, "prints", "sweeps", "blocks"),
            "aggr_ratio": first(x, "aggr_ratio", "aggressor_ratio", "ask_hit_ratio"),
            "volume": first(x, "volume"),
            "open_interest": first(x, "open_interest", "oi"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in ["expiry_weeks", "notional", "prints", "aggr_ratio", "volume", "open_interest"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_uw_flow(api_base, api_key, endpoint, window_min):
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {UW_MIN_PARAM: window_min}
    url = f"{api_base}{endpoint}"
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    payload = data.get("data", data) if isinstance(data, dict) else data
    st.session_state["uw_endpoint_used"] = url
    return _map_uw_rows_to_df(payload)

def summarize_ticker(g):
    bull = (g["notional"] * g["aggr_ratio"]).sum()
    bear = (g["notional"] * (1 - g["aggr_ratio"])).sum()
    if bull > bear * 1.3:
        return "BUY CALLS"
    elif bear > bull * 1.3:
        return "BUY PUTS"
    else:
        return "NO TRADE"

# -----------------------------
# Scan Button
# -----------------------------
run_scan = st.button("🔎 Scan Market", type="primary")
if auto and int(time.time()) % 60 == 0:
    run_scan = True

if run_scan:
    try:
        if PROVIDER and API_BASE and API_KEY:
            df = fetch_uw_flow(API_BASE, API_KEY, UW_ENDPOINT, window_min)
        else:
            df = _demo_flow()
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        df = _demo_flow()

    st.session_state["flow"] = df
    st.session_state["scan_ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -----------------------------
# Display Results
# -----------------------------
if "flow" in st.session_state:
    df = st.session_state["flow"]
    st.markdown(f"**Last scan:** {st.session_state['scan_ts']} • Mode **{mode}** • Window **{window_min}m**")

    if "uw_endpoint_used" in st.session_state:
        st.caption(f"UW endpoint: {st.session_state['uw_endpoint_used']}")

    if df.empty:
        st.warning("No data found.")
    else:
        df = df[df["notional"] >= cfg["MIN_PREM"]]
        results = []
        for tkr, g in df.groupby("ticker"):
            results.append((tkr, summarize_ticker(g)))
        results = pd.DataFrame(results, columns=["Ticker", "Decision"])
        st.subheader("Top Picks")
        for _, row in results.iterrows():
            st.metric(row["Ticker"], row["Decision"])

        st.divider()
        st.write(f"Eligible prints: {len(df):,}")
        st.write(f"Unique tickers: {df['ticker'].nunique():,}")
        st.write(f"BUY CALLS: {(results['Decision'] == 'BUY CALLS').sum()}")
        st.write(f"BUY PUTS: {(results['Decision'] == 'BUY PUTS').sum()}")

else:
    st.info("Click **Scan Market** to fetch institutional flow and generate signals.")
