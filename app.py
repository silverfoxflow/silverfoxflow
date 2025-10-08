# ==============================
# SilverFoxFlow — Market UOA Scanner (UW-native fields, kid-simple decisions)
# ==============================
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SilverFoxFlow — Market UOA Scanner", page_icon="🦊", layout="wide")
st.title("🦊 SilverFoxFlow — Market UOA Scanner")
st.caption("We scan institutional options flow across the market and surface the strongest, cleanest signals.")

# ---------- Secrets ----------
UOA = st.secrets.get("uoa", {})
PROVIDER = UOA.get("provider")
API_BASE = (UOA.get("api_base") or "").rstrip("/")
API_KEY = UOA.get("api_key")
UW_ENDPOINT = (UOA.get("endpoint") or "").strip()
UW_MIN_PARAM = (UOA.get("minutes_param") or "minutes").strip()

BUILD_TAG = datetime.utcnow().strftime("Build %Y-%m-%d %H:%M:%S UTC")
st.sidebar.success(BUILD_TAG)
st.sidebar.info({
    "provider": PROVIDER, "api_base": API_BASE, "endpoint": UW_ENDPOINT,
    "minutes_param": UW_MIN_PARAM, "has_api_key": bool(API_KEY and len(API_KEY) > 10)
})

status = "Demo Mode"
if PROVIDER and API_BASE and API_KEY:
    status = f"Live: {PROVIDER.upper()}"
st.caption(f"Data source: **{status}**")

# ---------- Controls ----------
left, right = st.columns([3,2], gap="large")
with left:
    mode = st.radio("Scan profile", ["Strict","Balanced","Explorer"], index=1, horizontal=True)
with right:
    window_min = st.select_slider("Lookback window (minutes)", options=[15,30,45,60,90,120,180], value=60)
    auto = st.toggle("Auto-refresh every 60s", value=False)

# Profile thresholds (on aggregated totals per ticker)
PROFILE = {
    "Strict":   dict(MIN_TOTAL_PREM=4_000_000, MIN_PRINTS=5, DOM=1.35),
    "Balanced": dict(MIN_TOTAL_PREM=2_000_000, MIN_PRINTS=4, DOM=1.25),
    "Explorer": dict(MIN_TOTAL_PREM=800_000,   MIN_PRINTS=3, DOM=1.15),
}[mode]

# ---------- UW fetch (multi-auth) ----------
def _map_rows_uw(data):
    """
    Map UW 'flow-alerts' style rows to consistent columns we use downstream.
    We rely on: total_premium, total_ask_side_prem, total_bid_side_prem, nprints, ticker/underlying.
    """
    def first(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    rows = []
    for x in data:
        rows.append({
            "timestamp": first(x, "timestamp","ts","time","created_at","start_time"),
            "ticker": first(x, "ticker","symbol","underlying_symbol","median_ticker","underlying"),
            "total_premium": first(x, "total_premium","sum_premium","total_prem"),
            "ask_prem": first(x, "total_ask_side_prem","ask_prem","ask_side_prem"),
            "bid_prem": first(x, "total_bid_side_prem","bid_prem","bid_side_prem"),
            "prints": first(x, "nprints","prints","count","sweep_count"),
            # Optional/useful extras:
            "has_sweep": first(x, "has_sweep","has_sweeps"),
            "volume_oi_ratio": first(x, "volume_oi_ratio","vol_oi_ratio"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in ["total_premium","ask_prem","bid_prem","prints","volume_oi_ratio"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Fill NaNs for dollars/prints to zero so aggregation works cleanly
        for c in ["total_premium","ask_prem","bid_prem","prints"]:
            df[c] = df[c].fillna(0.0)
    return df

def fetch_uw_flow(api_base: str, endpoint: str, api_key: str, minutes: int):
    base = api_base.rstrip("/")
    url = f"{base}{endpoint if endpoint.startswith('/') else '/' + endpoint}"

    attempts = [
        ({"Authorization": f"Bearer {api_key}"}, {}),
        ({"Authorization": f"Token {api_key}"}, {}),
        ({"X-API-KEY": api_key}, {}),
        ({"x-api-key": api_key}, {}),
        ({}, {"token": api_key}),
    ]
    param_keys = [UW_MIN_PARAM] + [p for p in ["minutes","window_min"] if p != UW_MIN_PARAM]
    params = {k: minutes for k in param_keys}

    errors = []
    for headers, extra_q in attempts:
        try:
            q = {**params, **extra_q}
            r = requests.get(url, headers=headers, params=q, timeout=30)
            if r.status_code != 200:
                errors.append(f"{r.status_code} {list(headers.keys())} {list(q.keys())}")
                continue
            payload = r.json()
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            df = _map_rows_uw(data)
            dbg = {
                "used_url": url, "status_code": r.status_code,
                "auth_headers_used": list(headers.keys()),
                "params_used": list(q.keys()),
                "raw_count": (len(data) if isinstance(data, list) else (len(data) if hasattr(data, "__len__") else "n/a")),
                "first_row_keys": (list(data[0].keys()) if isinstance(data, list) and data else [])
            }
            st.session_state["uw_endpoint_used"] = url
            return df, dbg
        except Exception as e:
            errors.append(str(e))
            continue

    raise RuntimeError("UW fetch failed. Tried:\n- " + "\n- ".join(errors))

# ---------- Kid-simple decision on aggregated ticker ----------
def decide_group(sum_ask: float, sum_bid: float, dom_req: float) -> str:
    # dom = bigger_side / smaller_side
    bull = float(sum_ask or 0.0)   # paying the ask → generally call-buying pressure
    bear = float(sum_bid or 0.0)   # hitting the bid → generally put-buying or selling calls
    if bull == 0 and bear == 0:
        return "NO TRADE"
    bigger = max(bull, bear); smaller = max(min(bull, bear), 1e-9)
    dom = bigger / smaller
    if bull > bear and dom >= dom_req:  # strong net ask-side dollars
        return "BUY CALLS"
    if bear > bull and dom >= dom_req:  # strong net bid-side dollars
        return "BUY PUTS"
    return "NO TRADE"

# ---------- Scan ----------
run_scan = st.button("🔎 Scan Market", type="primary")
if auto and int(time.time()) % 60 == 0:
    run_scan = True

if run_scan:
    try:
        if status.startswith("Live"):
            df, dbg = fetch_uw_flow(API_BASE, UW_ENDPOINT, API_KEY, window_min)
            st.sidebar.info({"uw_debug": dbg})
        else:
            df = pd.DataFrame()  # no demo here to avoid confusion
            st.sidebar.info({"uw_debug": "demo (disabled)"})
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        df = pd.DataFrame()

    st.session_state["flow"] = df
    st.session_state["scan_ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------- Render ----------
if "flow" in st.session_state:
    df = st.session_state["flow"].copy()
    st.markdown(f"**Last scan:** {st.session_state['scan_ts']} • Mode **{mode}** • Window **{window_min}m**")
    if "uw_endpoint_used" in st.session_state:
        st.caption(f"UW endpoint: {st.session_state['uw_endpoint_used']}")

    with st.expander("Debug: fetched rows / columns", expanded=False):
        st.write(f"Rows: {len(df)}")
        if not df.empty:
            st.write("Columns:", list(df.columns))
            st.dataframe(df.head(10), use_container_width=True)

    if df.empty:
        st.warning("No rows returned. Try increasing the lookback to 120–180 minutes.")
    else:
        # --- aggregate per ticker with UW-native dollars ---
        agg = (df.groupby("ticker", dropna=True)
                 .agg(total_premium=("total_premium","sum"),
                      ask_dollars=("ask_prem","sum"),
                      bid_dollars=("bid_prem","sum"),
                      prints_total=("prints","sum"),
                      alerts=("ticker","count"))
                 .reset_index())

        # eligibility on aggregated numbers
        eligible = agg[
            (agg["total_premium"] >= PROFILE["MIN_TOTAL_PREM"]) &
            (agg["prints_total"]  >= PROFILE["MIN_PRINTS"])
        ].copy()

        # compute decision
        eligible["decision"] = eligible.apply(
            lambda r: decide_group(r["ask_dollars"], r["bid_dollars"], PROFILE["DOM"]),
            axis=1
        )

        st.subheader("Top Picks")
        if eligible.empty:
            st.info("Data returned, but nothing met the **aggregated** thresholds yet. Switch to **Explorer** or widen lookback.")
        else:
            top = (eligible.sort_values(["decision","total_premium"],
                                        ascending=[True, False])
                          .sort_values("total_premium", ascending=False)
                          .head(10))
            cols = st.columns(min(5, len(top)))
            for i, (_, r) in enumerate(top.iterrows()):
                helper = f"${r['total_premium']:,.0f} total • ask ${r['ask_dollars']:,.0f} vs bid ${r['bid_dollars']:,.0f} • prints {int(r['prints_total'])}"
                with cols[i % len(cols)]:
                    st.metric(r["ticker"], r["decision"], help=helper)

        st.divider()
        # summary counters (safe)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Eligible tickers", int(len(eligible)))
        c2.metric("Unique tickers (raw)", int(df["ticker"].nunique()))
        c3.metric("BUY CALLS", int((eligible["decision"] == "BUY CALLS").sum()))
        c4.metric("BUY PUTS",  int((eligible["decision"] == "BUY PUTS").sum()))
else:
    st.info("Click **Scan Market** to fetch institutional flow and generate signals.")
