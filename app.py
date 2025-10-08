# ==============================
# SilverFoxFlow — Market UOA Scanner (UW fixed aggregation + safe metrics)
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

# Profile thresholds (now used on AGGREGATED totals)
PROFILE = {
    "Strict":   dict(MIN_TOTAL_PREM=5_000_000, MIN_PRINTS=8,  DOMINANCE=1.3),
    "Balanced": dict(MIN_TOTAL_PREM=3_000_000, MIN_PRINTS=6,  DOMINANCE=1.3),
    "Explorer": dict(MIN_TOTAL_PREM=1_500_000, MIN_PRINTS=4,  DOMINANCE=1.2),
}[mode]

# ---------- Demo fallback ----------
def _demo_flow(n: int = 300, max_window: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.utcnow()
    tickers = ["AAPL","MSFT","NVDA","META","TSLA","AMZN","AMD","NFLX","MRVL","CAT","MCD","DDOG","SMCI","AVGO"]
    ts = [now - timedelta(minutes=int(rng.integers(0, max_window))) for _ in range(n)]
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in ts],
        "ticker": rng.choice(tickers, size=n),
        "expiry_weeks": rng.integers(1, 16, size=n),
        "notional": rng.choice([4e5, 8e5, 1.2e6, 2.5e6, 4e6, 6e6, 8e6], size=n),
        "prints": rng.integers(1, 6, size=n),
        "aggr_ratio": np.round(rng.uniform(0.3, 0.8, size=n), 2),
        "volume": rng.integers(500, 30000, size=n),
        "open_interest": rng.integers(300, 25000, size=n),
    })
    return df

# ---------- UW fetch + diagnostics ----------
def _map_rows(data):
    def first(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    rows = []
    for x in data:
        rows.append({
            "timestamp": first(x, "timestamp","ts","time","created_at"),
            "ticker": first(x, "ticker","symbol","underlying","underlying_symbol"),
            "expiry_weeks": first(x, "expiry_weeks","weeks_to_expiry","w","weeks"),
            "notional": first(x, "premium","prem","notional","usd_value","dollar_value","amount"),
            "prints": first(x, "prints","nprints","count","num_trades","sweeps","blocks", "sweep_count"),
            "aggr_ratio": first(x, "aggr_ratio","aggressor_ratio","ask_hit_ratio","at_ask_ratio","sentiment"),
            "volume": first(x, "volume","contracts","contracts_traded"),
            "open_interest": first(x, "open_interest","open_int","oi"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in ["expiry_weeks","notional","prints","aggr_ratio","volume","open_interest"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # If 'sentiment' came in (0..1 bullishness), cap to [0,1]
        if df["aggr_ratio"].notna().any():
            df["aggr_ratio"] = df["aggr_ratio"].clip(lower=0, upper=1)
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
                errors.append(f"{r.status_code} with headers {list(headers.keys())} & params {list(q.keys())}")
                continue
            payload = r.json()
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            df = _map_rows(data)
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

# ---------- Decision on aggregated group ----------
def decide_group(g: pd.DataFrame) -> str:
    # dollars toward calls vs puts using aggressor ratio
    bull = float((g["notional"] * g["aggr_ratio"]).sum())
    bear = float((g["notional"] * (1 - g["aggr_ratio"])).sum())
    if bear == 0 and bull == 0:
        return "NO TRADE"
    dom = (bull / max(bear, 1e-9)) if bear > 0 else float("inf")
    if dom >= PROFILE["DOMINANCE"]:
        return "BUY CALLS"
    if (1.0/dom) >= PROFILE["DOMINANCE"]:
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
            df = _demo_flow(max_window=window_min)
            st.sidebar.info({"uw_debug": "demo"})
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        df = _demo_flow(max_window=window_min)

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
            st.write("Null counts:", df.isna().sum().to_dict())
            st.dataframe(df.head(10), use_container_width=True)

    if df.empty:
        st.warning("No rows returned. Increase lookback to 120–180 minutes or confirm your plan includes flow alerts.")
    else:
        # --------- aggregate per ticker ---------
        grp = df.groupby("ticker", dropna=True)

        agg = grp.agg(
            total_premium = ("notional","sum"),
            prints_total  = ("prints","sum"),
            avg_aggr      = ("aggr_ratio","mean")
        ).reset_index()

        # eligibility on aggregated totals
        elig = agg[
            (agg["total_premium"] >= PROFILE["MIN_TOTAL_PREM"]) &
            (agg["prints_total"]  >= PROFILE["MIN_PRINTS"])
        ]

        st.subheader("Top Picks")
        if elig.empty:
            st.info("Data returned, but nothing met the **aggregated** thresholds yet. Try **Explorer** profile or widen lookback.")
        else:
            # Rank by total premium
            elig_sorted = elig.sort_values("total_premium", ascending=False)
            top_ticks = elig_sorted["ticker"].head(10).tolist()

            cols = st.columns(min(5, len(top_ticks)))
            for i, tkr in enumerate(top_ticks):
                g = df[df["ticker"] == tkr]
                decision = decide_group(g)
                with cols[i % len(cols)]:
                    st.metric(tkr, decision, help=f"${elig_sorted.loc[elig_sorted['ticker']==tkr, 'total_premium'].iloc[0]:,.0f} total")

        st.divider()
        # --------- safe metrics ---------
        st.metric("Eligible tickers", int(len(elig)))
        st.metric("Unique tickers (raw)", int(df["ticker"].nunique()))

        if not elig.empty:
            # compute decisions only for eligible tickers (avoids Series conversion errors)
            dec_series = pd.Series(
                {t: decide_group(df[df["ticker"] == t]) for t in elig["ticker"]}
            )
            st.metric("BUY CALLS", int((dec_series == "BUY CALLS").sum()))
            st.metric("BUY PUTS",  int((dec_series == "BUY PUTS").sum()))
        else:
            st.metric("BUY CALLS", 0)
            st.metric("BUY PUTS", 0)
else:
    st.info("Click **Scan Market** to fetch institutional flow and generate signals.")
