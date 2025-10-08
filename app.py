# ==============================
# SilverFoxFlow — Market UOA Scanner (UW multi-auth + deep debug)
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

# ---------- Demo fallback ----------
def _demo_flow(n: int = 200, max_window: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.utcnow()
    tickers = ["AAPL","MSFT","NVDA","META","TSLA","AMZN","AMD","NFLX","MRVL","CAT","MCD","DDOG","SMCI","AVGO"]
    ts = [now - timedelta(minutes=int(rng.integers(0, max_window))) for _ in range(n)]
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in ts],
        "ticker": rng.choice(tickers, size=n),
        "expiry_weeks": rng.integers(1, 16, size=n),
        "notional": rng.choice([1.2e6, 2.5e6, 4e6, 6e6, 8e6, 12e6], size=n),
        "prints": rng.integers(3, 20, size=n),
        "aggr_ratio": np.round(rng.uniform(0.3, 0.8, size=n), 2),
        "volume": rng.integers(1000, 30000, size=n),
        "open_interest": rng.integers(800, 25000, size=n),
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
            "notional": first(x, "premium","prem","notional","usd_value","dollar_value"),
            "prints": first(x, "prints","nprints","count","num_trades","sweeps","blocks"),
            "aggr_ratio": first(x, "aggr_ratio","aggressor_ratio","ask_hit_ratio","at_ask_ratio"),
            "volume": first(x, "volume","contracts","contracts_traded"),
            "open_interest": first(x, "open_interest","open_int","oi"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in ["expiry_weeks","notional","prints","aggr_ratio","volume","open_interest"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_uw_flow(api_base: str, endpoint: str, api_key: str, minutes: int):
    """
    Try multiple auth styles + show deep diagnostics.
    Returns (df, debug_dict)
    """
    base = api_base.rstrip("/")
    url = f"{base}{endpoint if endpoint.startswith('/') else '/' + endpoint}"

    # Try multiple header styles then query-param token as last resort
    attempts = [
        ({"Authorization": f"Bearer {api_key}"}, {}),             # Bearer
        ({"Authorization": f"Token {api_key}"}, {}),              # Token
        ({"X-API-KEY": api_key}, {}),                             # X-API-KEY (titlecase)
        ({"x-api-key": api_key}, {}),                             # x-api-key (lower)
        ({}, {"token": api_key}),                                 # token in query
    ]

    # Try both 'minutes' and 'window_min' (and the custom minutes_param from secrets)
    param_keys = [UW_MIN_PARAM] + [p for p in ["minutes","window_min"] if p != UW_MIN_PARAM]
    params = {k: minutes for k in param_keys}

    errors = []
    for headers, extra_q in attempts:
        try:
            q = {**params, **extra_q}
            r = requests.get(url, headers=headers, params=q, timeout=30)
            code = r.status_code
            # accept 200; anything else collect and keep trying
            if code != 200:
                errors.append(f"{code} with headers {list(headers.keys())} & params {list(q.keys())}")
                continue
            payload = r.json()
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            df = _map_rows(data)
            dbg = {
                "used_url": url,
                "status_code": code,
                "auth_headers_used": list(headers.keys()),
                "params_used": list(q.keys()),
                "raw_count": (len(data) if isinstance(data, list) else (len(data) if hasattr(data, "__len__") else "n/a")),
                "first_row_keys": (list(data[0].keys()) if isinstance(data, list) and data else []),
            }
            st.session_state["uw_endpoint_used"] = url
            return df, dbg
        except Exception as e:
            errors.append(str(e))
            continue

    raise RuntimeError("UW fetch failed. Tried auth/param combos:\n- " + "\n- ".join(errors))

# ---------- Simple decision rule ----------
def decide(g: pd.DataFrame) -> str:
    if g.empty: return "NO TRADE"
    bull = (g["notional"] * g["aggr_ratio"]).sum()
    bear = (g["notional"] * (1 - g["aggr_ratio"])).sum()
    if bull > bear * 1.3: return "BUY CALLS"
    if bear > bull * 1.3: return "BUY PUTS"
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

    # fast diagnostics
    with st.expander("Debug: fetched rows / columns", expanded=False):
        st.write(f"Rows: {len(df)}")
        if not df.empty:
            st.write("Columns:", list(df.columns))
            st.write("Null counts:", df.isna().sum().to_dict())
            st.dataframe(df.head(10), use_container_width=True)

    if df.empty:
        st.warning("No rows returned. Try increasing the lookback to 120–180 minutes or verify the endpoint/plan includes flow alerts.")
    else:
        # very light eligibility (keep it simple for now)
        filtered = df[df["notional"] >= (3_000_000 if mode=="Strict" else 2_000_000 if mode=="Balanced" else 1_000_000)]
        st.subheader("Top Picks")
        if filtered.empty:
            st.info("Data returned but nothing met the minimum notional. Lower the profile (Explorer) or widen lookback.")
        else:
            # Rank by total notional per ticker
            agg = filtered.groupby("ticker").agg(total_notional=("notional","sum")).reset_index()
            top = agg.sort_values("total_notional", ascending=False).head(10)["ticker"].tolist()
            cols = st.columns(min(5, len(top)))
            for i, tkr in enumerate(top):
                g = filtered[filtered["ticker"] == tkr]
                decision = decide(g)
                with cols[i % len(cols)]:
                    st.metric(tkr, decision)

        st.divider()
        st.metric("Eligible prints", f"{len(filtered):,}")
        st.metric("Unique tickers", f"{filtered['ticker'].nunique():,}")
        st.metric("BUY CALLS", int((filtered.groupby('ticker').apply(decide) == "BUY CALLS").sum()))
        st.metric("BUY PUTS", int((filtered.groupby('ticker').apply(decide) == "BUY PUTS").sum()))
else:
    st.info("Click **Scan Market** to fetch institutional flow and generate signals.")
