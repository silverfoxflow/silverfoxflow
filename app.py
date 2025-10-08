# ==============================
# SilverFoxFlow — Market UOA Scanner
# Simple picks + contract suggestion (robust to UW schemas)
# ==============================
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SilverFoxFlow — Market UOA Scanner", page_icon="🦊", layout="wide")
st.title("🦊 SilverFoxFlow — Market UOA Scanner")
st.caption("We add up big-money options dollars and recommend simple trades: BUY CALLS, BUY PUTS, or NO TRADE.")

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
c1, c2, c3 = st.columns([2,2,2])
with c1:
    profile = st.radio("Scan profile", ["Strict","Balanced","Explorer"], index=2, horizontal=True)
with c2:
    window_min = st.select_slider("Lookback (minutes)", options=[30,60,90,120,180], value=180)
with c3:
    auto = st.toggle("Auto-refresh every 60s", value=False)

BASE = {
    "Strict":   dict(min_total=3_000_000, min_prints=5, dom=1.30),
    "Balanced": dict(min_total=1_800_000, min_prints=4, dom=1.22),
    "Explorer": dict(min_total=600_000,   min_prints=2, dom=1.10),
}[profile]

# ---------- UW fetch ----------
CALL_KEYS = ["call", "c", "C", "CALL", "Call"]
PUT_KEYS  = ["put", "p", "P", "PUT", "Put"]

def _first(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _map_rows(data):
    rows = []
    for x in data:
        # Try to detect call/put if present
        opt_type = _first(x, "option_type","call_put","right","cp","type")
        if isinstance(opt_type, str):
            t = opt_type.strip().lower()
            if t in ["c","call","calls","right_call"]: opt_type = "CALL"
            elif t in ["p","put","puts","right_put"]:  opt_type = "PUT"
            else: opt_type = None
        else:
            opt_type = None

        rows.append({
            "timestamp": _first(x, "timestamp","ts","time","created_at","start_time"),
            "ticker": _first(x, "ticker","symbol","underlying_symbol","median_ticker","underlying"),
            "strike": _first(x, "strike","k","strike_price"),
            "expiry": _first(x, "expiry","expiration","exp","expir","exp_date"),
            "total_premium": _first(x, "total_premium","sum_premium","total_prem"),
            "ask_prem": _first(x, "total_ask_side_prem","ask_prem","ask_side_prem"),
            "bid_prem": _first(x, "total_bid_side_prem","bid_prem","bid_side_prem"),
            "prints": _first(x, "nprints","prints","count","sweep_count","trade_count"),
            "has_sweep": _first(x, "has_sweep","has_sweeps"),
            "vol_oi_ratio": _first(x, "volume_oi_ratio","vol_oi_ratio"),
            "underlying_price": _first(x, "underlying_price","u_price","underlying"),
            "option_type": opt_type,  # normalized or None
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        for c in ["total_premium","ask_prem","bid_prem","prints","vol_oi_ratio","strike","underlying_price"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["total_premium","ask_prem","bid_prem","prints"]:
            df[c] = df[c].fillna(0.0)
        df["has_sweep"] = df["has_sweep"].fillna(False).astype(bool)
    return df

def fetch_uw(api_base: str, endpoint: str, api_key: str, minutes: int):
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

# ---------- Core logic ----------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Split dollars by type if we can; otherwise everything stays in totals
    df["call_ask"] = np.where(df["option_type"] == "CALL", df["ask_prem"], 0.0)
    df["put_ask"]  = np.where(df["option_type"] == "PUT",  df["ask_prem"], 0.0)
    df["call_bid"] = np.where(df["option_type"] == "CALL", df["bid_prem"], 0.0)
    df["put_bid"]  = np.where(df["option_type"] == "PUT",  df["bid_prem"], 0.0)

    agg = (df.groupby("ticker", dropna=True)
             .agg(
                 total_premium=("total_premium","sum"),
                 ask_dollars=("ask_prem","sum"),
                 bid_dollars=("bid_prem","sum"),
                 prints_total=("prints","sum"),
                 sweeps=("has_sweep","sum"),
                 vol_oi_avg=("vol_oi_ratio","mean"),
                 call_ask=("call_ask","sum"),
                 put_ask=("put_ask","sum"),
                 call_bid=("call_bid","sum"),
                 put_bid=("put_bid","sum"),
             ).reset_index())

    # Direction logic (prefers proper call/put dollars if available)
    agg["bull_score"] = agg["call_ask"] + agg["put_bid"]   # buy calls + sell puts (if bid seen)
    agg["bear_score"] = agg["put_ask"] + agg["call_bid"]   # buy puts + sell calls
    has_type_info = (agg[["call_ask","put_ask"]].sum().sum() > 0)

    if has_type_info:
        agg["side"] = np.where(agg["bull_score"] > agg["bear_score"], "BUY CALLS",
                        np.where(agg["bear_score"] > agg["bull_score"], "BUY PUTS", "NO TRADE"))
        dom_raw = np.maximum(agg["bull_score"], agg["bear_score"]) / np.maximum(np.minimum(agg["bull_score"], agg["bear_score"]), 1e-9)
    else:
        # fallback: net ask vs bid (less precise if we cannot separate calls/puts)
        agg["side"] = np.where(agg["ask_dollars"] > agg["bid_dollars"], "BUY CALLS",
                        np.where(agg["bid_dollars"] > agg["ask_dollars"], "BUY PUTS", "NO TRADE"))
        dom_raw = np.maximum(agg["ask_dollars"], agg["bid_dollars"]) / np.maximum(np.minimum(agg["ask_dollars"], agg["bid_dollars"]), 1e-9)

    agg["dominance"] = np.clip(dom_raw, 1.0, None)
    denom = max(agg["total_premium"].quantile(0.95), 1.0)
    s_dollars = np.clip(agg["total_premium"]/denom, 0, 1)
    s_dom = np.clip((agg["dominance"]-1.0)/1.0, 0, 1)
    s_prints = np.clip(agg["prints_total"]/8.0, 0, 1)
    s_sweep = np.clip(agg["sweeps"]/3.0, 0, 1)
    s_voi = np.clip((agg["vol_oi_avg"]-0.8)/0.8, 0, 1)
    agg["confidence"] = np.round(100 * (0.45*s_dollars + 0.25*s_dom + 0.20*s_prints + 0.07*s_sweep + 0.03*s_voi), 1)

    # attach a representative contract (pick the biggest contributor on the winning side)
    reps = []
    for t in agg["ticker"]:
        sub = df[df["ticker"] == t].copy()
        if sub.empty:
            reps.append({"ticker": t})
            continue
        # choose by side & availability
        side = agg.loc[agg["ticker"] == t, "side"].iloc[0]
        if side == "BUY CALLS":
            if (sub["option_type"] == "CALL").any():
                pick = sub[sub["option_type"] == "CALL"].sort_values("ask_prem", ascending=False).head(1)
            else:
                pick = sub.sort_values("ask_prem", ascending=False).head(1)
        elif side == "BUY PUTS":
            if (sub["option_type"] == "PUT").any():
                pick = sub[sub["option_type"] == "PUT"].sort_values("ask_prem", ascending=False).head(1)
            else:
                # if type unknown, use the row with biggest bid or ask depending on direction
                pick = sub.sort_values(["bid_prem","ask_prem"], ascending=False).head(1)
        else:
            pick = sub.sort_values("total_premium", ascending=False).head(1)

        pr = pick.iloc[0].to_dict()
        reps.append({
            "ticker": t,
            "strike": pr.get("strike"),
            "expiry": pr.get("expiry"),
            "underlying_price": pr.get("underlying_price"),
            "prints_rep": pr.get("prints"),
        })
    reps = pd.DataFrame(reps)
    out = agg.merge(reps, on="ticker", how="left")
    return out.sort_values(["confidence","total_premium"], ascending=[False, False]).reset_index(drop=True)

def decide_with_gate(row, min_total, min_prints, min_dom):
    if (row["total_premium"] >= min_total) and (row["prints_total"] >= min_prints) and (row["dominance"] >= min_dom):
        return row["side"]
    return "NO TRADE"

def adaptive_pick(agg: pd.DataFrame, base, want=5):
    if agg.empty:
        return agg, dict(step="no-data", min_total=0, min_prints=0, min_dom=1.0)

    min_total = base["min_total"]; min_prints = base["min_prints"]; min_dom = base["dom"]
    floor_total, floor_prints, floor_dom = 200_000, 1, 1.05

    for step in range(10):
        tmp = agg.copy()
        tmp["decision"] = tmp.apply(lambda r: decide_with_gate(r, min_total, min_prints, min_dom), axis=1)
        picks = tmp[tmp["decision"] != "NO TRADE"].copy().sort_values(["confidence","total_premium"], ascending=[False, False]).head(want)
        if len(picks) >= want or (min_total <= floor_total and min_prints <= floor_prints and min_dom <= floor_dom):
            if len(picks) < want:  # fill with top confidence names
                fillers = tmp[tmp["decision"] == "NO TRADE"].head(want - len(picks))
                picks = pd.concat([picks, fillers], ignore_index=True)
            return picks, dict(step=step, min_total=min_total, min_prints=min_prints, min_dom=min_dom)
        # relax
        min_total = max(floor_total, int(min_total * 0.70))
        min_prints = max(floor_prints, int(np.floor(min_prints * 0.8)))
        min_dom = max(floor_dom, round(min_dom - 0.05, 2))

    # final fallback
    tmp = agg.copy()
    tmp["decision"] = tmp.apply(lambda r: decide_with_gate(r, floor_total, floor_prints, floor_dom), axis=1)
    picks = tmp.sort_values(["confidence","total_premium"], ascending=[False, False]).head(want)
    return picks, dict(step="fallback", min_total=floor_total, min_prints=floor_prints, min_dom=floor_dom)

def trade_card(row) -> str:
    """Human-friendly one-liner for the contract & exit plan."""
    side = row["side"] if row.get("decision","NO TRADE") != "NO TRADE" else "NO TRADE"
    exp = row.get("expiry")
    exp_txt = exp.strftime("%Y-%m-%d") if pd.notna(exp) else "near-term weekly"
    strike = row.get("strike")
    strike_txt = (f"{strike:.0f}" if pd.notna(strike) else "nearest liquid")
    # simple exits: time stop + profit targets
    return f"{side}: buy {row['ticker']} {strike_txt} exp {exp_txt}. Targets +30% / +50%. Time-stop: exit by Thu/Fri close."

# ---------- Scan ----------
run = st.button("🔎 Scan Market", type="primary")
if auto and int(time.time()) % 60 == 0:
    run = True

if run:
    try:
        if status.startswith("Live"):
            df, dbg = fetch_uw(API_BASE, UW_ENDPOINT, API_KEY, window_min)
            st.sidebar.info({"uw_debug": dbg})
        else:
            df = pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        df = pd.DataFrame()

    st.session_state["flow"] = df
    st.session_state["scan_ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------- Render ----------
if "flow" in st.session_state:
    df = st.session_state["flow"].copy()
    st.markdown(f"**Last scan:** {st.session_state['scan_ts']} • Mode **{profile}** • Window **{window_min}m**")
    if "uw_endpoint_used" in st.session_state:
        st.caption(f"UW endpoint: {st.session_state['uw_endpoint_used']}")

    with st.expander("Debug: fetched rows / columns", expanded=False):
        st.write(f"Rows: {len(df)}")
        if not df.empty:
            st.write("Columns:", list(df.columns))
            st.dataframe(df.head(12), use_container_width=True)

    if df.empty:
        st.warning("No rows returned. Try 120–180 minutes after the open.")
    else:
        agg = summarize(df)
        picks, gates = adaptive_pick(agg, BASE, want=5)

        st.subheader("Top Picks (auto-tuned)")
        st.caption(f"Gates → min_total=${gates['min_total']:,}, min_prints={gates['min_prints']}, dominance≥{gates['min_dom']} (step {gates['step']})")

        cols = st.columns(min(5, len(picks)))
        for i, (_, r) in enumerate(picks.iterrows()):
            # make a user-friendly card
            side = r["side"] if r["decision"] != "NO TRADE" else "NO TRADE"
            helper = (
                f"${r['total_premium']:,.0f} total | dom {r['dominance']:.2f} | "
                f"prints {int(r['prints_total'])} | sweeps {int(r['sweeps'])}"
            )
            with cols[i % len(cols)]:
                st.metric(f"{r['ticker']}  •  {int(r['confidence'])}⚡", side, help=helper)
                st.write(trade_card(r))

        st.divider()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Names shown", len(picks))
        s2.metric("Raw tickers", int(df["ticker"].nunique()))
        s3.metric("BUY CALLS", int((picks["decision"] == "BUY CALLS").sum()))
        s4.metric("BUY PUTS",  int((picks["decision"] == "BUY PUTS").sum()))
else:
    st.info("Click **Scan Market** to fetch institutional flow and generate signals.")
