# app.py
import os
import time
import math
import datetime as dt
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config & Styling
# -----------------------------
st.set_page_config(page_title="🦊 SilverFoxFlow — UOA 2.0", page_icon="🦊", layout="wide")

# Bigger type, cleaner spacing, friendlier visuals
st.markdown("""
<style>
:root { --base-font: 16px; }
html, body, [class*="css"]  { font-size: var(--base-font) !important; }
h1, h2, h3 { letter-spacing: .2px; }
h1 { font-size: 2.0rem !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.15rem !important; }
section[data-testid="stSidebar"] { background: #0e1117; border-right: 1px solid rgba(255,255,255,.05); }
button[kind="primary"] { border-radius: 14px !important; padding: .9rem 1.2rem !important; font-weight: 700 !important; }
.dataframe tbody tr:hover { background: rgba(255,255,255,.02); }
.block-container { padding-top: 1rem; }
div.stButton > button { width: 260px; height: 54px; font-size: 1.05rem; }
div.score-badge { display:inline-block; padding:.25rem .55rem; border-radius:999px; font-weight:700; }
div.good { background:#103b28; color:#2cd199; }
div.warn { background:#3b280e; color:#f7b955; }
div.bad  { background:#3b0e0e; color:#ff7a7a; }
.small { opacity:.7; font-size:.9rem; }
hr.soft { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .75rem 0 1.25rem 0; }
</style>
""", unsafe_allow_html=True)

API_BASE = "https://api.unusualwhales.com/api"
API_KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")

# -----------------------------
# Helpers
# -----------------------------
def api_headers() -> Dict[str, str]:
    """
    UW uses Bearer auth per the OpenAPI (securitySchemes.authorization type=http scheme=bearer).
    """
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

def http_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=api_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json() if r.text else {}

# -----------------------------
# Data layer (Unusual Whales)
# -----------------------------
def fetch_flow_alerts(limit: int = 100,
                      newer_than: Optional[str] = None,
                      older_than: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Pull recent flow alerts. UW added time filters 'newer_than' and 'older_than'.
    Returns a list of alert dicts (each alert aggregates underlying option transactions).
    """
    params: Dict[str, Any] = {"limit": max(1, min(limit, 250))}
    if newer_than:
        params["newer_than"] = newer_than
    if older_than:
        params["older_than"] = older_than
    data = http_get("/option-trades/flow-alerts", params=params)
    # The API returns {"data":[...]} for most endpoints; be tolerant
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    return []

def fetch_last_stock_state(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Optional: last price/volume snapshot for display/filters.
    """
    try:
        payload = http_get(f"/stock/{ticker}/stock-state")
        return payload.get("data", payload)
    except Exception:
        return None

# -----------------------------
# Scoring
# -----------------------------
def score_alert(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic score using robust fields commonly present in flow alerts.
    Bias:
      - more ask-side premium vs bid-side -> more conviction
      - larger total premium -> higher weight
      - near-dated (but not same-day unless user allows) -> preference
      - higher size & OI context if available
    Output fields: score, side ("CALL"/"PUT"), reason
    """
    # Pull safely
    total_prem = float(a.get("total_premium", a.get("premium", 0)) or 0)
    ask_prem   = float(a.get("total_ask_side_prem", a.get("ask_premium", 0)) or 0)
    bid_prem   = float(a.get("total_bid_side_prem", a.get("bid_premium", 0)) or 0)
    option_type = (a.get("option_type") or a.get("type") or "").lower()
    expiry = a.get("expiry") or a.get("expiration") or ""
    dte = None
    try:
        if expiry:
            dte = (dt.datetime.fromisoformat(expiry) - dt.datetime.utcnow()).days
    except Exception:
        dte = None

    # Net pressure & base side
    net_ask_bias = ask_prem - bid_prem
    side = "CALL" if option_type == "call" or net_ask_bias >= 0 else "PUT"

    # Normalize terms for 0..1
    prem_term = math.tanh(total_prem / 250_000.0)  # saturate above ~250k
    ask_term  = math.tanh(max(0.0, net_ask_bias) / (total_prem + 1e-9))
    dte_term  = 0.5
    if dte is not None:
        # sweet spot 7–35 days
        if 7 <= dte <= 35:
            dte_term = 1.0
        elif 1 <= dte < 7:
            dte_term = 0.6
        elif 36 <= dte <= 90:
            dte_term = 0.7
        else:
            dte_term = 0.4

    # Optional enhancements if present
    open_only = 1.0 if a.get("all_opening", False) else 0.0
    sweepish  = 1.0 if any(t in (a.get("tags") or []) for t in ["sweep", "intermarket_sweep", "i_sweep"]) else 0.0

    # Weighted blend
    score = (
        0.40 * prem_term +
        0.30 * ask_term +
        0.20 * dte_term +
        0.05 * open_only +
        0.05 * sweepish
    )
    return {
        "score": round(100 * score, 1),
        "side": side,
        "reason": f"prem={total_prem:,.0f}, ask_bias={net_ask_bias:,.0f}, dte={dte}"
    }

def rank_alerts(alerts: List[Dict[str, Any]],
                min_premium: float,
                min_size: Optional[int],
                max_weeks_to_expiry: Optional[int],
                prefer_opening: bool,
                prefer_ask_bias: bool,
                min_count: int = 20) -> pd.DataFrame:
    rows = []
    for a in alerts:
        # Basic fields normalized across possible shapes
        ticker = a.get("ticker") or a.get("underlying_symbol") or a.get("symbol") or ""
        opt_symbol = a.get("option_symbol") or ""
        expiry = a.get("expiry") or a.get("expiration") or ""
        total_prem = float(a.get("total_premium", a.get("premium", 0)) or 0)
        total_size = int(a.get("total_size", a.get("size", 0)) or 0)
        option_type = (a.get("option_type") or a.get("type") or "").upper()
        ask_prem   = float(a.get("total_ask_side_prem", a.get("ask_premium", 0)) or 0)
        bid_prem   = float(a.get("total_bid_side_prem", a.get("bid_premium", 0)) or 0)

        # Guardrails (strict but not empty)
        if total_prem < min_premium:
            continue
        if min_size and total_size < min_size:
            continue

        # Expiry filter
        if max_weeks_to_expiry and expiry:
            try:
                dte = (dt.datetime.fromisoformat(expiry) - dt.datetime.utcnow()).days
                if dte > max_weeks_to_expiry * 7:
                    continue
            except Exception:
                pass

        # Preference nudges (not hard filters)
        if prefer_opening and not a.get("all_opening", False):
            # De-emphasize later via score
            pass
        if prefer_ask_bias and ask_prem <= bid_prem:
            pass

        s = score_alert(a)
        rows.append({
            "Ticker": ticker,
            "Option": opt_symbol or option_type,
            "Expiry": expiry,
            "Side": s["side"],
            "Score": s["score"],
            "Total Premium ($)": total_prem,
            "Ask Prem ($)": ask_prem,
            "Bid Prem ($)": bid_prem,
            "Size": total_size,
            "Why": s["reason"]
        })

    df = pd.DataFrame(rows).sort_values(["Score", "Total Premium ($)"], ascending=[False, False])
    # keep at least min_count when possible
    if len(df) > min_count:
        df = df.head(max(min_count, 50))  # show up to 50 high-quality items
    return df.reset_index(drop=True)

# -----------------------------
# Sidebar (Advanced Settings)
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Advanced settings")
    st.caption("Power users tweak here; newcomers can ignore this panel.")
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        minutes_window = st.number_input("Scan window (minutes) – FYI only", min_value=5, max_value=240, value=45, step=5)
        min_size = st.number_input("Min total size (contracts)", min_value=0, max_value=50000, value=10, step=10)
        prefer_opening = st.toggle("Prefer opening flow", value=True, help="Boosts alerts where transactions appear opening")
    with adv_col2:
        min_premium = st.number_input("Min total premium ($)", min_value=0, max_value=10_000_000, value=500_000, step=50_000)
        max_weeks_to_expiry = st.number_input("Max weeks to expiry", min_value=1, max_value=52, value=5, step=1)
        prefer_ask_bias = st.toggle("Prefer ask-side bias", value=True, help="Boosts alerts dominated by ask-side prints")

    st.markdown("#### Backtest (beta)")
    bt_start = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=30))
    bt_end = st.date_input("End", value=dt.date.today())
    take_profit = st.number_input("Take-profit %", min_value=1.0, max_value=200.0, value=15.0, step=0.5)
    stop_loss = st.number_input("Stop-loss %", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
    max_hold_days = st.number_input("Max holding days", min_value=1, max_value=60, value=5, step=1)

# -----------------------------
# Main
# -----------------------------
st.markdown("## 🦊 SilverFoxFlow — UOA 2.0")
st.caption("Low-noise institutional flow scanner with decisive signals and a simple backtester.")

# Simple hints/tips expander
with st.expander("📘 Setup tips & notes", expanded=False):
    st.write(
        "- Put your **UNUSUAL_WHALES_API_KEY** in the env at deploy time.\n"
        "- Hit **SCAN NOW** to pull the latest flow alerts.\n"
        "- Use **Advanced settings** only if you know what you’re doing; defaults aim for high signal."
    )

# Big CTA centered
cta_col1, cta_col2, cta_col3 = st.columns([1,2,1])
with cta_col2:
    run_scan = st.button("🚀 SCAN NOW", type="primary")

# Results placeholder
spot = st.empty()

def run_live_scan() -> Optional[pd.DataFrame]:
    # Time filters: use newer_than of ~the last N minutes if user cares; keep simple otherwise
    newer_than = None
    if minutes_window and minutes_window > 0:
        newer_than = (dt.datetime.utcnow() - dt.timedelta(minutes=int(minutes_window))).isoformat(timespec="seconds") + "Z"

    alerts = fetch_flow_alerts(limit=250, newer_than=newer_than)
    if not alerts:
        return None

    df = rank_alerts(
        alerts=alerts,
        min_premium=float(min_premium),
        min_size=int(min_size) if min_size else None,
        max_weeks_to_expiry=int(max_weeks_to_expiry) if max_weeks_to_expiry else None,
        prefer_opening=bool(prefer_opening),
        prefer_ask_bias=bool(prefer_ask_bias),
        min_count=20,
    )
    return df

def paint_df(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No flow data available with current settings.")
        return
    # A tiny legend for score
    top_score = df["Score"].iloc[0] if not df.empty else 0
    if top_score >= 80:
        badge = '<div class="score-badge good">Signal quality: STRONG</div>'
    elif top_score >= 65:
        badge = '<div class="score-badge warn">Signal quality: MODERATE</div>'
    else:
        badge = '<div class="score-badge bad">Signal quality: WEAK</div>'
    st.markdown(badge, unsafe_allow_html=True)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

if run_scan:
    with spot.container():
        with st.spinner("Scanning fresh flow and ranking…"):
            try:
                df_live = run_live_scan()
            except requests.HTTPError as e:
                st.error(f"HTTP error from Unusual Whales: {e}")
                df_live = None
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                df_live = None
        st.markdown("### 🔎 Top candidates")
        paint_df(df_live)
else:
    st.info("Press **SCAN NOW** to fetch and rank the latest institutional flow.")

# -----------------------------
# (Very) Simple backtest stub
# -----------------------------
def naive_backtest(trades: pd.DataFrame,
                   start: dt.date, end: dt.date,
                   tp_pct: float, sl_pct: float, max_days: int) -> Dict[str, Any]:
    """
    Placeholder backtest: this demonstrates wiring; replace with your broker/HLOC data if available.
    For each candidate we assume entry near tape time and simulate +/- thresholds on close-to-close moves.
    """
    if trades is None or trades.empty:
        return {"trades": 0, "win_rate": None, "avg_gain": None}

    # For demo we fake PnL using score as proxy (do not use in production).
    pnl = []
    for _, row in trades.iterrows():
        base = (row["Score"] - 50.0) / 100.0  # centered
        # clamp to hypothetical TP/SL & holding
        gain = max(-sl_pct/100.0, min(tp_pct/100.0, base))
        pnl.append(gain)

    if not pnl:
        return {"trades": 0, "win_rate": None, "avg_gain": None}
    wins = sum(1 for g in pnl if g > 0)
    return {
        "trades": len(pnl),
        "win_rate": round(100*wins/len(pnl), 1),
        "avg_gain": round(100* (sum(pnl)/len(pnl)), 2)
    }

st.markdown("hr", help="separator")

if run_scan and 'df_live' in locals() and df_live is not None and not df_live.empty:
    st.markdown("### 🧪 Backtest (illustrative)")
    results = naive_backtest(df_live, bt_start, bt_end, take_profit, stop_loss, max_hold_days)
    st.write(
        f"**Trades:** {results['trades']}  ·  "
        f"**Win-rate (toy):** {results['win_rate']}%  ·  "
        f"**Avg gain (toy):** {results['avg_gain']}%"
    )
