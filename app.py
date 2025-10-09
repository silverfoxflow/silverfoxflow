# SilverFoxFlow — Recommended Trades + UI Polish
# UOA 2.0 scanner using Unusual Whales Flow Alerts + Recommendations + Backtester (toy)
# Design: Kid‑simple main (SCAN → Results → Recommended trades), advanced knobs in sidebar.
# ---------------------------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas requests yfinance
# Secrets/Env:
#   UNUSUAL_WHALES_API_KEY in Streamlit Secrets or environment variables

from __future__ import annotations
import os
import math
import json
import traceback
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# =========================
# ---- Page & Styles --------
# =========================
st.set_page_config(page_title="🦊 SilverFoxFlow — UOA 2.0", page_icon="🦊", layout="wide")

st.markdown(
    """
    <style>
      :root { --bg:#0f1117; --panel:#171a23; --muted:#9aa3b2; --text:#e7e9ef; --green:#22c55e; --red:#ef4444; --amber:#f59e0b; --accent:#fb7185; }
      html, body, [class*="css"], .stApp { font-size: 16px; background: var(--bg); color: var(--text); }
      section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid #202636; }
      .title { text-align:center; font-size: 32px; font-weight: 800; margin: 6px 0 2px; }
      .sub { text-align:center; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
      .cta button { width: 280px; height: 56px; border-radius: 14px !important; font-weight: 800; font-size: 18px; box-shadow: 0 6px 18px rgba(255,99,99,.12); }
      .cta button:hover { box-shadow: 0 8px 22px rgba(255,99,99,.18); }
      .chip { display:inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; border:1px solid #2a3245; }
      .chip.green { background: rgba(34,197,94,.12); color: var(--green); border-color: rgba(34,197,94,.35); }
      .chip.red { background: rgba(239,68,68,.12); color: var(--red); border-color: rgba(239,68,68,.35); }
      .chip.gray { background: rgba(148,163,184,.14); color: #94a3b8; border-color: rgba(148,163,184,.35); }
      .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:800; font-size:12px; }
      .b-strong { background:#103b28; color:#2cd199; }
      .b-mod { background:#3a2e12; color:#f7c257; }
      .b-weak { background:#3b1a1a; color:#ff8c8c; }
      .status { color: var(--muted); font-size: 12px; margin-top: 2px; text-align:center; }
      .section-title { font-size: 18px; font-weight: 800; margin: 16px 0 8px; }
      .small { color: var(--muted); font-size: 12px; }
      thead tr th { text-align:left !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = "https://api.unusualwhales.com/api"
API_KEY = (
    st.secrets.get("UW_API_KEY") if hasattr(st, "secrets") and "UW_API_KEY" in st.secrets else os.getenv("UNUSUAL_WHALES_API_KEY")
)

# =========================
# ---- UW Client ------------
# =========================

def api_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


def http_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=api_headers(), params=params, timeout=30)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}


@st.cache_data(show_spinner=False, ttl=60)
def fetch_flow_alerts(limit: int = 250, newer_than_iso: Optional[str] = None) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": max(1, min(limit, 250))}
    if newer_than_iso:
        params["newer_than"] = newer_than_iso
    data = http_get("/option-trades/flow-alerts", params)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    return []

# =========================
# ---- Scoring --------------
# =========================

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _dte_days(expiry_iso: Optional[str]) -> Optional[int]:
    if not expiry_iso:
        return None
    try:
        return (datetime.fromisoformat(expiry_iso) - datetime.utcnow()).days
    except Exception:
        return None


def score_alert(a: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    total_prem = _safe_float(a.get("total_premium", a.get("premium")))
    ask_prem = _safe_float(a.get("total_ask_side_prem", a.get("ask_premium")))
    bid_prem = _safe_float(a.get("total_bid_side_prem", a.get("bid_premium")))
    option_type = (a.get("option_type") or a.get("type") or "").lower()
    expiry = a.get("expiry") or a.get("expiration")
    dte = _dte_days(expiry)

    net_ask_bias = ask_prem - bid_prem
    side = "CALL" if option_type == "call" or net_ask_bias >= 0 else "PUT"

    prem_term = math_tanh(total_prem / 250_000.0)
    ask_term = math_tanh(max(0.0, net_ask_bias) / (total_prem + 1e-9))
    if dte is None:
        dte_term = 0.5
    elif 7 <= dte <= 35:
        dte_term = 1.0
    elif 1 <= dte < 7:
        dte_term = 0.6
    elif 36 <= dte <= 90:
        dte_term = 0.7
    else:
        dte_term = 0.4

    open_only = 1.0 if a.get("all_opening", False) else 0.0
    sweepish = 1.0 if any(t in (a.get("tags") or []) for t in ["sweep", "intermarket_sweep", "i_sweep"]) else 0.0

    score = 0.40 * prem_term + 0.30 * ask_term + 0.20 * dte_term + 0.05 * open_only + 0.05 * sweepish
    out = {
        "total_premium": total_prem,
        "ask_bias": net_ask_bias,
        "dte": dte,
        "all_opening": bool(a.get("all_opening", False)),
        "is_sweep": bool(sweepish),
    }
    return round(100 * score, 1), side, out


def math_tanh(x: float) -> float:
    try:
        import math
        return math.tanh(x)
    except Exception:
        return 0.0


def rank_alerts(alerts: List[Dict[str, Any]],
                min_premium: float,
                min_size: int,
                max_weeks_to_expiry: int,
                prefer_opening: bool,
                prefer_ask_bias: bool,
                target_count: int = 20,
                cap: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for a in alerts:
        ticker = a.get("ticker") or a.get("underlying_symbol") or a.get("symbol") or ""
        opt_symbol = a.get("option_symbol") or ""
        expiry = a.get("expiry") or a.get("expiration")
        total_prem = _safe_float(a.get("total_premium", a.get("premium")))
        total_size = int(a.get("total_size", a.get("size", 0)) or 0)

        if total_prem < float(min_premium):
            continue
        if min_size and total_size < int(min_size):
            continue
        if max_weeks_to_expiry and expiry:
            dte = _dte_days(expiry)
            if dte is not None and dte > max_weeks_to_expiry * 7:
                continue

        score, side, extras = score_alert(a)
        if prefer_opening and not extras.get("all_opening"):
            score *= 0.95
        if prefer_ask_bias and extras.get("ask_bias", 0) <= 0:
            score *= 0.95

        rows.append({
            "Ticker": ticker,
            "Option": opt_symbol or ("CALL" if side == "CALL" else "PUT"),
            "Expiry": expiry or "",
            "Side": side,
            "Score": score,
            "Total Premium ($)": int(total_prem),
            "Ask Prem ($)": int(_safe_float(a.get("total_ask_side_prem", a.get("ask_premium")))),
            "Bid Prem ($)": int(_safe_float(a.get("total_bid_side_prem", a.get("bid_premium")))),
            "Size": total_size,
            "Why": f"prem={int(total_prem):,}, ask_bias={int(extras.get('ask_bias',0)):,}, dte={extras.get('dte')}"
        })

    df = pd.DataFrame(rows).sort_values(["Score", "Total Premium ($)"], ascending=[False, False]).reset_index(drop=True)

    # If fewer than target_count, don't force junk; just return what we have.
    if len(df) > cap:
        df = df.head(cap)
    return df

# =========================
# ---- Sidebar (Advanced) ---
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Advanced settings")
    st.caption("Power users tweak here; newcomers can ignore this panel.")

    minutes_window = st.number_input("Scan window (minutes)", min_value=5, max_value=240, value=45, step=5, help="Used as 'newer_than' filter for flow alerts")
    min_premium = st.number_input("Min total premium ($)", min_value=0, max_value=5_000_000, value=500_000, step=50_000)
    min_size = st.number_input("Min total size (contracts)", min_value=0, max_value=100_000, value=10, step=10)
    max_weeks_to_expiry = st.number_input("Max weeks to expiry", min_value=1, max_value=52, value=5, step=1)

    prefer_opening = st.toggle("Favor opening flow", value=True)
    prefer_ask_bias = st.toggle("Favor ask-side buying", value=True)

    st.markdown("#### Recommendations")
    rec_top_n = st.slider("Number of trade ideas", min_value=5, max_value=20, value=10)
    rec_weeks_min = st.number_input("Target min weeks (expiry)", min_value=1, max_value=6, value=1)
    rec_weeks_max = st.number_input("Target max weeks (expiry)", min_value=1, max_value=8, value=3)
    rec_otm_pct = st.slider("Moneyness (±% OTM)", min_value=1, max_value=15, value=4)

    st.markdown("#### Backtest (toy)")
    bt_start = st.date_input("Start", value=date.today() - timedelta(days=30))
    bt_end = st.date_input("End", value=date.today())
    take_profit = st.number_input("Take-profit %", min_value=1.0, max_value=200.0, value=15.0, step=0.5)
    stop_loss = st.number_input("Stop-loss %", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
    max_hold_days = st.number_input("Max holding days", min_value=1, max_value=30, value=5, step=1)

# =========================
# ---- Main (Kid‑simple) ----
# =========================
st.markdown('<div class="title">🦊 SilverFoxFlow — UOA 2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Tap SCAN to pull live flow from Unusual Whales. Green = calls, Red = puts, Gray = skip.</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,2,1])
with c2:
    scan_now = st.button("🚀 SCAN NOW", type="primary")

status = st.empty()

# ============== SCAN ===============
alerts: List[Dict[str, Any]] = []
ranked_df: Optional[pd.DataFrame] = None

if scan_now:
    if not API_KEY:
        st.error("No Unusual Whales API key found. Add UNUSUAL_WHALES_API_KEY in Secrets.")
    else:
        with st.spinner("Scanning flow alerts…"):
            try:
                newer_than = (datetime.utcnow() - timedelta(minutes=int(minutes_window))).strftime("%Y-%m-%dT%H:%M:%SZ")
                alerts = fetch_flow_alerts(limit=250, newer_than_iso=newer_than)
                fetched = len(alerts) if isinstance(alerts, list) else 0
                status.markdown(f"<div class='status'>Fetched <b>{fetched}</b> raw alerts from Unusual Whales • window: {minutes_window}m • {datetime.utcnow().strftime('%H:%M:%S')} UTC</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Fetch error: {e}")
                st.caption(traceback.format_exc())

        if alerts:
            ranked_df = rank_alerts(
                alerts=alerts,
                min_premium=min_premium,
                min_size=min_size,
                max_weeks_to_expiry=max_weeks_to_expiry,
                prefer_opening=prefer_opening,
                prefer_ask_bias=prefer_ask_bias,
                target_count=20,
            )

# ============ RESULTS TABLE ============
if ranked_df is not None:
    st.markdown("### 🔎 Top candidates")
    # quality badge
    top_score = ranked_df["Score"].max() if not ranked_df.empty else 0
    if top_score >= 80:
        st.markdown('<span class="badge b-strong">Signal quality: STRONG</span>', unsafe_allow_html=True)
    elif top_score >= 65:
        st.markdown('<span class="badge b-mod">Signal quality: MODERATE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge b-weak">Signal quality: WEAK</span>', unsafe_allow_html=True)

    # Side chips column
    disp = ranked_df.copy()
    disp["Verdict"] = disp["Side"].apply(lambda s: f"<span class='chip {'green' if s=='CALL' else 'red'}'>{'BUY ' + s}</span>")
    # Order columns
    order_cols = ["Ticker", "Verdict", "Score", "Expiry", "Option", "Total Premium ($)", "Ask Prem ($)", "Bid Prem ($)", "Size", "Why"]
    for c in order_cols:
        if c not in disp.columns:
            disp[c] = ""
    st.dataframe(disp[order_cols], use_container_width=True, hide_index=True)

# ============ RECOMMENDED TRADES ============

def nearest_friday_within_weeks(base: date, wmin: int, wmax: int) -> Optional[date]:
    cands = []
    for w in range(max(1, wmin), max(wmin, wmax) + 1):
        d = base + timedelta(days=7*w)
        # roll forward to Friday
        while d.weekday() != 4:
            d += timedelta(days=1)
        cands.append(d)
    return min(cands) if cands else None


def suggest_contracts(df: pd.DataFrame, top_n: int, weeks_min: int, weeks_max: int, otm_pct: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ideas = []
    picks = df.sort_values(["Score", "Total Premium ($)"], ascending=[False, False]).head(int(top_n))
    today = datetime.utcnow().date()

    for _, r in picks.iterrows():
        tk = str(r["Ticker"]).upper()
        side = str(r["Side"]).upper()
        try:
            last_close = float(yf.Ticker(tk).history(period="1d").iloc[-1]["Close"])  # safe fallback
        except Exception:
            last_close = None

        # Pick an expiry near 1–3 weeks (or within given range)
        target_exp = nearest_friday_within_weeks(today, weeks_min, weeks_max)
        exp_str = target_exp.isoformat() if target_exp else (str(r.get("Expiry", "")) or "")

        # Strike suggestion using OTM % if we have last price
        strike = None
        if last_close is not None:
            if side == "CALL":
                strike = round(last_close * (1 + otm_pct/100.0))
            else:
                strike = round(last_close * (1 - otm_pct/100.0))

        # Try to snap to real option chain if available
        chosen_strike = strike
        try:
            tkr = yf.Ticker(tk)
            # find closest expiry in chain
            exps = tkr.options or []
            if exps:
                # choose nearest to target_exp
                if target_exp:
                    closest = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - target_exp).days))
                else:
                    closest = exps[0]
                chain = tkr.option_chain(closest)
                tbl = chain.calls if side == "CALL" else chain.puts
                if strike is not None and not tbl.empty:
                    # snap to nearest strike
                    tbl = tbl.copy()
                    tbl["_diff"] = (tbl["strike"] - strike).abs()
                    row = tbl.sort_values("_diff").iloc[0]
                    chosen_strike = float(row["strike"])
                    exp_str = closest
        except Exception:
            pass

        ideas.append({
            "Ticker": tk,
            "Side": side,
            "Confidence": round(float(r["Score"]), 1),
            "Suggested Expiry": exp_str,
            "Suggested Strike": chosen_strike if chosen_strike is not None else "ATM",
            "Plan": "TP 15% / SL 8% / hold 3–5d",
            "Order": f"BUY 1x {tk} {exp_str} {int(chosen_strike) if isinstance(chosen_strike,(int,float)) else 'ATM'}{('C' if side=='CALL' else 'P')} @ MKT"
        })
    return pd.DataFrame(ideas)

if ranked_df is not None and not ranked_df.empty:
    st.markdown("### 🎯 Recommended trades (auto-generated)")
    st.caption("Advisory only. We select near-dated weeklies and ~±OTM strikes from the top signals.")
    rec_df = suggest_contracts(ranked_df, rec_top_n, rec_weeks_min, rec_weeks_max, rec_otm_pct)
    if rec_df.empty:
        st.info("Not enough clean signals for recommendations. Loosen filters or widen the scan window.")
    else:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        all_txt = "\n".join(rec_df["Order"].tolist())
        st.download_button("⬇️ Export orders (txt)", data=all_txt, file_name="silverfox_orders.txt")

# ============ Backtest (toy) ============

def naive_backtest(trades: pd.DataFrame, tp_pct: float, sl_pct: float) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {"trades": 0, "win_rate": None, "avg_gain": None}
    # Toy: map Score to pseudo returns
    pnl = []
    for _, row in trades.iterrows():
        base = (row["Confidence"] - 50.0) / 100.0  # center around 0
        gain = max(-sl_pct/100.0, min(tp_pct/100.0, base))
        pnl.append(gain)
    wins = sum(1 for g in pnl if g > 0)
    return {"trades": len(pnl), "win_rate": round(100*wins/len(pnl),1), "avg_gain": round(100*sum(pnl)/len(pnl),2)}

if ranked_df is not None and 'rec_df' in locals() and rec_df is not None and not rec_df.empty:
    st.markdown("### 🧪 Quick backtest (toy)")
    res = naive_backtest(rec_df, take_profit, stop_loss)
    st.write(f"**Trades:** {res['trades']} · **Win‑rate (toy):** {res['win_rate']}% · **Avg gain (toy):** {res['avg_gain']}%")

# ===== Footer tips =====
with st.expander("📘 Setup tips & notes", expanded=False):
    st.markdown(
        """
        • Source: **Unusual Whales** `/api/option-trades/flow-alerts` with `newer_than` window and Bearer auth.\
        • Filters are applied client‑side; if your tier exposes server‑side filters, we can add them to the request.\
        • Recommendations are **advisory**, using near‑dated expiries and ~±OTM strikes. Always confirm liquidity.\
        • Backtest is a **toy proxy**; for real P/L, wire historical bars and option chains per signal date.
        """
    )
