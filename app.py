# SilverFoxFlow — UOA 2.0 Scanner with Backtester
# -------------------------------------------------
# Streamlit app that ingests Unusual Whales (UW) flow, applies low-noise institutional filters,
# computes a Smart Money Score, and outputs decisive trade verdicts: BUY CALLS / BUY PUTS / NO TRADE.
# Includes a lightweight backtester on underlying price moves (proxy for options P/L).
#
# 🔧 How to run locally:
#   1) pip install streamlit pandas numpy requests pydantic yfinance python-dateutil
#   2) export UNUSUAL_WHALES_API_KEY="<your_key>"  (or set in .streamlit/secrets.toml as UW_API_KEY)
#   3) streamlit run app.py
#
# 🧠 Backtest method (simple, transparent):
#   - Uses yfinance to pull underlying OHLC.
#   - A signal is generated per ticker when the scanner says BUY CALLS or BUY PUTS.
#   - Entry = close price of the signal day (configurable intraday proxy, but close is robust).
#   - Exit rule (configurable): take-profit % or stop-loss % OR fixed holding period.
#   - Win/Loss and expectancy computed across signals.
#
# 📝 Notes:
#   - The exact Unusual Whales REST endpoints and field names vary by plan; this file provides a
#     robust fetch layer you can adapt in `fetch_uw_flows_api()`.
#   - If API fetch fails, you can upload a CSV export from UW with plausible columns; the mapper
#     tries to normalize columns (ticker, side, price, premium, expiry, is_sweep, is_block,
#     at_ask_flag, below_bid_flag, size, volume, open_interest, timestamp).
#   - The goal is to keep your logic deterministic and auditable for your users.

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dateutil import tz

# =========================
# ----- CONFIG / UX -------
# =========================
st.set_page_config(
    page_title="SilverFoxFlow — UOA 2.0",
    page_icon="🦊",
    layout="wide",
)

# Default Filters (from your last spec)
WINDOW_MIN = 45          # rolling minutes of flow to consider in live mode
LOCK_MIN = 20            # grace/lock window to avoid chasing too-late flow
SHARE_ENTRY = 0.70       # threshold for decisive direction to recommend entry
SHARE_FORCE_FLIP = 0.78  # if opposite-side share > this, flip verdict
MIN_PREM = 3_000_000     # minimum total premium per ticker in window
MIN_PRINTS = 8           # minimum number of sweeps/blocks
MIN_AGGR_RATIO = 0.60    # prints at/above ask (calls) or below bid (puts)
EXP_MIN_W, EXP_MAX_W = 1, 6  # near-term: 1–6 weeks to expiry
REQ_VOL_GT_OI = True
TOP_N = 10               # show top 10 tickers

# Runtime-configurable knobs gathered from the sidebar (no `global` needed)
CFG = {
    "MIN_PREM": MIN_PREM,
    "MIN_PRINTS": MIN_PRINTS,
    "MIN_AGGR_RATIO": MIN_AGGR_RATIO,
    "EXP_MIN_W": EXP_MIN_W,
    "EXP_MAX_W": EXP_MAX_W,
}


# =========================
# ---- Helper Utilities ----
# =========================

def _to_dt(x) -> datetime:
    if isinstance(x, (datetime, np.datetime64, pd.Timestamp)):
        return pd.to_datetime(x).to_pydatetime()
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)


def time_weight(minutes_old: float) -> float:
    if minutes_old < 10:
        return 1.0
    if minutes_old < 30:
        return 0.7
    return 0.5


def weeks_to_expiry(expiry_dt: datetime, now_dt: datetime) -> float:
    d = (expiry_dt - now_dt).days + (expiry_dt - now_dt).seconds/86400
    return max(0.0, d / 7.0)


def eligible(row: pd.Series, now_dt: datetime) -> bool:
    w = weeks_to_expiry(_to_dt(row.get("expiry")), now_dt)
    if not (CFG["EXP_MIN_W"] <= w <= CFG["EXP_MAX_W"]):
        return False
    if REQ_VOL_GT_OI:
        vol = row.get("volume", np.nan)
        oi = row.get("open_interest", np.nan)
        if pd.notna(vol) and pd.notna(oi) and not (vol > oi):
            return False
    return True


@dataclass
class FlowAggregate:
    ticker: str
    call_prem: float
    put_prem: float
    call_aggr_ratio: float  # fraction of call prints at/above ask
    put_aggr_ratio: float   # fraction of put prints below bid
    prints: int
    unique_strikes: int
    unique_exps: int
    recency_score: float
    total_premium: float

    def net_direction(self) -> float:
        # +1 = all calls, -1 = all puts
        tot = self.call_prem + self.put_prem + 1e-9
        return (self.call_prem - self.put_prem) / tot


# =========================
# -- Fetch & Normalize UW --
# =========================

@st.cache_data(show_spinner=False, ttl=120)
def fetch_uw_flows_api(start_dt: datetime, end_dt: datetime, api_key: Optional[str]) -> pd.DataFrame:
    """
    Fetch recent UW option flow between start_dt and end_dt.
    This is a placeholder you can wire to the official Unusual Whales endpoint you use.

    Expected fields in the returned DataFrame (or will be mapped by normalize_uw_columns):
      - ticker, side (CALL/PUT), price, premium, expiry, is_sweep, is_block,
        at_ask_flag, below_bid_flag, size, volume, open_interest, timestamp
    """
    if not api_key:
        return pd.DataFrame()

    # TODO: Replace with your actual UW endpoint + params.
    # The below is a schematic example — adjust path/params to match your plan.
    try:
        url = "https://api.unusualwhales.com/flow"  # <-- placeholder
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "start": start_dt.replace(tzinfo=timezone.utc).isoformat(),
            "end": end_dt.replace(tzinfo=timezone.utc).isoformat(),
            "limit": 5000,
        }
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.json_normalize(data)
        return df
    except Exception as e:
        st.warning(f"UW API fetch failed: {e}. You can upload a CSV export instead.")
        return pd.DataFrame()


def normalize_uw_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Try to map common/guessable column names from UW exports to our canonical schema.
    colmap_candidates = {
        "ticker": ["ticker", "symbol", "underlying"],
        "side": ["side", "type", "option_type"],  # CALL / PUT
        "price": ["price", "fill_price", "trade_price"],
        "premium": ["premium", "trade_value", "notional"],
        "expiry": ["expiry", "expiration", "exp", "expiration_date"],
        "is_sweep": ["is_sweep", "sweep"],
        "is_block": ["is_block", "block"],
        "at_ask_flag": ["at_ask_flag", "at_ask", "above_ask", "ask_aggr"],
        "below_bid_flag": ["below_bid_flag", "below_bid", "bid_aggr"],
        "size": ["size", "contracts"],
        "volume": ["volume", "vol"],
        "open_interest": ["open_interest", "oi"],
        "timestamp": ["timestamp", "time", "datetime"],
        "strike": ["strike", "strike_price"],
    }

    renamed = {}
    for target, opts in colmap_candidates.items():
        for o in opts:
            if o in df.columns:
                renamed[o] = target
                break

    out = df.rename(columns=renamed).copy()

    # sanitize values
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    out["premium"] = pd.to_numeric(out["premium"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["size"] = pd.to_numeric(out["size"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")

    # Side normalization
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out.loc[~out["side"].isin(["CALL", "PUT"]), "side"] = np.nan

    # Boolean flags
    for b in ["is_sweep", "is_block", "at_ask_flag", "below_bid_flag"]:
        if b in out.columns:
            out[b] = out[b].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        else:
            out[b] = False

    # Drop rows missing essentials
    essentials = ["ticker", "side", "premium", "expiry", "timestamp"]
    out = out.dropna(subset=essentials)

    return out


# =========================
# ---- Scoring Functions ---
# =========================

def aggregate_flows(df: pd.DataFrame, now_dt: datetime) -> List[FlowAggregate]:
    if df.empty:
        return []

    aggs: List[FlowAggregate] = []

    # Restrict to near-term, eligible rows
    elig = df[df.apply(lambda r: eligible(r, now_dt), axis=1)].copy()
    if elig.empty:
        return []

    # Compute recency weights per print
    elig["minutes_old"] = (now_dt - pd.to_datetime(elig["timestamp"]).dt.tz_convert(None)).dt.total_seconds() / 60.0
    elig["w"] = elig["minutes_old"].apply(time_weight)

    # Agg per ticker
    for tk, g in elig.groupby("ticker"):
        prints = len(g)
        unique_strikes = g["strike"].nunique(dropna=True)
        unique_exps = g["expiry"].nunique(dropna=True)

        # Premium by side (weighted by recency)
        call_prem = (g.loc[g["side"] == "CALL", "premium"] * g.loc[g["side"] == "CALL", "w"]).sum()
        put_prem  = (g.loc[g["side"] == "PUT", "premium"] * g.loc[g["side"] == "PUT", "w"]).sum()

        # Aggression ratios
        def safe_ratio(numer: pd.Series, denom: pd.Series) -> float:
            num = (numer * g.loc[numer.index, "w"]).sum()
            den = (denom * g.loc[denom.index, "w"]).sum() + 1e-9
            return float(num / den)

        call_aggr = safe_ratio(g["at_ask_flag"] & (g["side"] == "CALL"), g["side"] == "CALL")
        put_aggr  = safe_ratio(g["below_bid_flag"] & (g["side"] == "PUT"), g["side"] == "PUT")

        recency_score = float(g["w"].mean())
        total_premium = call_prem + put_prem

        aggs.append(FlowAggregate(
            ticker=tk,
            call_prem=call_prem,
            put_prem=put_prem,
            call_aggr_ratio=call_aggr,
            put_aggr_ratio=put_aggr,
            prints=prints,
            unique_strikes=int(unique_strikes),
            unique_exps=int(unique_exps),
            recency_score=recency_score,
            total_premium=total_premium,
        ))

    return aggs


def smart_money_score(agg: FlowAggregate, prem_median: float) -> float:
    # Components (0–100 total)
    # 1) Premium heft (0–35): log-scaled vs universe median
    heft = 35 * np.tanh(np.log1p(agg.total_premium) / (np.log1p(prem_median + 1e-9) + 1e-9))

    # 2) Directional imbalance (0–25)
    dir_strength = 25 * abs(agg.net_direction())

    # 3) Aggression (0–20)
    call_aggr = agg.call_aggr_ratio if agg.net_direction() >= 0 else 0
    put_aggr = agg.put_aggr_ratio if agg.net_direction() < 0 else 0
    aggr_component = 20 * max(call_aggr, put_aggr)

    # 4) Clustering / prints (0–10)
    cluster = 10 * np.tanh((agg.prints + agg.unique_strikes + agg.unique_exps) / 20)

    # 5) Recency (0–10)
    recent = 10 * agg.recency_score

    score = float(heft + dir_strength + aggr_component + cluster + recent)
    return max(0.0, min(100.0, score))


def verdict_from_agg(agg: FlowAggregate) -> Tuple[str, float, float]:
    """Returns (verdict, share_calls, share_puts)."""
    tot = agg.call_prem + agg.put_prem + 1e-9
    share_calls = agg.call_prem / tot
    share_puts = agg.put_prem / tot

    if tot < CFG["MIN_PREM"] or agg.prints < CFG["MIN_PRINTS"]:
        return ("NO TRADE", share_calls, share_puts)

    if agg.net_direction() > 0:  # leaning calls
        if share_calls >= SHARE_ENTRY and agg.call_aggr_ratio >= CFG["MIN_AGGR_RATIO"]:
            return ("BUY CALLS", share_calls, share_puts)
    else:  # leaning puts
        if share_puts >= SHARE_ENTRY and agg.put_aggr_ratio >= CFG["MIN_AGGR_RATIO"]:
            return ("BUY PUTS", share_calls, share_puts)

    # Force flip if one side is overwhelming
    if share_calls >= SHARE_FORCE_FLIP and agg.call_aggr_ratio >= CFG["MIN_AGGR_RATIO"]:
        return ("BUY CALLS", share_calls, share_puts)
    if share_puts >= SHARE_FORCE_FLIP and agg.put_aggr_ratio >= CFG["MIN_AGGR_RATIO"]:
        return ("BUY PUTS", share_calls, share_puts)

    return ("NO TRADE", share_calls, share_puts)

    if agg.net_direction() > 0:  # leaning calls
        if share_calls >= SHARE_ENTRY and agg.call_aggr_ratio >= MIN_AGGR_RATIO:
            return ("BUY CALLS", share_calls, share_puts)
    else:  # leaning puts
        if share_puts >= SHARE_ENTRY and agg.put_aggr_ratio >= MIN_AGGR_RATIO:
            return ("BUY PUTS", share_calls, share_puts)

    # Force flip if one side is overwhelming
    if share_calls >= SHARE_FORCE_FLIP and agg.call_aggr_ratio >= MIN_AGGR_RATIO:
        return ("BUY CALLS", share_calls, share_puts)
    if share_puts >= SHARE_FORCE_FLIP and agg.put_aggr_ratio >= MIN_AGGR_RATIO:
        return ("BUY PUTS", share_calls, share_puts)

    return ("NO TRADE", share_calls, share_puts)


# =========================
# --------- UI -------------
# =========================

st.title("🦊 SilverFoxFlow — UOA 2.0")
st.caption("Low-noise institutional flow scanner with decisive signals and a simple backtester.")

with st.sidebar:
    st.header("Configuration")

    # API key input — env or secrets preferred
    api_key = (
        st.secrets.get("UW_API_KEY")
        if hasattr(st, "secrets") and "UW_API_KEY" in st.secrets
        else os.getenv("UNUSUAL_WHALES_API_KEY")
    )
    api_key = st.text_input("Unusual Whales API Key", value=api_key or "", type="password")

    mode = st.radio("Data source", ["Unusual Whales API", "Upload CSV"], index=0 if api_key else 1)

    colA, colB = st.columns(2)
    with colA:
        window_min = st.number_input("Window (minutes)", min_value=10, max_value=240, value=WINDOW_MIN, step=5)
        min_prem = st.number_input("Min total premium ($)", min_value=0, value=MIN_PREM, step=500000, format="%d")
        min_prints = st.number_input("Min prints", min_value=0, value=MIN_PRINTS, step=1)
    with colB:
        min_aggr = st.slider("Min aggression ratio", 0.0, 1.0, MIN_AGGR_RATIO, 0.01)
        exp_min = st.number_input("Min weeks to expiry", min_value=0, max_value=52, value=EXP_MIN_W)
        exp_max = st.number_input("Max weeks to expiry", min_value=1, max_value=52, value=EXP_MAX_W)

    global MIN_PREM, MIN_PRINTS, MIN_AGGR_RATIO, EXP_MIN_W, EXP_MAX_W
    MIN_PREM = int(min_prem)
    MIN_PRINTS = int(min_prints)
    MIN_AGGR_RATIO = float(min_aggr)
    EXP_MIN_W, EXP_MAX_W = int(exp_min), int(exp_max)

    st.markdown("---")
    st.subheader("Backtest Settings")
    bt_start = st.date_input("Backtest start", value=(datetime.now().date() - timedelta(days=30)))
    bt_end = st.date_input("Backtest end", value=datetime.now().date())
    take_profit = st.number_input("Take-profit %", min_value=1.0, max_value=200.0, value=15.0, step=0.5)
    stop_loss = st.number_input("Stop-loss %", min_value=1.0, max_value=200.0, value=8.0, step=0.5)
    hold_days = st.number_input("Max holding days", min_value=1, max_value=30, value=5, step=1)

    st.markdown("---")
    do_scan = st.button("🚀 SCAN NOW", use_container_width=True)

# Data ingest
now_local = datetime.now(tz=tz.tzlocal())
start_dt = now_local - timedelta(minutes=window_min)

raw_df = pd.DataFrame()
if do_scan:
    if mode == "Unusual Whales API":
        raw_df = fetch_uw_flows_api(start_dt, now_local, api_key)
    else:
        st.info("Upload a CSV export from Unusual Whales (flow).")
        uploaded = st.file_uploader("Upload UW CSV", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)

    df = normalize_uw_columns(raw_df)

    if df.empty:
        st.warning("No flow data available with current settings.")
    else:
        st.subheader("Raw Flow (normalized)")
        st.dataframe(df.head(500))

        # Aggregate & rank
        aggs = aggregate_flows(df, now_local)
        if not aggs:
            st.warning("No eligible flow after filters.")
        else:
            prem_median = np.median([a.total_premium for a in aggs]) or 1.0
            rows = []
            for a in aggs:
                score = smart_money_score(a, prem_median)
                verdict, sc, sp = verdict_from_agg(a)
                rows.append({
                    "Ticker": a.ticker,
                    "Score": round(score, 1),
                    "Verdict": verdict,
                    "Calls %": round(sc*100, 1),
                    "Puts %": round(sp*100, 1),
                    "Agg(Call)": round(a.call_aggr_ratio*100,1),
                    "Agg(Put)": round(a.put_aggr_ratio*100,1),
                    "Prints": a.prints,
                    "Strikes": a.unique_strikes,
                    "Exps": a.unique_exps,
                    "Total Prem ($)": int(a.total_premium),
                })

            out = pd.DataFrame(rows).sort_values(["Verdict", "Score", "Total Prem ($)"], ascending=[True, False, False])
            # Rank by score within BUY CALLS / BUY PUTS first
            calls_df = out[out["Verdict"] == "BUY CALLS"].sort_values("Score", ascending=False).head(TOP_N)
            puts_df = out[out["Verdict"] == "BUY PUTS"].sort_values("Score", ascending=False).head(TOP_N)
            no_df = out[out["Verdict"] == "NO TRADE"].sort_values("Score", ascending=False).head(TOP_N)

            st.markdown("### ✅ Top Signals")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📈 BUY CALLS")
                st.dataframe(calls_df.reset_index(drop=True))
            with c2:
                st.markdown("#### 📉 BUY PUTS")
                st.dataframe(puts_df.reset_index(drop=True))

            with st.expander("Other tickers (No Trade)"):
                st.dataframe(no_df.reset_index(drop=True))

            st.success("Signals generated. Use Backtest below to validate.")

            st.markdown("---")
            st.header("🔙 Backtest Signals (Underlying Proxy)")
            st.caption("Backtest evaluates whether the underlying moved favorably after signals within your take-profit/stop-loss or holding window. This is a proxy for options performance.")

            # Build signal list (today only) for display & optional backtest range below
            signal_rows = pd.concat([
                calls_df.assign(_dir="CALLS"),
                puts_df.assign(_dir="PUTS"),
            ], ignore_index=True)
            st.dataframe(signal_rows[["Ticker", "Verdict", "Score", "Total Prem ($)", "Calls %", "Puts %"]])

            do_backtest = st.button("🏁 Run Backtest on Range", use_container_width=True)
            if do_backtest:
                # Backtest over bt_start..bt_end using daily signals recomputed for each day.
                res = run_backtest_range(
                    api_key=api_key if mode == "Unusual Whales API" else None,
                    start_date=pd.to_datetime(bt_start).date(),
                    end_date=pd.to_datetime(bt_end).date(),
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    hold_days=hold_days,
                )
                render_backtest(res)


# =========================
# ----- Backtest Engine ----
# =========================

@st.cache_data(show_spinner=True)
def daily_signals_for(date_obj, api_key: Optional[str]) -> pd.DataFrame:
    """Recompute signals for a historical date using entire regular-session window.
    For simplicity, we approximate by using that day's 09:30–15:30 ET flow window.
    (Adjust to your liking.)
    """
    # Convert date to 13:30–21:00 UTC (US market) — approximate
    start = datetime.combine(pd.to_datetime(date_obj).to_pydatetime(), datetime.min.time()).replace(hour=13, minute=30, tzinfo=timezone.utc)
    end = start.replace(hour=21, minute=0)

    df = fetch_uw_flows_api(start, end, api_key) if api_key else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Verdict", "Score"])  # nothing that day

    df = normalize_uw_columns(df)
    aggs = aggregate_flows(df, end)
    if not aggs:
        return pd.DataFrame(columns=["Ticker", "Verdict", "Score"])  # none eligible

    prem_median = np.median([a.total_premium for a in aggs]) or 1.0
    rows = []
    for a in aggs:
        score = smart_money_score(a, prem_median)
        verdict, sc, sp = verdict_from_agg(a)
        if verdict in ("BUY CALLS", "BUY PUTS"):
            rows.append({"Ticker": a.ticker, "Verdict": verdict, "Score": round(score,1)})

    out = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return out.head(TOP_N)


def simulate_trade(ticker: str, verdict: str, signal_date: datetime, take_profit: float, stop_loss: float, hold_days: int) -> dict:
    """Simulate trade on underlying: entry=close at signal_date, exit by TP/SL or max hold.
    Returns dict with outcome metrics.
    """
    try:
        hist = yf.download(ticker, start=signal_date - timedelta(days=3), end=signal_date + timedelta(days=hold_days+7), progress=False)
    except Exception:
        hist = pd.DataFrame()

    if hist.empty or ticker not in yf.Tickers(ticker).tickers:
        return {"ticker": ticker, "ok": False, "reason": "No price data"}

    # Align to dates
    df = hist.copy()
    df.index = pd.to_datetime(df.index).date

    if signal_date.date() not in df.index:
        # Entry = next trading day close
        trade_dates = sorted(df.index)
        future = [d for d in trade_dates if d >= signal_date.date()]
        if not future:
            return {"ticker": ticker, "ok": False, "reason": "No session after signal"}
        entry_day = future[0]
    else:
        entry_day = signal_date.date()

    entry_price = float(df.loc[entry_day, "Close"])

    # Iterate forward until TP/SL/hold_days reached
    trade_dates = sorted([d for d in df.index if d >= entry_day])
    exit_day = trade_dates[min(hold_days-1, len(trade_dates)-1)]
    exit_price = float(df.loc[exit_day, "Close"])

    # Track intraday extremes for earlier exits
    for d in trade_dates:
        high = float(df.loc[d, "High"]) if "High" in df.columns else float(df.loc[d, "Close"]) 
        low = float(df.loc[d, "Low"]) if "Low" in df.columns else float(df.loc[d, "Close"]) 
        change_up = (high - entry_price) / entry_price * 100
        change_dn = (low - entry_price) / entry_price * 100
        if verdict == "BUY CALLS" and change_up >= take_profit:
            exit_day, exit_price = d, entry_price * (1 + take_profit/100)
            break
        if verdict == "BUY PUTS" and (-change_dn) >= take_profit:  # price down equals profit for puts
            exit_day, exit_price = d, entry_price * (1 - take_profit/100)
            break
        if verdict == "BUY CALLS" and change_dn <= -stop_loss:
            exit_day, exit_price = d, entry_price * (1 - stop_loss/100)
            break
        if verdict == "BUY PUTS" and (-change_up) <= -stop_loss:  # price up equals loss for puts
            exit_day, exit_price = d, entry_price * (1 + stop_loss/100)
            break

    ret_pct = (exit_price - entry_price) / entry_price * (100 if verdict == "BUY CALLS" else -100)

    return {
        "ticker": ticker,
        "ok": True,
        "entry_day": entry_day,
        "exit_day": exit_day,
        "entry": round(entry_price, 4),
        "exit": round(exit_price, 4),
        "return_%": round(ret_pct, 2),
        "verdict": verdict,
    }


def run_backtest_range(api_key: Optional[str], start_date, end_date, take_profit: float, stop_loss: float, hold_days: int):
    cur = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_trades = []
    while cur <= end:
        sigs = daily_signals_for(cur.date(), api_key)
        for _, r in sigs.iterrows():
            trade = simulate_trade(
                ticker=r["Ticker"],
                verdict=r["Verdict"],
                signal_date=pd.to_datetime(cur),
                take_profit=take_profit,
                stop_loss=stop_loss,
                hold_days=hold_days,
            )
            if trade.get("ok"):
                all_trades.append(trade)
        cur += timedelta(days=1)

    return pd.DataFrame(all_trades)


def render_backtest(bt: pd.DataFrame):
    if bt.empty:
        st.warning("No trades simulated in the selected range (maybe no signals or data).")
        return

    st.subheader("Backtest Results")
    st.dataframe(bt)

    wins = (bt["return_%"] > 0).sum()
    losses = (bt["return_%"] <= 0).sum()
    winrate = wins / max(1, (wins + losses)) * 100
    avg_win = bt.loc[bt["return_%"] > 0, "return_%"].mean() if wins else 0.0
    avg_loss = bt.loc[bt["return_%"] <= 0, "return_%"].mean() if losses else 0.0
    expectancy = (winrate/100) * avg_win + (1 - winrate/100) * avg_loss

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", len(bt))
    c2.metric("Win rate", f"{winrate:.1f}%")
    c3.metric("Avg win %", f"{avg_win:.2f}%")
    c4.metric("Avg loss %", f"{avg_loss:.2f}%")

    st.metric("Expectancy per trade", f"{expectancy:.2f}%")

    # Per-ticker summary
    st.markdown("#### Per-Ticker Performance")
    per_tk = bt.groupby("ticker")["return_%"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
    st.dataframe(per_tk)


# =========================
# --- Footer / Guidance ----
# =========================
with st.expander("ℹ️ Setup tips & troubleshooting"):
    st.markdown(
        """
        **API wiring:** Replace the placeholder URL in `fetch_uw_flows_api()` with your actual Unusual Whales endpoint.
        If you hit auth or schema errors, upload a CSV export temporarily and verify the pipeline.

        **Secrets:** Add your key in Streamlit Cloud under *App → Settings → Secrets* as `UW_API_KEY`.

        **Why underlying proxy for backtest?** Option-level historical bid/ask is expensive and plan-specific.
        Using the underlying with TP/SL approximates directional edge honestly. If you prefer, extend `simulate_trade`
        to fetch option chains and track a specific strike/expiry.

        **Strict filters:** Tune \"Min total premium\", \"Min prints\", \"Min aggression\" to tighten noise. Keep expiries
        near-term (1–6 weeks) as specified in UOA 2.0.
        """
    )
