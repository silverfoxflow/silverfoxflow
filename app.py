import hashlib
from datetime import datetime
import pandas as pd
import streamlit as st

# =========================
# SIMPLE + STABLE CONFIG
# =========================
PRESETS = {
    "Strict (Pro)": dict(
        MIN_PREM=3_000_000,
        MIN_PRINTS=8,
        EXP_MIN_W=1, EXP_MAX_W=8,
        REQ_VOL_GT_OI=True,
        DOMINANCE=1.40,
        MIN_TOTAL_PREM=5_000_000,
        MIN_AVG_AGGR_CALL=0.60,
        MAX_AVG_AGGR_PUT=0.40
    ),
    "Balanced": dict(
        MIN_PREM=2_000_000,
        MIN_PRINTS=6,
        EXP_MIN_W=1, EXP_MAX_W=10,
        REQ_VOL_GT_OI=False,              # allow sleepers where vol<=OI
        DOMINANCE=1.30,
        MIN_TOTAL_PREM=3_000_000,
        MIN_AVG_AGGR_CALL=0.58,
        MAX_AVG_AGGR_PUT=0.42
    ),
    "Explorer (Catch Sleepers)": dict(
        MIN_PREM=1_000_000,
        MIN_PRINTS=4,
        EXP_MIN_W=1, EXP_MAX_W=16,
        REQ_VOL_GT_OI=False,
        DOMINANCE=1.20,
        MIN_TOTAL_PREM=2_000_000,
        MIN_AVG_AGGR_CALL=0.55,
        MAX_AVG_AGGR_PUT=0.45
    ),
}

KID_HELP = "We add up big-money option dollars per stock. If BULL $ >> BEAR $, we say BUY CALLS. If BEAR $ >> BULL $, we say BUY PUTS. If it’s close, NO TRADE."

# =========================
# BASIC APP SHELL
# =========================
st.set_page_config(page_title="SilverFoxFlow — Multi-Ticker UOA", page_icon="🦊", layout="wide")
st.title("🦊 SilverFoxFlow — EASY Whale Scanner (Multi-Ticker)")
st.caption("Kid-simple decisions per stock. " + KID_HELP)

mode = st.radio("Choose scan mode:", list(PRESETS.keys()), index=1, horizontal=True)
cfg = PRESETS[mode]

st.write("**Upload CSV** with columns: `timestamp, ticker, expiry_weeks, notional, prints, aggr_ratio, volume, open_interest`")

uploaded = st.file_uploader("Drop your options flow CSV here", type=["csv"])

# =========================
# HELPERS
# =========================
def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def _eligible_mask(df: pd.DataFrame) -> pd.Series:
    near_term = df["expiry_weeks"].fillna(0).between(cfg["EXP_MIN_W"], cfg["EXP_MAX_W"])
    prem_ok = df["notional"].fillna(0) >= cfg["MIN_PREM"]
    prints_ok = df["prints"].fillna(0) >= cfg["MIN_PRINTS"]
    aggr_ok = df["aggr_ratio"].fillna(0).between(0, 1)
    if cfg["REQ_VOL_GT_OI"]:
        vol_ok = df["volume"].fillna(0) > df["open_interest"].fillna(0)
    else:
        vol_ok = True
    return near_term & prem_ok & prints_ok & aggr_ok & vol_ok

def _stable_minutes_old(df: pd.DataFrame) -> pd.Series:
    # Stable recency: anchor to the latest timestamp in the file
    as_of = df["timestamp"].max()
    mins = (as_of - df["timestamp"]).dt.total_seconds() / 60.0
    return mins.clip(lower=0).fillna(60)

def _recency_weight(minutes_old: pd.Series) -> pd.Series:
    # Simple step function, stable because it’s relative to the uploaded data
    return pd.Series([1.0 if m < 10 else 0.7 if m < 30 else 0.5 for m in minutes_old], index=minutes_old.index)

def _ticker_summary(edf: pd.DataFrame) -> dict:
    """Summarize one ticker’s eligible flow and produce a kid-simple verdict."""
    if edf.empty:
        return dict(
            total_premium=0.0, bull_notional=0.0, bear_notional=0.0, dominance=1.0,
            avg_aggr=0.0, count=0, verdict="NO TRADE", why="No qualifying big-money flow."
        )

    edf = edf.copy()
    edf["minutes_old"] = _stable_minutes_old(edf)
    edf["w_recency"]  = _recency_weight(edf["minutes_old"])

    edf["bull_$"] = edf["notional"] * edf["aggr_ratio"] * edf["w_recency"]
    edf["bear_$"] = edf["notional"] * (1.0 - edf["aggr_ratio"]) * edf["w_recency"]

    bull = float(edf["bull_$"].sum())
    bear = float(edf["bear_$"].sum())
    total = float(edf["notional"].sum())
    avg_aggr = float(edf["aggr_ratio"].mean())
    count = int(len(edf))

    dominance = (bull / max(bear, 1e-9)) if bear > 0 else float("inf")

    # Decision
    if total < cfg["MIN_TOTAL_PREM"] or count < cfg["MIN_PRINTS"]:
        verdict = "NO TRADE"; why = "Not enough $ or prints from big players."
    else:
        if (dominance >= cfg["DOMINANCE"]) and (avg_aggr >= cfg["MIN_AVG_AGGR_CALL"]):
            verdict = "BUY CALLS"; why = "Bullish dollars dominate and buyers were aggressive."
        elif (dominance <= 1.0 / cfg["DOMINANCE"]) and (avg_aggr <= cfg["MAX_AVG_AGGR_PUT"]):
            verdict = "BUY PUTS";  why = "Bearish dollars dominate and sellers were aggressive."
        else:
            verdict = "NO TRADE";  why = "Flow is mixed or indecisive."

    return dict(
        total_premium=round(total, 2),
        bull_notional=round(bull, 2),
        bear_notional=round(bear, 2),
        dominance=round(dominance, 2) if dominance != float("inf") else float("inf"),
        avg_aggr=round(avg_aggr, 3),
        count=count,
        verdict=verdict,
        why=why
    )

def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()

# =========================
# MAIN FLOW
# =========================
if not uploaded:
    st.info("👉 Tip: Switch to **Explorer** mode to catch lesser-known tickers. Then upload your CSV.")
else:
    raw = pd.read_csv(uploaded)
    needed = {"timestamp","ticker","expiry_weeks","notional","prints","aggr_ratio","volume","open_interest"}
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    df_hash = _hash_df(raw)
    if "df_hash" not in st.session_state or st.session_state["df_hash"] != df_hash or "by_ticker" not in st.session_state:
        df = _parse_timestamps(raw)
        mask = _eligible_mask(df)
        edf = df.loc[mask].copy()

        # Group by ticker and compute per-ticker summaries
        results = []
        for tkr, g in (edf.groupby("ticker") if not edf.empty else []):
            summ = _ticker_summary(g)
            results.append(dict(
                ticker=str(tkr),
                decision=summ["verdict"],
                reason=summ["why"],
                total_premium=summ["total_premium"],
                bull_dollars=summ["bull_notional"],
                bear_dollars=summ["bear_notional"],
                dominance=("∞" if summ["dominance"] == float("inf") else summ["dominance"]),
                avg_aggr=summ["avg_aggr"],
                prints=summ["count"]
            ))

        by_ticker = pd.DataFrame(results).sort_values(
            ["decision","total_premium"], ascending=[True, False]
        ) if results else pd.DataFrame(columns=[
            "ticker","decision","reason","total_premium","bull_dollars","bear_dollars","dominance","avg_aggr","prints"
        ])

        # Also compute a simple global snapshot (eligible only)
        global_snapshot = {
            "eligible_rows": int(len(edf)),
            "unique_tickers": int(edf["ticker"].nunique()) if not edf.empty else 0
        }

        st.session_state["df_hash"] = df_hash
        st.session_state["by_ticker"] = by_ticker
        st.session_state["global_snapshot"] = global_snapshot

    by_ticker = st.session_state["by_ticker"]
    snap = st.session_state["global_snapshot"]

    # ======= Kid-simple headline =======
    st.subheader("📢 One-Tap Picks (per stock)")
    if by_ticker.empty:
        st.warning("No tickers matched the rules. Try **Explorer** mode to catch sleepers.")
    else:
        # Top panel: show up to 6 quick picks (you can scroll the full list below)
        top = by_ticker.sort_values("total_premium", ascending=False).head(6)
        cols = st.columns(min(6, len(top)))
        for i, (_, r) in enumerate(top.iterrows()):
            with cols[i]:
                st.metric(f"{r['ticker']}", r["decision"], help=r["reason"])

    st.divider()

    # ======= Super clear counters =======
    st.subheader("🧮 Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligible rows", f"{snap['eligible_rows']:,}")
    c2.metric("Unique tickers", f"{snap['unique_tickers']:,}")
    c3.metric("BUY CALLS", int((by_ticker["decision"] == "BUY CALLS").sum()))
    c4.metric("BUY PUTS", int((by_ticker["decision"] == "BUY PUTS").sum()))

    # ======= Full table: every ticker, popular or not =======
    st.subheader("📦 All Tickers (scroll & sort)")
    st.caption("Sort by Total $ Premium to see whales first. Everything is per-ticker and won’t change on refresh for the same file.")
    st.dataframe(
        by_ticker.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    # Tiny tips for widening the net
    with st.expander("How to catch more (sleepers)"):
        st.markdown("""
- Switch to **Explorer (Catch Sleepers)** mode — this relaxes the filters.
- Make sure your CSV includes **every ticker** you want scanned (popular and less-known).
- If your data source supports it, extend to **more expiries** (up to 16 weeks) and include **smaller but clustered** prints.
        """)
