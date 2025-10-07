import json, os
from datetime import datetime
import streamlit as st

from providers.mock import generate_mock_market
from uoa_engine import score_market, DEFAULT_WEIGHTS

st.set_page_config(page_title="SilverFoxFlow — UOA 2.0", page_icon="🦊", layout="wide")

st.title("🦊 SilverFoxFlow — UOA 2.0")
st.caption("Cunning, institutional-grade Unusual Options Activity with Smart Money Scores & clear trade verdicts.")

# Sidebar
st.sidebar.header("Scanner Controls")
mode = st.sidebar.selectbox("Mode", ["Use Latest Report.json", "Run Fresh Scan (mock data)"])
weights = DEFAULT_WEIGHTS.copy()

with st.sidebar.expander("Smart Money Score Weights", expanded=False):
    for k, v in list(weights.items()):
        weights[k] = st.slider(k.replace("_"," ").title(), 0, 30, v, 1)
st.sidebar.markdown("---")
st.sidebar.write("**Tip:** Replace the mock provider with a real data source later.")

def load_report():
    path = "data/report.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def write_report(verdicts):
    os.makedirs("data", exist_ok=True)
    payload = {"generated_at_utc": datetime.utcnow().isoformat(), "verdicts": verdicts}
    with open("data/report.json","w") as f:
        json.dump(payload, f, indent=2)

col1, col2 = st.columns([2,1])

if mode == "Run Fresh Scan (mock data)":
    with st.spinner("Scanning market (mock provider)…"):
        market, contexts = generate_mock_market()
        ideas = score_market(market, contexts, weights)
        verdicts = [i.__dict__ for i in ideas[:5]]
        write_report(verdicts)
    st.success("Scan complete. Report updated.")
    latest = {"generated_at_utc": datetime.utcnow().isoformat(), "verdicts": verdicts}
else:
    latest = load_report()
    if not latest:
        st.warning("No report.json yet. Run a fresh scan to generate one.")
        latest = {"generated_at_utc": "-", "verdicts": []}

with col1:
    st.subheader("Top Signals")
    if latest["verdicts"]:
        import pandas as pd
        df = pd.DataFrame(latest["verdicts"])
        show = df[["ticker","expiry","strike","type","premium_usd","score","verdict"]]
        show = show.sort_values("score", ascending=False)
        st.dataframe(show, use_container_width=True)
    else:
        st.info("No signals to display.")

with col2:
    st.subheader("Morning Verdict")
    if latest["verdicts"]:
        top = sorted(latest["verdicts"], key=lambda x: x["score"], reverse=True)[:3]
        lines = []
        for v in top:
            lines.append(f"{v['ticker']}: **{v['verdict']}** (Score {v['score']})")
        st.markdown("\n\n".join(lines))
    st.markdown("---")
    st.caption(f"Generated (UTC): {latest.get('generated_at_utc','-')}")

st.markdown("---")
st.markdown("**How this works:** UOA 2.0 filters for aggressive, short-dated institutional flow, removes hedges, then scores each idea (0–100) using aggression, volume vs OI, premium size, clusters, VWAP/RSI/EMA trend, and catalysts.")
