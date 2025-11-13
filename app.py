# app.py — SilverFoxFlow MACD Scanner — Mach 6.2 (daily only; no sidebar; full hard-coded S&P tickers)
# Requirements: streamlit, yfinance, pandas, numpy
#   pip install streamlit yfinance pandas numpy

import time
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="SilverFoxFlow MACD Scanner — Mach 6.2", layout="wide")

# =========================
# 1) FULL HARD-CODED S&P 500 LIST (from your Mach 6.1 baseline) 
# =========================
SP500 = [
    "NVDA","AAPL","MSFT","GOOG","GOOGL","AMZN","AVGO","META","TSLA","BRK.B","JPM","WMT","LLY","ORCL","V","MA","XOM",
    "PLTR","NFLX","JNJ","AMD","COST","BAC","ABBV","HD","PG","GE","CVX","UNH","KO","CSCO","IBM","WFC","CAT","MS","MU",
    "GS","AXP","CRM","RTX","TMUS","PM","APP","ABT","MRK","TMO","MCD","DIS","UBER","PEP","ANET","LRCX","LIN","QCOM",
    "NOW","INTC","ISRG","INTU","AMAT","C","BX","BLK","T","SCHW","APH","NEE","VZ","BKNG","AMGN","KLAC","GEV","TJX",
    "ACN","BA","DHR","BSX","PANW","GILD","ETN","SPGI","TXN","ADBE","PFE","COF","CRWD","SYK","LOW","UNP","HOOD","HON",
    "DE","WELL","PGR","PLD","CEG","MDT","ADI","LMT","COP","VRTX","CB","DASH","DELL","HCA","KKR","ADP","SO","CMCSA",
    "MCK","TT","CVS","PH","DUK","CME","NKE","MO","BMY","GD","CDNS","SBUX","MMM","NEM","COIN","MMC","MCO","SHW","SNPS",
    "AMT","ICE","NOC","EQIX","HWM","UPS","WM","ORLY","EMR","RCL","ABNB","BK","JCI","MDLZ","TDG","CTAS","AON","TEL",
    "ECL","USB","GLW","PNC","APO","ITW","MAR","WMB","ELV","MSI","CSX","PWR","REGN","SPG","FTNT","COR","MNST","CI",
    "PYPL","GM","RSG","AEP","ADSK","AJG","WDAY","ZTS","VST","NSC","CL","AZO","CMI","SRE","TRV","FDX","FCX","HLT",
    "DLR","MPC","KMI","EOG","AXON","AFL","TFC","DDOG","WBD","URI","PSX","STX","LHX","APD","SLB","O","MET","NXPI",
    "F","VLO","ROST","PCAR","WDC","BDX","ALL","IDXX","CARR","D","EA","PSA","NDAQ","EW","MPWR","ROP","XEL","BKR",
    "TTWO","FAST","GWW","AME","EXC","XYZ","CAH","CBRE","MSCI","DHI","AIG","ETR","KR","OKE","AMP","TGT","PAYX","CMG",
    "CTVA","CPRT","A","FANG","ROK","GRMN","OXY","PEG","LVS","FICO","KMB","CCI","YUM","VMC","CCL","TKO","DAL","EBAY",
    "MLM","KDP","IQV","XYL","PRU","WEC","OTIS","RMD","FI","CHTR","SYY","CTSH","ED","PCG","WAB","VTR","EL","LYV","HIG",
    "NUE","HSY","DD","GEHC","MCHP","HUM","EQT","NRG","TRGP","FIS","STT","HPE","VICI","ACGL","LEN","KEYS","RJF","IBKR",
    "SMCI","VRSK","UAL","IRM","EME","IR","WTW","EXR","ODFL","KHC","MTD","CSGP","ADM","TER","K","FOXA","TSCO","FSLR",
    "MTB","DTE","ROL","AEE","KVUE","ATO","FITB","ES","FOX","BRO","EXPE","WRB","PPL","SYF","FE","HPQ","EFX","BR",
    "CBOE","AWK","HUBB","CNP","DOV","GIS","AVB","TDY","EXE","TTD","VLTO","LDOS","NTRS","HBAN","CINF","PTC","WSM",
    "JBL","NTAP","PHM","ULTA","STE","EQR","STZ","STLD","TPR","DXCM","BIIB","HAL","TROW","VRSN","PODD","CMS","CFG",
    "PPG","DG","TPL","RF","CHD","EIX","LH","DRI","CDW","WAT","L","NVR","DVN","SBAC","TYL","ON","IP","WST","LULU","NI",
    "DLTR","ZBH","KEY","DGX","RL","SW","TRMB","BG","GPN","IT","J","PFG","CPAY","TSN","INCY","AMCR","CHRW","CTRA",
    "GDDY","LII","GPC","EVRG","APTV","PKG","SNA","PNR","CNC","INVH","BBY","MKC","LNT","DOW","PSKY","ESS","WY","EXPD",
    "HOLX","GEN","IFF","JBHT","FTV","LUV","NWS","MAA","ERIE","LYB","NWSA","FFIV","OMC","ALLE","TXT","KIM","COO","UHS",
    "CLX","ZBRA","AVY","CF","DPZ","MAS","EG","NDSN","BF.B","BLDR","IEX","BALL","DOC","HII","BXP","REG","WYNN","UDR",
    "DECK","VTRS","SOLV","HRL","BEN","ALB","SWKS","HST","SJM","DAY","RVTY","JKHY","CPT","AKAM","HAS","AIZ","MRNA",
    "PNW","GL","IVZ","PAYC","SWK","NCLH","ARE","ALGN","FDS","POOL","AES","GNRC","TECH","BAX","IPG","AOS","EPAM","CPB",
    "CRL","MGM","MOS","TAP","LW","DVA","FRT","LKQ","CAG","APA","MOH","MTCH","HSIC","MHK","EMN","KMX"
]

# =========================================
# 2) Helper functions (MACD, EMA, signal detection)
# =========================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def crossed_from_below(macd_line: pd.Series, signal_line: pd.Series, lookback_days=3) -> bool:
    """True if a bullish cross (from below) occurred within last `lookback_days` bars."""
    # Find the most recent index where MACD crosses above signal from below
    if len(macd_line) < lookback_days + 2:
        return False
    recent = macd_line.index[-(lookback_days+1):]
    for i in range(1, len(recent)):
        prev_i = recent[i-1]
        cur_i = recent[i]
        if pd.notna(macd_line[prev_i]) and pd.notna(signal_line[prev_i]) and pd.notna(macd_line[cur_i]) and pd.notna(signal_line[cur_i]):
            if macd_line[prev_i] < signal_line[prev_i] and macd_line[cur_i] >= signal_line[cur_i]:
                return True
    return False

def about_to_cross_from_below(macd_line: pd.Series, signal_line: pd.Series) -> bool:
    """
    'About to cross' definition:
      - Today: MACD < Signal (still below)
      - MACD rising faster than signal (slope up and leading toward a cross)
      - Distance is 'close': |signal - macd| <= proximity band based on MACD ATR-like measure
    """
    if len(macd_line) < 5:
        return False
    m_now, s_now = macd_line.iloc[-1], signal_line.iloc[-1]
    m_prev, s_prev = macd_line.iloc[-2], signal_line.iloc[-2]

    # Must be below, and rising toward it
    if not (m_now < s_now and m_now > m_prev and (m_now - m_prev) >= 0 and (s_now - s_prev) <= (m_now - m_prev) + 1e-9):
        return False

    # Proximity band: use rolling std of (signal - macd) as adaptive threshold
    spread = (signal_line - macd_line).dropna()
    band = spread.rolling(20).std().iloc[-1] if len(spread) >= 20 else spread.std()
    if pd.isna(band) or band == 0:
        band = abs(s_now) * 0.02 + 0.02  # fallback

    return abs(s_now - m_now) <= max(band * 0.75, 0.02)

def already_crossed_bullish(macd_line: pd.Series, signal_line: pd.Series, min_days_ago=4, max_days_ago=20) -> bool:
    """
    Cross happened more than `min_days_ago` bars but within `max_days_ago` bars,
    and MACD remains above signal now.
    """
    if len(macd_line) < max_days_ago + 2:
        return False
    # Find last cross from below
    cross_idx = None
    for i in range(len(macd_line)-max_days_ago-1, len(macd_line)-1):
        if i <= 0: 
            continue
        if macd_line.iloc[i-1] < signal_line.iloc[i-1] and macd_line.iloc[i] >= signal_line.iloc[i]:
            cross_idx = i
    if cross_idx is None:
        return False
    bars_ago = len(macd_line) - 1 - cross_idx
    return (bars_ago >= min_days_ago) and (bars_ago <= max_days_ago) and (macd_line.iloc[-1] > signal_line.iloc[-1])

def ema200(series: pd.Series) -> pd.Series:
    return ema(series, 200)

# =========================================
# 3) Data fetch (robust batching)
# =========================================
def fetch_history_batch(tickers, start, end, retry=2, pause=1.0):
    """
    Use yfinance.download in batches to reduce rate-limit issues.
    Returns dict[ticker] = Series(Adj Close)
    """
    out = {}
    batch_size = 40
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i+batch_size]
        tries = 0
        while tries <= retry:
            try:
                df = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=True)
                if isinstance(df.columns, pd.MultiIndex):
                    # MultiTicker format
                    for t in chunk:
                        if t in df.columns.get_level_values(0):
                            adj = df[(t, "Close")].dropna()
                            if not adj.empty:
                                out[t] = adj
                else:
                    # Single combined Close (single ticker case)
                    if "Close" in df.columns:
                        for t in chunk:
                            out[t] = df["Close"].dropna()
                break
            except Exception:
                tries += 1
                time.sleep(pause * (tries+1))
        time.sleep(0.25)  # be gentle
    return out

# =========================================
# 4) UI Header
# =========================================
st.markdown(
    """
    <style>
      h1, h2, h3 { margin-top: 0.25rem; }
      .section-card {
          border: 1px solid rgba(120,120,120,0.25);
          border-radius: 16px;
          padding: 16px 16px 6px 16px;
          margin-bottom: 16px;
          background: rgba(250,250,255,0.6);
      }
      .small-note { color: #666; font-size: 0.9rem; }
      .good { background: rgba(0,200,0,0.07); }
      .ok   { background: rgba(240,180,0,0.10); }
      .warn { background: rgba(200,0,0,0.07); }
      .tbl th { position: sticky; top: 0; background: #fff; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("SilverFoxFlow — MACD Daily Scanner (Mach 6.2)")
st.caption("Bullish-only, **from below**. Categories: About to Cross • Just Crossed • Already Crossed. Zero line ignored; focus is the cross.")

# =========================================
# 5) Run scan
# =========================================
# We’ll look back ~300 trading days to guarantee 200-EMA and MACD windows
end_date = datetime.now(timezone.utc).date() + timedelta(days=1)  # include last bar
start_date = end_date - timedelta(days=420)

with st.status("Fetching data & scanning…", expanded=False) as status:
    status.update(label="Downloading history (daily)…")
    price_map = fetch_history_batch(SP500, start=start_date.isoformat(), end=end_date.isoformat(), retry=2, pause=1.0)

    about_rows = []
    just_rows = []
    already_rows = []
    skipped = []

    for t, series in price_map.items():
        try:
            close = series.copy()
            if close.isna().sum() > 0 or len(close) < 60:
                skipped.append((t, "insufficient data"))
                continue

            m_line, s_line, h_line = macd(close)
            ema200_series = ema200(close)

            if len(m_line.dropna()) < 35 or len(s_line.dropna()) < 35:
                skipped.append((t, "insufficient macd length"))
                continue

            price_now = float(close.iloc[-1])
            ema200_now = float(ema200_series.iloc[-1])
            above_200 = price_now > ema200_now if not math.isnan(ema200_now) else False

            # Classification (exclusive, in this order)
            if crossed_from_below(m_line, s_line, lookback_days=3):
                cat = "Just Crossed"
                just_rows.append({
                    "Ticker": t,
                    "Price": round(price_now, 2),
                    "Above 200 EMA": "Yes" if above_200 else "No",
                    "Days Since Cross": 0  # will recalc below
                })
            elif about_to_cross_from_below(m_line, s_line):
                cat = "About to Cross"
                about_rows.append({
                    "Ticker": t,
                    "Price": round(price_now, 2),
                    "Above 200 EMA": "Yes" if above_200 else "No",
                    "MACD Dist": round((s_line.iloc[-1] - m_line.iloc[-1]), 4)
                })
            elif already_crossed_bullish(m_line, s_line, min_days_ago=4, max_days_ago=20):
                cat = "Already Crossed"
                # Compute days since the cross:
                # find last cross index
                cross_idx = None
                for i in range(len(m_line)-21, len(m_line)):
                    if i <= 0:
                        continue
                    if m_line.iloc[i-1] < s_line.iloc[i-1] and m_line.iloc[i] >= s_line.iloc[i]:
                        cross_idx = i
                days_since = (len(m_line) - 1 - cross_idx) if cross_idx is not None else None

                already_rows.append({
                    "Ticker": t,
                    "Price": round(price_now, 2),
                    "Above 200 EMA": "Yes" if above_200 else "No",
                    "Days Since Cross": days_since
                })
            else:
                skipped.append((t, "no bullish-from-below setup"))
        except Exception as e:
            skipped.append((t, f"error: {e}"))

    status.update(label="Scan complete.")

# =========================================
# 6) Sorting rules (LLY-style priority up top)
#     - prioritize Above 200 EMA = Yes first, then strongest “freshness”
# =========================================
def sort_about(df: pd.DataFrame):
    if df.empty: 
        return df
    # Smaller distance = closer to cross (better). Above 200 first.
    return df.sort_values(by=["Above 200 EMA","MACD Dist","Ticker"], ascending=[False, True, True])

def sort_just(df: pd.DataFrame):
    if df.empty:
        return df
    # For "Just Crossed", 'Days Since Cross' close to 0 is fresher. Above 200 first.
    if "Days Since Cross" in df.columns and df["Days Since Cross"].notna().any():
        return df.sort_values(by=["Above 200 EMA","Days Since Cross","Ticker"], ascending=[False, True, True])
    # Fallback
    return df.sort_values(by=["Above 200 EMA","Ticker"], ascending=[False, True])

def sort_already(df: pd.DataFrame):
    if df.empty:
        return df
    # For "Already Crossed", smaller days since cross is stronger. Above 200 first.
    return df.sort_values(by=["Above 200 EMA","Days Since Cross","Ticker"], ascending=[False, True, True])

about_df  = pd.DataFrame(about_rows,  columns=["Ticker","Price","Above 200 EMA","MACD Dist"])
just_df   = pd.DataFrame(just_rows,   columns=["Ticker","Price","Above 200 EMA","Days Since Cross"])
already_df= pd.DataFrame(already_rows,columns=["Ticker","Price","Above 200 EMA","Days Since Cross"])

about_df   = sort_about(about_df)
just_df    = sort_just(just_df)
already_df = sort_already(already_df)

st.markdown("### Scan Summary")
colA, colB, colC, colD = st.columns(4)
colA.metric("About to Cross (Bullish)", len(about_df))
colB.metric("Just Crossed (Bullish)", len(just_df))
colC.metric("Already Crossed (Bullish)", len(already_df))
colD.metric("Processed", len(price_map))

with st.expander("Skipped / Not in a bullish-from-below setup (reason)", expanded=False):
    if skipped:
        sk = pd.DataFrame(skipped, columns=["Ticker","Reason"])
        st.dataframe(sk, use_container_width=True, height=260)
    else:
        st.write("None 🎉")

# =========================
# 7) Sections (separate tables; sticky headers)
# =========================
st.markdown('<div class="section-card good">', unsafe_allow_html=True)
st.subheader("1) About to Cross (Bullish)")
st.caption("MACD is **below** Signal, rising, and within a tight proximity band — likely to cross soon.")
if about_df.empty:
    st.write("No symbols detected right now.")
else:
    st.dataframe(about_df.reset_index(drop=True), use_container_width=True, height=min(600, 42 + 28*len(about_df)))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card good">', unsafe_allow_html=True)
st.subheader("2) Just Crossed (Bullish)")
st.caption("Crossed from **below** within last 3 trading days — freshest signals.")
if just_df.empty:
    st.write("No symbols detected right now.")
else:
    st.dataframe(just_df.reset_index(drop=True), use_container_width=True, height=min(600, 42 + 28*len(just_df)))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card ok">', unsafe_allow_html=True)
st.subheader("3) Already Crossed (Bullish)")
st.caption("Cross occurred >3 and ≤20 days ago and MACD remains above Signal — still valid trend continuation.")
if already_df.empty:
    st.write("No symbols detected right now.")
else:
    st.dataframe(already_df.reset_index(drop=True), use_container_width=True, height=min(600, 42 + 28*len(already_df)))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f"<p class='small-note'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Timeframe: Daily • Universe size: {len(SP500)} tickers</p>",
    unsafe_allow_html=True
)
