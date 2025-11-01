import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P MACD Scanner – Fresh", layout="wide")

# ===== 1) YOUR S&P LIST (from your paste) =====
SP500_TICKERS = [
    "NVDA","AAPL","MSFT","GOOG","GOOGL","AMZN","AVGO","META","TSLA","BRK.B",
    "JPM","WMT","LLY","ORCL","V","MA","XOM","PLTR","NFLX","JNJ","AMD","COST",
    "BAC","ABBV","HD","PG","GE","CVX","UNH","KO","CSCO","IBM","WFC","CAT",
    "MS","MU","GS","AXP","CRM","RTX","TMUS","PM","APP","ABT","MRK","TMO",
    "MCD","DIS","UBER","PEP","ANET","LRCX","LIN","QCOM","NOW","INTC","ISRG",
    "INTU","AMAT","C","BX","BLK","T","SCHW","APH","NEE","VZ","BKNG","AMGN",
    "KLAC","GEV","TJX","ACN","BA","DHR","BSX","PANW","GILD","ETN","SPGI",
    "TXN","ADBE","PFE","COF","CRWD","SYK","LOW","UNP","HOOD","HON","DE",
    "WELL","PGR","PLD","CEG","MDT","ADI","LMT","COP","VRTX","CB","DASH",
    "DELL","HCA","KKR","ADP","SO","CMCSA","MCK","TT","CVS","PH","DUK","CME",
    "NKE","MO","BMY","GD","CDNS","SBUX","MMM","NEM","COIN","MMC","MCO","SHW",
    "SNPS","AMT","ICE","NOC","EQIX","HWM","UPS","WM","ORLY","EMR","RCL",
    "ABNB","BK","JCI","MDLZ","TDG","CTAS","AON","TEL","ECL","USB","GLW",
    "PNC","APO","ITW","MAR","WMB","ELV","MSI","CSX","PWR","REGN","SPG",
    "FTNT","COR","MNST","CI","PYPL","GM","RSG","AEP","ADSK","AJG","WDAY",
    "ZTS","VST","NSC","CL","AZO","CMI","SRE","TRV","FDX","FCX","HLT","DLR",
    "MPC","KMI","EOG","AXON","AFL","TFC","DDOG","WBD","URI","PSX","STX",
    "LHX","APD","SLB","O","MET","NXPI","F","VLO","ROST","PCAR","WDC","BDX",
    "ALL","IDXX","CARR","D","EA","PSA","NDAQ","EW","MPWR","ROP","XEL","BKR",
    "TTWO","FAST","GWW","AME","EXC","XYZ","CAH","CBRE","MSCI","DHI","AIG",
    "ETR","KR","OKE","AMP","TGT","PAYX","CMG","CTVA","CPRT","A","FANG","ROK",
    "GRMN","OXY","PEG","LVS","FICO","KMB","CCI","YUM","VMC","CCL","TKO",
    "DAL","EBAY","MLM","KDP","IQV","XYL","PRU","WEC","OTIS","RMD","FI",
    "CHTR","SYY","CTSH","ED","PCG","WAB","VTR","EL","LYV","HIG","NUE","HSY",
    "DD","GEHC","MCHP","HUM","EQT","NRG","TRGP","FIS","STT","HPE","VICI",
    "ACGL","LEN","KEYS","RJF","IBKR","SMCI","VRSK","UAL","IRM","EME","IR",
    "WTW","EXR","ODFL","KHC","MTD","CSGP","ADM","TER","K","FOXA","TSCO",
    "FSLR","MTB","DTE","ROL","AEE","KVUE","ATO","FITB","ES","FOX","BRO",
    "EXPE","WRB","PPL","FE","HPQ","EFX","BR","CBOE","AWK","HUBB","CNP","DOV",
    "GIS","AVB","TDY","EXE","TTD","VLTO","LDOS","NTRS","HBAN","CINF","PTC",
    "WSM","JBL","NTAP","PHM","ULTA","STE","EQR","STZ","STLD","TPR","DXCM",
    "BIIB","HAL","TROW","VRSN","PODD","CMS","CFG","PPG","DG","TPL","RF",
    "CHD","EIX","LH","DRI","CDW","WAT","L","NVR","DVN","SBAC","TYL","ON",
    "IP","WST","LULU","NI","DLTR","ZBH","KEY","DGX","RL","SW","TRMB","BG",
    "GPN","IT","J","PFG","CPAY","TSN","INCY","AMCR","CHRW","CTRA","GDDY",
    "LII","GPC","EVRG","APTV","PKG","SNA","PNR","CNC","INVH","BBY","MKC",
    "LNT","DOW","PSKY","ESS","WY","EXPD","HOLX","GEN","IFF","JBHT","FTV",
    "LUV","NWS","MAA","ERIE","LYB","NWSA","FFIV","OMC","ALLE","TXT","KIM",
    "COO","UHS","CLX","ZBRA","AVY","CF","DPZ","MAS","EG","NDSN","BF.B",
    "BLDR","IEX","BALL","DOC","HII","BXP","REG","WYNN","UDR","DECK","VTRS",
    "SOLV","HRL","BEN","ALB","SWKS","HST","SJM","DAY","RVTY","JKHY","CPT",
    "AKAM","HAS","AIZ","MRNA","PNW","GL","IVZ","PAYC","SWK","NCLH","ARE",
    "ALGN","FDS","POOL","AES","GNRC","TECH","BAX","IPG","AOS","EPAM","CPB",
    "CRL","MGM","MOS","TAP","LW","DVA","FRT","LKQ","CAG","APA","MOH","MTCH",
    "HSIC","MHK","EMN","KMX"
]

# ===== 2) CONTROLS =====
st.sidebar.header("Controls")
max_per_section = st.sidebar.slider("Max tickers to show per section", 5, 80, 30, 5)
lookback_days = st.sidebar.selectbox("Lookback data (days)", [60, 90, 120, 180, 252], index=2)

st.title("S&P MACD Scanner – Fresh / Just Crossed / About To 🟢")
st.caption("Logic (daily): 1) about to cross up 2) just crossed up 3) still bullish. Always show something.")

# ===== 3) HELPERS =====

@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, days: int):
    try:
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

def compute_macd_from_df(df: pd.DataFrame):
    """
    Return LAST and PREV values for MACD, signal, hist.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    if len(macd) < 2:
        return None

    out = {
        "close": float(close.iloc[-1]),
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "hist": float(hist.iloc[-1]),
        "macd_prev": float(macd.iloc[-2]),
        "signal_prev": float(signal.iloc[-2]),
        "hist_prev": float(hist.iloc[-2]),
    }
    return out

def classify_fresh_bullish(row: dict):
    """
    Return (bucket_name, note, fresh_score)
    Buckets:
      1. ABOUT_TO_BULL   – macd below signal but curling up
      2. JUST_CROSSED_UP – macd_prev < signal_prev and macd >= signal
      3. STILL_BULLISH   – macd >= signal (already up)
    """
    macd = row["macd"]
    sig = row["signal"]
    hist = row["hist"]
    macd_prev = row["macd_prev"]
    sig_prev = row["signal_prev"]
    hist_prev = row["hist_prev"]

    diff = macd - sig
    diff_prev = macd_prev - sig_prev

    # 2) JUST CROSSED UP
    if diff_prev < 0 and diff >= 0:
        return ("JUST_CROSSED_UP", "MACD crossed above signal on the latest bar.", 95)

    # 1) ABOUT TO CROSS (still below, but improving)
    #  - below signal
    #  - closer to zero than yesterday
    #  - histogram improving
    if diff < 0 and diff > diff_prev and hist > hist_prev:
        return ("ABOUT_TO_BULL", "MACD below signal but curling up / histogram rising.", 85)

    # 3) STILL BULLISH (already above signal, hist >= 0)
    if diff > 0 and hist >= 0:
        return ("STILL_BULLISH", "MACD above signal and histogram is positive.", 70)

    # else → neutral
    return (None, "Does not meet bullish freshness criteria.", 0)

def classify_fresh_bearish(row: dict):
    macd = row["macd"]
    sig = row["signal"]
    hist = row["hist"]
    macd_prev = row["macd_prev"]
    sig_prev = row["signal_prev"]
    hist_prev = row["hist_prev"]

    diff = macd - sig
    diff_prev = macd_prev - sig_prev

    # JUST CROSSED DOWN
    if diff_prev > 0 and diff <= 0:
        return ("JUST_CROSSED_DOWN", "MACD just crossed down.", 95)

    # ABOUT TO CROSS DOWN
    if diff > 0 and diff < diff_prev and hist < hist_prev:
        return ("ABOUT_TO_BEAR", "MACD above signal but curling down / hist falling.", 85)

    # STILL BEARISH
    if diff < 0 and hist <= 0:
        return ("STILL_BEARISH", "MACD below signal, histogram negative.", 70)

    return (None, "Does not meet bearish freshness criteria.", 0)

# ===== 4) SCAN =====

bull_about = []
bull_just = []
bull_still = []
bear_about = []
bear_just = []
bear_still = []
neutral = []

with st.spinner("Scanning tickers... (first run can be slow)"):
    for ticker in SP500_TICKERS:
        df = fetch_history(ticker, lookback_days)
        macd_row = compute_macd_from_df(df)
        if macd_row is None:
            neutral.append({
                "ticker": ticker,
                "close": None,
                "macd": None,
                "signal": None,
                "hist": None,
                "fresh_score": 0,
                "verdict": "NO DATA",
                "note": "yfinance returned empty / not enough candles",
            })
            continue

        base = {
            "ticker": ticker,
            "close": round(macd_row["close"], 2),
            "macd": round(macd_row["macd"], 4),
            "signal": round(macd_row["signal"], 4),
            "hist": round(macd_row["hist"], 4),
        }

        bull_label, bull_note, bull_score = classify_fresh_bullish(macd_row)
        bear_label, bear_note, bear_score = classify_fresh_bearish(macd_row)

        # PRIORITY: bullish first, then bearish, else neutral
        if bull_label == "ABOUT_TO_BULL":
            bull_about.append({**base, "fresh_score": bull_score, "note": bull_note, "verdict": "ABOUT TO (CALLS)"})
        elif bull_label == "JUST_CROSSED_UP":
            bull_just.append({**base, "fresh_score": bull_score, "note": bull_note, "verdict": "JUST CROSSED (CALLS)"})
        elif bull_label == "STILL_BULLISH":
            bull_still.append({**base, "fresh_score": bull_score, "note": bull_note, "verdict": "STILL BULLISH (CALLS)"})
        elif bear_label == "ABOUT_TO_BEAR":
            bear_about.append({**base, "fresh_score": bear_score, "note": bear_note, "verdict": "ABOUT TO (PUTS)"})
        elif bear_label == "JUST_CROSSED_DOWN":
            bear_just.append({**base, "fresh_score": bear_score, "note": bear_note, "verdict": "JUST CROSSED (PUTS)"})
        elif bear_label == "STILL_BEARISH":
            bear_still.append({**base, "fresh_score": bear_score, "note": bear_note, "verdict": "STILL BEARISH (PUTS)"})
        else:
            neutral.append({**base, "fresh_score": 0, "verdict": "NEUTRAL / LATE / NO TRADE", "note": "MACD not in ideal spot"})

# sort by fresh_score desc
bull_about = sorted(bull_about, key=lambda x: x["fresh_score"], reverse=True)[:max_per_section]
bull_just = sorted(bull_just, key=lambda x: x["fresh_score"], reverse=True)[:max_per_section]
bull_still = sorted(bull_still, key=lambda x: x["fresh_score"], reverse=True)[:max_per_section]
bear_about = sorted(bear_about, key=lambda x: x["fresh_score"], reverse=True)[:max_per_section]
bear_just = sorted(bear_just, key=lambda x: x["fresh_score"], reverse=True)[:max_per_section]
bear_still = sorted(bear_still, key=lambda x: x["fresh_score"], reverse=True)[:max_per_section]
neutral = sorted(neutral, key=lambda x: x["ticker"])[:max_per_section]

# ===== 5) DISPLAY =====

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🟢 1. About to cross bullish (CALLS)")
    st.dataframe(pd.DataFrame(bull_about))

with col2:
    st.subheader("🟢 2. Just crossed bullish (CALLS)")
    st.dataframe(pd.DataFrame(bull_just))

with col3:
    st.subheader("🟢 3. Still bullish (CALLS)")
    st.dataframe(pd.DataFrame(bull_still))

st.markdown("---")

col4, col5, col6 = st.columns(3)
with col4:
    st.subheader("🔴 About to cross bearish (PUTS)")
    st.dataframe(pd.DataFrame(bear_about))

with col5:
    st.subheader("🔴 Just crossed bearish (PUTS)")
    st.dataframe(pd.DataFrame(bear_just))

with col6:
    st.subheader("🔴 Still bearish (PUTS)")
    st.dataframe(pd.DataFrame(bear_still))

st.markdown("---")
st.subheader("😐 Neutral / Late / No Trade")
st.dataframe(pd.DataFrame(neutral))

st.caption(f"Last run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
