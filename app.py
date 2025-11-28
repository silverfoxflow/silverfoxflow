
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# ==============================
#  CONFIG
# ==============================

st.set_page_config(
    page_title="SilverFoxFlow — MACD UOA SCANNER LOGIC (Mach 7.4)",
    layout="wide",
)

# ==============================
#  CONSTANTS
# ==============================

# --- MACD side: Static S&P 500 list (Mach 6.2 core) ---
SP500_TICKERS = [
    "AAPL","MSFT","GOOG","GOOGL","AMZN","NVDA","AVGO","META","TSLA","BRK.B","LLY","JPM","WMT","ORCL","V","MA","XOM",
    "PLTR","NFLX","JNJ","AMD","COST","BAC","ABBV","HD","PG","GE","CVX","UNH","KO","CSCO","IBM","WFC","CAT","MS","MU","GS",
    "AXP","CRM","MRK","RTX","TMUS","PM","APP","ABT","TMO","MCD","DIS","UBER","PEP","ANET","LRCX","LIN","QCOM","NOW","INTC",
    "ISRG","INTU","AMAT","C","BX","BLK","T","SCHW","APH","NEE","VZ","BKNG","AMGN","KLAC","GEV","TJX","ACN","BA","DHR","BSX",
    "PANW","GILD","ETN","SPGI","TXN","ADBE","PFE","COF","CRWD","SYK","LOW","UNP","HOOD","HON","DE","WELL","PGR","PLD","CEG",
    "MDT","ADI","LMT","COP","VRTX","CB","DASH","DELL","HCA","KKR","ADP","SO","CMCSA","MCK","TT","CVS","PH","DUK","CME","NKE",
    "MO","BMY","GD","CDNS","SBUX","MMM","NEM","COIN","MMC","MCO","SHW","SNPS","AMT","ICE","NOC","EQIX","HWM","UPS","WM",
    "ORLY","EMR","RCL","ABNB","BK","JCI","MDLZ","TDG","CTAS","AON","TEL","ECL","USB","GLW","PNC","APO","ITW","MAR","WMB",
    "ELV","MSI","CSX","PWR","REGN","SPG","FTNT","COR","MNST","CI","PYPL","GM","RSG","AEP","ADSK","AJG","WDAY","ZTS","VST",
    "NSC","CL","AZO","CMI","SRE","TRV","FDX","FCX","HLT","DLR","MPC","KMI","EOG","AXON","AFL","TFC","DDOG","WBD","URI","PSX",
    "STX","LHX","APD","SLB","O","MET","NXPI","F","VLO","ROST","PCAR","WDC","BDX","ALL","IDXX","CARR","D","EA","PSA","NDAQ",
    "EW","MPWR","ROP","XEL","BKR","TTWO","FAST","GWW","AME","EXC","XYZ","CAH","CBRE","MSCI","DHI","AIG","ETR","KR","OKE",
    "AMP","TGT","PAYX","CMG","CTVA","CPRT","A","FANG","ROK","GRMN","OXY","PEG","LVS","FICO","KMB","CCI","YUM","VMC","CCL",
    "TKO","DAL","EBAY","MLM","KDP","IQV","XYL","PRU","WEC","OTIS","RMD","FI","CHTR","SYY","CTSH","ED","PCG","WAB","VTR","EL",
    "LYV","HIG","NUE","HSY","DD","GEHC","MCHP","HUM","EQT","NRG","TRGP","FIS","STT","HPE","VICI","ACGL","LEN","KEYS","RJF",
    "IBKR","SMCI","VRSK","UAL","IRM","EME","IR","WTW","EXR","ODFL","KHC","MTD","CSGP","ADM","TER","K","FOXA","TSCO","FSLR",
    "MTB","DTE","ROL","AEE","KVUE","ATO","FITB","ES","FOX","BRO","EXPE","WRB","PPL","SYF","FE","HPQ","EFX","BR","CBOE","AWK",
    "HUBB","CNP","DOV","GIS","AVB","TDY","EXE","TTD","VLTO","LDOS","NTRS","HBAN","CINF","PTC","WSM","JBL","NTAP","PHM","ULTA",
    "STE","EQR","STZ","STLD","TPR","DXCM","BIIB","HAL","TROW","VRSN","PODD","CMS","CFG","PPG","DG","TPL","RF","CHD","EIX","LH",
    "DRI","CDW","WAT","L","NVR","DVN","SBAC","TYL","ON","IP","WST","LULU","NI","DLTR","ZBH","KEY","DGX","RL","SW","TRMB","BG",
    "GPN","IT","J","PFG","CPAY","TSN","INCY","AMCR","CHRW","CTRA","GDDY","LII","GPC","EVRG","APTV","PKG","SNA","PNR","CNC",
    "INVH","BBY","MKC","LNT","DOW","PSKY","ESS","WY","EXPD","HOLX","GEN","IFF","JBHT","FTV","LUV","NWS","MAA","ERIE","LYB",
    "NWSA","FFIV","OMC","ALLE","TXT","KIM","COO","UHS","CLX","ZBRA","AVY","CF","DPZ","MAS","EG","NDSN","BF.B","BLDR","IEX",
    "BALL","DOC","HII","BXP","REG","WYNN","UDR","DECK","VTRS","SOLV","HRL","BEN","ALB","SWKS","HST","SJM","DAY","RVTY","JKHY",
    "CPT","AKAM","HAS","AIZ","MRNA","PNW","GL","IVZ","PAYC","SWK","NCLH","ARE","ALGN","FDS","POOL","AES","GNRC","TECH","BAX",
]

# --- UOA side: Unusual Whales API (STRICT UOA 2.0) ---
UW_API_KEY = "47ec6a4d-e5be-4d37-a6af-73f97545771a"
UW_BASE_URL = "https://api.unusualwhales.com/api"
UOA_MIN_PREMIUM = 150_000   # strict institutional cluster
SOFT_MIN_PREMIUM = 40_000   # soft heads-up sweeps
LOOKBACK_DAYS = 3           # last 3 days of flow

# ==============================
#  MACD HELPERS (Mach 6.2 core)
# ==============================

@st.cache_data(show_spinner=False)
def download_history(tickers, period="1y", interval="1d", batch_size=60):
    """Download OHLCV data for tickers in batches."""
    all_data = {}
    tickers = list(dict.fromkeys([t.upper() for t in tickers]))

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            df = yf.download(
                batch,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            for t in batch:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna()
                    if not sub.empty:
                        all_data[t] = sub
        else:
            t = batch[0]
            sub = df.dropna()
            if not sub.empty:
                all_data[t] = sub

    return all_data


def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def find_last_bull_cross(macd, sig, lookback=60):
    if len(macd) < 2:
        return None

    macd = macd.dropna()
    sig = sig.dropna()
    common_index = macd.index.intersection(sig.index)
    macd = macd.loc[common_index]
    sig = sig.loc[common_index]
    if len(macd) < 2:
        return None

    recent = macd.iloc[-lookback:]
    recent_sig = sig.iloc[-lookback:]
    last_cross_idx = None

    for i in range(1, len(recent)):
        prev_macd = recent.iloc[i - 1]
        prev_sig = recent_sig.iloc[i - 1]
        curr_macd = recent.iloc[i]
        curr_sig = recent_sig.iloc[i]
        if prev_macd < prev_sig and curr_macd > curr_sig:
            last_cross_idx = recent.index[i]

    if last_cross_idx is None:
        return None

    days_since = (recent.index[-1] - last_cross_idx).days
    return days_since if days_since >= 0 else None


def classify_ticker(ticker, df, max_lookback_cross=60):
    close = df["Close"]
    if len(close) < 50:
        return None, "skipped", "Not enough history"

    macd, sig, _ = compute_macd(close)
    macd_curr = macd.iloc[-1]
    sig_curr = sig.iloc[-1]
    macd_prev = macd.iloc[-2]

    ema200 = close.ewm(span=200, adjust=False).mean()
    ema200_curr = float(ema200.iloc[-1])
    price_curr = float(close.iloc[-1])
    above_200 = price_curr > ema200_curr

    macd_dist = float(macd_curr - sig_curr)
    days_since_cross = find_last_bull_cross(macd, sig, lookback=max_lookback_cross)

    if days_since_cross is None:
        return {
            "ticker": ticker,
            "price": price_curr,
            "above_200": above_200,
            "macd_dist": macd_dist,
            "days_since_cross": None,
        }, "skipped", "No bullish-from-below cross in lookback"

    if macd_curr > sig_curr and days_since_cross <= 3:
        section = "just_crossed"
    elif macd_curr > sig_curr and days_since_cross > 3:
        section = "already_crossed"
    else:
        rising = macd_curr > macd_prev
        below = macd_curr < sig_curr
        band = abs(macd_dist) <= 0.2
        if below and rising and band:
            section = "about_to_cross"
        else:
            section = "skipped"

    result = {
        "ticker": ticker,
        "price": price_curr,
        "above_200": above_200,
        "macd_dist": macd_dist,
        "days_since_cross": days_since_cross,
    }

    if section == "skipped":
        reason = []
        if not (macd_curr > sig_curr or (below and rising and band)):
            reason.append("Not in clean bullish-from-below setup")
        if days_since_cross is None:
            reason.append("No recent bullish cross")
        return result, "skipped", "; ".join(reason) or "Filtered out"

    return result, section, None


def build_tables(data_map):
    about_rows, just_rows, already_rows, skipped_rows = [], [], [], []

    for ticker, df in data_map.items():
        info, section, reason = classify_ticker(ticker, df)
        if info is None:
            continue

        base_row = {
            "Ticker": info["ticker"],
            "Price": round(info["price"], 2),
            "Above 200 EMA": "Yes" if info["above_200"] else "No",
            "MACD Dist": round(info["macd_dist"], 4),
        }

        if section == "about_to_cross":
            about_rows.append(base_row)
        elif section == "just_crossed":
            row = base_row.copy()
            row["Days Since Cross"] = info["days_since_cross"]
            just_rows.append(row)
        elif section == "already_crossed":
            row = base_row.copy()
            row["Days Since Cross"] = info["days_since_cross"]
            already_rows.append(row)
        elif section == "skipped":
            row = base_row.copy()
            row["Reason"] = reason or ""
            skipped_rows.append(row)

    about_df = (
        pd.DataFrame(about_rows).sort_values("MACD Dist")
        if about_rows
        else pd.DataFrame(columns=["Ticker", "Price", "Above 200 EMA", "MACD Dist"])
    )
    just_df = (
        pd.DataFrame(just_rows).sort_values("Days Since Cross")
        if just_rows
        else pd.DataFrame(
            columns=["Ticker", "Price", "Above 200 EMA", "MACD Dist", "Days Since Cross"]
        )
    )
    already_df = (
        pd.DataFrame(already_rows).sort_values("Days Since Cross")
        if already_rows
        else pd.DataFrame(
            columns=["Ticker", "Price", "Above 200 EMA", "MACD Dist", "Days Since Cross"]
        )
    )
    skipped_df = (
        pd.DataFrame(skipped_rows)
        if skipped_rows
        else pd.DataFrame(
            columns=["Ticker", "Price", "Above 200 EMA", "MACD Dist", "Reason"]
        )
    )

    return about_df, just_df, already_df, skipped_df

# ==============================
#  UOA HELPERS (Mach 7.4, STRICT UOA 2.0)
# ==============================

def _normalize_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    numeric_cols = [
        "total_premium",
        "total_ask_side_prem",
        "total_bid_side_prem",
        "underlying_price",
        "price",
        "volume",
        "open_interest",
        "strike",
        "dte",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _detect_time_col(df: pd.DataFrame):
    for cand in ["start_time", "created_at", "time", "end_time"]:
        if cand in df.columns:
            return cand
    return None


from pandas.api import types as ptypes  # put near the top with imports

def _parse_time_series(s: pd.Series) -> pd.Series:
    """
    Handle UW timestamps that might already be datetime, strings, or ms-since-epoch.
    """
    # If it's numeric (ms since epoch)
    if ptypes.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms", errors="coerce", utc=True)
    # If it's already datetime or string, this will just normalize it
    return pd.to_datetime(s, errors="coerce", utc=True)



def _format_recency(ts, now):
    if pd.isna(ts):
        return None
    delta = now - ts
    days = delta.total_seconds() / 86400.0
    if days <= 1:
        return "Fresh (last 24h)"
    elif days <= 3:
        return "1–3 days old"
    else:
        return ">3 days old"


def _add_dte_if_missing(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if "dte" in df.columns and df["dte"].notna().any():
        return df
    if "expiry" not in df.columns or time_col is None:
        return df
    df = df.copy()
    df["expiry_dt"] = pd.to_datetime(df["expiry"], errors="coerce", utc=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    mask = df["expiry_dt"].notna() & df[time_col].notna()
    df.loc[mask, "dte"] = (df.loc[mask, "expiry_dt"] - df.loc[mask, time_col]).dt.days
    return df


def _compute_pct_otm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "underlying_price" in df.columns and "strike" in df.columns:
        up = df["underlying_price"]
        st = df["strike"]
        mask = up.notna() & (up != 0) & st.notna()
        df["pct_otm"] = np.nan
        df.loc[mask, "pct_otm"] = (st[mask] - up[mask]) / up[mask] * 100.0
    else:
        df["pct_otm"] = np.nan
    return df


def _add_direction_from_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "type" not in df.columns:
        df["direction"] = np.nan
        return df

    def map_dir(v):
        v = str(v).lower()
        if v.startswith("c"):
            return "bullish"
        if v.startswith("p"):
            return "bearish"
        return None

    df["direction"] = df["type"].apply(map_dir)
    return df


def fetch_uoa_flow_alerts_last_3_days(max_pages=8, per_page=500) -> pd.DataFrame:
    """Pull paginated high-level flow alerts from Unusual Whales (last ~3 days)."""
    url = f"{UW_BASE_URL}/option-trades/flow-alerts"
    headers = {
        "Authorization": f"Bearer {UW_API_KEY}",
        "Accept": "application/json",
    }

    frames = []
    for page in range(1, max_pages + 1):
        params = {"limit": per_page, "page": page}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                records = data["data"]
            else:
                records = data
            if not records:
                break
            frames.append(pd.DataFrame(records))
        except Exception as e:
            st.warning(f"Error fetching UOA page {page}: {e}")
            break

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = _normalize_and_numeric(df)

    time_col = _detect_time_col(df)
    if time_col:
        df[time_col] = _parse_time_series(df[time_col])
        now = pd.Timestamp.utcnow()
        cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
        df = df[df[time_col] >= cutoff]

    return df


def apply_strict_uoa_filters(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT UOA 2.0 using Option C:
      - row = cluster (already aggregated by Unusual Whales)
      - Sweeps only
      - total_premium >= 150k
      - volume > open_interest
      - DTE 1–56
      - |%OTM| <= 15
      - trade_count >= 2
      - Direction: calls → bullish, puts → bearish
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = _normalize_and_numeric(df_raw)
    required = ["ticker", "total_premium", "has_sweep", "strike", "underlying_price", "type"]
    if any(r not in df.columns for r in required):
        return pd.DataFrame()

    time_col = _detect_time_col(df)
    if time_col:
        df[time_col] = _parse_time_series(df[time_col])
        now = pd.Timestamp.utcnow()
        cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
        df = df[df[time_col] >= cutoff]
    if df.empty:
        return pd.DataFrame()

    df = df[df["has_sweep"] == True]
    df = df[df["total_premium"] >= UOA_MIN_PREMIUM]
    if df.empty:
        return pd.DataFrame()

    if "volume" in df.columns and "open_interest" in df.columns:
        df = df[df["volume"] > df["open_interest"]]
    if df.empty:
        return pd.DataFrame()

    df = _add_dte_if_missing(df, time_col)
    if "dte" in df.columns:
        df = df[(df["dte"] >= 1) & (df["dte"] <= 56)]

    df = _compute_pct_otm(df)
    if "pct_otm" in df.columns:
        df = df[df["pct_otm"].abs() <= 15]

    df = _add_direction_from_type(df)
    df = df[df["direction"].notna()]

    if "trade_count" in df.columns:
        df = df[df["trade_count"] >= 2]
    if df.empty:
        return pd.DataFrame()

    now = pd.Timestamp.utcnow()
    rec = (
        df[time_col].apply(lambda x: _format_recency(x, now))
        if time_col and time_col in df.columns
        else None
    )

    out = pd.DataFrame()
    out["Ticker"] = df["ticker"]
    out["Direction"] = df["direction"]
    out["Total Premium ($)"] = df["total_premium"]
    if "trade_count" in df.columns:
        out["# Trades"] = df["trade_count"]
    if "dte" in df.columns:
        out["DTE"] = df["dte"]
    if "pct_otm" in df.columns:
        out["% OTM"] = df["pct_otm"]
    out["Strike"] = df["strike"]
    out["Underlying Price"] = df["underlying_price"]
    if "expiry" in df.columns:
        out["Expiration"] = df["expiry"]
    if rec is not None:
        out["Recency"] = rec

    out["Total Premium ($)"] = out["Total Premium ($)"].round(0).astype(int)
    if "% OTM" in out.columns:
        out["% OTM"] = out["% OTM"].round(2)
    if "Underlying Price" in out.columns:
        out["Underlying Price"] = out["Underlying Price"].round(2)
    if "Strike" in out.columns:
        out["Strike"] = out["Strike"].round(2)

    out = out.sort_values("Total Premium ($)", ascending=False).reset_index(drop=True)
    return out


def apply_soft_alerts(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Softer "heads-up" sweeps:
      - Sweeps only
      - 40k <= total_premium < 150k
      - Same recency window (3 days)
      - Direction by type (calls/puts)
    These are NOT trade signals by themselves.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = _normalize_and_numeric(df_raw)
    required = ["ticker", "total_premium", "has_sweep", "strike", "underlying_price", "type"]
    if any(r not in df.columns for r in required):
        return pd.DataFrame()

    time_col = _detect_time_col(df)
    if time_col:
        df[time_col] = _parse_time_series(df[time_col])
        now = pd.Timestamp.utcnow()
        cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
        df = df[df[time_col] >= cutoff]
    if df.empty:
        return pd.DataFrame()

    df = df[df["has_sweep"] == True]
    df = df[
        (df["total_premium"] >= SOFT_MIN_PREMIUM)
        & (df["total_premium"] < UOA_MIN_PREMIUM)
    ]
    if df.empty:
        return pd.DataFrame()

    df = _compute_pct_otm(df)
    df = _add_direction_from_type(df)
    df = df[df["direction"].notna()]
    if df.empty:
        return pd.DataFrame()

    now = pd.Timestamp.utcnow()
    rec = (
        df[time_col].apply(lambda x: _format_recency(x, now))
        if time_col and time_col in df.columns
        else None
    )

    out = pd.DataFrame()
    out["Ticker"] = df["ticker"]
    out["Direction"] = df["direction"]
    out["Premium ($)"] = df["total_premium"]
    out["Strike"] = df["strike"]
    out["Underlying Price"] = df["underlying_price"]
    if "expiry" in df.columns:
        out["Expiration"] = df["expiry"]
    if rec is not None:
        out["Recency"] = rec
    if "pct_otm" in df.columns:
        out["% OTM"] = df["pct_otm"]
    if time_col and time_col in df.columns:
        out["Time"] = df[time_col]

    out["Premium ($)"] = out["Premium ($)"].round(0).astype(int)
    if "% OTM" in out.columns:
        out["% OTM"] = out["% OTM"].round(2)
    if "Underlying Price" in out.columns:
        out["Underlying Price"] = out["Underlying Price"].round(2)
    if "Strike" in out.columns:
        out["Strike"] = out["Strike"].round(2)

    sort_cols = []
    if "Time" in out.columns:
        sort_cols.append("Time")
    sort_cols.append("Premium ($)")
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(
        drop=True
    )
    return out


def verdict_for_direction(direction: str) -> str:
    if direction == "bullish":
        return "BUY CALLS"
    if direction == "bearish":
        return "BUY PUTS"
    return "NO TRADE"

# ==============================
#  UI: MODE TOGGLE
# ==============================

st.markdown(
    """
    # MACD UOA SCANNER LOGIC — MACH 7.4

    Unified scanner with two modules:
    - **MACD Scanner** — Daily bullish-from-below MACD cross finder (Mach 6.2 core)
    - **UOA Scanner (STRICT)** — Institutional sweeps (last 3 days, UOA 2.0, Option C)
    """.strip()
)

mode = st.radio(
    "Select Scanner Mode:",
    ["MACD Scanner", "UOA Scanner (STRICT UOA 2.0)"],
    index=0,
    horizontal=True,
)

# ==============================
#  MODE 1: MACD SCANNER
# ==============================

if mode == "MACD Scanner":
    st.markdown("### MACD Scanner — Mach 6.2 Core (Bullish-Only, From Below)")
    st.caption("From-below MACD(12,26,9) crosses on your static S&P list.")

    st.button("🔄 Rescan MACD Now")

    with st.container():
        st.success("✅ MACD scan ready (Mach 6.2 core).")

    st.markdown("### Scan Summary")

    with st.spinner("Downloading price history and running MACD scans..."):
        history = download_history(SP500_TICKERS, period="1y", interval="1d")
        about_df, just_df, already_df, skipped_df = build_tables(history)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("About to Cross (Bullish)", len(about_df))
    col2.metric("Just Crossed (Bullish)", len(just_df))
    col3.metric("Already Crossed (Bullish)", len(already_df))
    col4.metric("Processed", len(history))

    with st.expander("Skipped / Not in a bullish-from-below setup (reason)"):
        st.write(
            "Tickers that didn't meet the strict bullish-from-below MACD filter or lacked sufficient history."
        )
        if not skipped_df.empty:
            st.dataframe(
                skipped_df.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("None skipped under current rules.")

    st.markdown("---")

    st.markdown("## 1) About to Cross (Bullish)")
    st.caption(
        "MACD is below Signal, rising, and within a tight proximity band — likely to cross soon."
    )
    if not about_df.empty:
        st.dataframe(
            about_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No tickers are in an 'about to cross' setup right now.")

    st.markdown("---")

    st.markdown("## 2) Just Crossed (Bullish)")
    st.caption(
        "MACD has just crossed above Signal from below within the last few days — fresh bullish signal."
    )
    if not just_df.empty:
        st.dataframe(
            just_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No recent bullish-from-below crosses in the last few days.")

    st.markdown("---")

    st.markdown("## 3) Already Crossed (Bullish)")
    st.caption(
        "MACD is above Signal after a prior bullish-from-below cross — ongoing bullish trend."
    )
    if not already_df.empty:
        st.dataframe(
            already_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No ongoing bullish-from-below MACD trends under current filters.")

    st.markdown("---")

    st.markdown("### Export Results")
    col_a, col_b, col_c = st.columns(3)

    about_csv = about_df.to_csv(index=False).encode("utf-8")
    just_csv = just_df.to_csv(index=False).encode("utf-8")
    already_csv = already_df.to_csv(index=False).encode("utf-8")

    col_a.download_button(
        "Download 'About to Cross' CSV",
        about_csv,
        "about_to_cross_macd.csv",
        "text/csv",
    )
    col_b.download_button(
        "Download 'Just Crossed' CSV",
        just_csv,
        "just_crossed_macd.csv",
        "text/csv",
    )
    col_c.download_button(
        "Download 'Already Crossed' CSV",
        already_csv,
        "already_crossed_macd.csv",
        "text/csv",
    )

    st.caption("Mach 6.2 core · SilverFoxFlow · MACD(12,26,9) · From-below bullish only.")

# ==============================
#  MODE 2: UOA SCANNER (STRICT)
# ==============================

else:
    st.markdown("### UOA Scanner (STRICT UOA 2.0) — Institutional Flow (Last 3 Days)")
    st.caption("Sweeps-only, $150K+ clusters, volume>OI, 1–8wk, ±15% OTM, ≥2 trades. Option C: calls→bullish, puts→bearish.")

    debug = st.checkbox("Dev debug: show raw UW data", value=False)

    st.button("🔄 Rescan UOA Now")

    with st.spinner("Pulling STRICT institutional flow from Unusual Whales (last 3 days)..."):
        df_raw = fetch_uoa_flow_alerts_last_3_days(max_pages=8, per_page=500)
        strict_df = apply_strict_uoa_filters(df_raw)
        soft_df = apply_soft_alerts(df_raw)

    total_alerts = len(df_raw)
    num_signals = len(strict_df)
    num_bullish = (
        (strict_df["Direction"] == "bullish").sum() if not strict_df.empty else 0
    )
    num_bearish = (
        (strict_df["Direction"] == "bearish").sum() if not strict_df.empty else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts Pulled (3 days)", total_alerts)
    col2.metric("Strict Signals", num_signals)
    col3.metric("Bullish Signals", int(num_bullish))
    col4.metric("Bearish Signals", int(num_bearish))

    if debug and not df_raw.empty:
        with st.expander("🔍 Raw Unusual Whales Data (Dev Debug)", expanded=False):
            st.write("Columns:", list(df_raw.columns))
            st.dataframe(df_raw.head(30), use_container_width=True)

    # ---- Soft Alerts (yellow heads-up) ----
    st.markdown("#### ⚠️ Soft Alerts (Heads-Up Only — NOT Trade Signals)")
    if not soft_df.empty:
        for _, row in soft_df.head(5).iterrows():
            tkr = row.get("Ticker", "")
            direction = row.get("Direction", "")
            prem = row.get("Premium ($)", np.nan)
            pct = row.get("% OTM", np.nan)
            exp = row.get("Expiration", "")
            rec = row.get("Recency", None)

            msg_parts = [f"{tkr} {str(direction).upper()} sweep"]
            if not pd.isna(prem):
                msg_parts.append(f"~${int(prem):,}")
            if not pd.isna(pct):
                msg_parts.append(f"{pct:.1f}% OTM")
            if isinstance(exp, str) and exp:
                msg_parts.append(f"exp {exp}")
            if rec:
                msg_parts.append(rec)

            st.warning(" · ".join(msg_parts))
    else:
        st.caption("No soft sweeps detected in this 3-day window.")

    st.markdown("---")

    # ---- Strict Flow Table ----
    st.markdown("## Strict Institutional Flow Table (Last 3 Days)")
    st.caption("Filtered UOA 2.0 — best-of-the-best institutional sweeps only, labeled by recency.")

    if not strict_df.empty:
        st.dataframe(strict_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No strict institutional UOA signals found under current rules in this 3-day window."
        )

    st.markdown("---")

# ---- Verdict Cards (Consolidated) ----
st.markdown("## Signal Cards (Consolidated)")
st.caption("Each ticker aggregated into one clean card instead of repeated rows.")

if strict_df.empty:
    st.write("No trades to summarize. Verdict: **NO TRADE**.")

else:
    grouped = strict_df.groupby("Ticker")

    for ticker, grp in grouped:
        st.markdown("---")

        # Determine direction (majority)
        direction = grp["Direction"].mode()[0]
        verdict = verdict_for_direction(direction)

        # Aggregate metrics
        total_premium = grp["Total Premium ($)"].sum()
        total_trades = grp["# Trades"].sum() if "# Trades" in grp.columns else len(grp)
        avg_pct_otm = grp["% OTM"].mean() if "% OTM" in grp.columns else None
        avg_dte = grp["DTE"].mean() if "DTE" in grp.columns else None
        avg_strike = grp["Strike"].mean() if "Strike" in grp.columns else None
        avg_underlying = grp["Underlying Price"].mean() if "Underlying Price" in grp.columns else None
        common_exp = grp["Expiration"].mode()[0] if "Expiration" in grp.columns else None
        recency_tags = grp["Recency"].dropna().unique()

        # Title
        title = f"### {ticker} — **{verdict}**"
        if any(isinstance(r, str) and r.startswith("Fresh") for r in recency_tags):
            title += " 🔥"
        st.markdown(title)

        # Bias Line
        if direction == "bullish":
            st.write("Bias: 🟢 Institutional **bullish** flow (CALL sweeps).")
        else:
            st.write("Bias: 🔴 Institutional **bearish** flow (PUT sweeps).")

        # Summary Lines
        summary = []

        if len(recency_tags) > 0:
            summary.append(f"Recency: **{', '.join([str(r) for r in recency_tags])}**")

        summary.append(f"Total Premium (combined): **${total_premium:,.0f}**")
        summary.append(f"Sweeps in cluster: **{int(total_trades)}**")

        if avg_dte is not None:
            summary.append(f"Avg DTE: **{avg_dte:.1f} days**")
        if avg_strike is not None:
            summary.append(f"Avg Strike: **{avg_strike:.2f}**")
        if avg_underlying is not None:
            summary.append(f"Underlying: **{avg_underlying:.2f}**")
        if common_exp:
            summary.append(f"Expiration: **{common_exp}**")
        if avg_pct_otm is not None:
            summary.append(f"Avg % OTM: **{avg_pct_otm:.2f}%**")

        st.write(" • ".join(summary))

        st.caption(
            "Strict UOA 2.0 aggregated: sweeps-only · $150K+ premium · volume>OI · 1–8wk expirations · clustered institutions · last 3 days."
        )

st.markdown("---")
