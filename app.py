# SilverFoxFlow Mach 8.0 — Recovery MACD Scanner
# Run: streamlit run app.py
# Purpose: fewer trades, stronger setups, faster failed-cross warnings.

import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SilverFoxFlow Mach 8.0", page_icon="🦊", layout="wide")

# ============================================================
# UNIVERSE
# ============================================================

# Kept from your Mach 7.4 file. Mach 8.0 ranks this list down to a tradable 100/150/200.
SP500_TICKERS = ['AAPL',
 'MSFT',
 'GOOG',
 'GOOGL',
 'AMZN',
 'NVDA',
 'AVGO',
 'META',
 'TSLA',
 'BRK.B',
 'LLY',
 'JPM',
 'WMT',
 'ORCL',
 'V',
 'MA',
 'XOM',
 'PLTR',
 'NFLX',
 'JNJ',
 'AMD',
 'COST',
 'BAC',
 'ABBV',
 'HD',
 'PG',
 'GE',
 'CVX',
 'UNH',
 'KO',
 'CSCO',
 'IBM',
 'WFC',
 'CAT',
 'MS',
 'MU',
 'GS',
 'AXP',
 'CRM',
 'MRK',
 'RTX',
 'TMUS',
 'PM',
 'APP',
 'ABT',
 'TMO',
 'MCD',
 'DIS',
 'UBER',
 'PEP',
 'ANET',
 'LRCX',
 'LIN',
 'QCOM',
 'NOW',
 'INTC',
 'ISRG',
 'INTU',
 'AMAT',
 'C',
 'BX',
 'BLK',
 'T',
 'SCHW',
 'APH',
 'NEE',
 'VZ',
 'BKNG',
 'AMGN',
 'KLAC',
 'GEV',
 'TJX',
 'ACN',
 'BA',
 'DHR',
 'BSX',
 'PANW',
 'GILD',
 'ETN',
 'SPGI',
 'TXN',
 'ADBE',
 'PFE',
 'COF',
 'CRWD',
 'SYK',
 'LOW',
 'UNP',
 'HOOD',
 'HON',
 'DE',
 'WELL',
 'PGR',
 'PLD',
 'CEG',
 'MDT',
 'ADI',
 'LMT',
 'COP',
 'VRTX',
 'CB',
 'DASH',
 'DELL',
 'HCA',
 'KKR',
 'ADP',
 'SO',
 'CMCSA',
 'MCK',
 'TT',
 'CVS',
 'PH',
 'DUK',
 'CME',
 'NKE',
 'MO',
 'BMY',
 'GD',
 'CDNS',
 'SBUX',
 'MMM',
 'NEM',
 'COIN',
 'MMC',
 'MCO',
 'SHW',
 'SNPS',
 'AMT',
 'ICE',
 'NOC',
 'EQIX',
 'HWM',
 'UPS',
 'WM',
 'ORLY',
 'EMR',
 'RCL',
 'ABNB',
 'BK',
 'JCI',
 'MDLZ',
 'TDG',
 'CTAS',
 'AON',
 'TEL',
 'ECL',
 'USB',
 'GLW',
 'PNC',
 'APO',
 'ITW',
 'MAR',
 'WMB',
 'ELV',
 'MSI',
 'CSX',
 'PWR',
 'REGN',
 'SPG',
 'FTNT',
 'COR',
 'MNST',
 'CI',
 'PYPL',
 'GM',
 'RSG',
 'AEP',
 'ADSK',
 'AJG',
 'WDAY',
 'ZTS',
 'VST',
 'NSC',
 'CL',
 'AZO',
 'CMI',
 'SRE',
 'TRV',
 'FDX',
 'FCX',
 'HLT',
 'DLR',
 'MPC',
 'KMI',
 'EOG',
 'AXON',
 'AFL',
 'TFC',
 'DDOG',
 'WBD',
 'URI',
 'PSX',
 'STX',
 'LHX',
 'APD',
 'SLB',
 'O',
 'MET',
 'NXPI',
 'F',
 'VLO',
 'ROST',
 'PCAR',
 'WDC',
 'BDX',
 'ALL',
 'IDXX',
 'CARR',
 'D',
 'EA',
 'PSA',
 'NDAQ',
 'EW',
 'MPWR',
 'ROP',
 'XEL',
 'BKR',
 'TTWO',
 'FAST',
 'GWW',
 'AME',
 'EXC',
 'XYZ',
 'CAH',
 'CBRE',
 'MSCI',
 'DHI',
 'AIG',
 'ETR',
 'KR',
 'OKE',
 'AMP',
 'TGT',
 'PAYX',
 'CMG',
 'CTVA',
 'CPRT',
 'A',
 'FANG',
 'ROK',
 'GRMN',
 'OXY',
 'PEG',
 'LVS',
 'FICO',
 'KMB',
 'CCI',
 'YUM',
 'VMC',
 'CCL',
 'TKO',
 'DAL',
 'EBAY',
 'MLM',
 'KDP',
 'IQV',
 'XYL',
 'PRU',
 'WEC',
 'OTIS',
 'RMD',
 'FI',
 'CHTR',
 'SYY',
 'CTSH',
 'ED',
 'PCG',
 'WAB',
 'VTR',
 'EL',
 'LYV',
 'HIG',
 'NUE',
 'HSY',
 'DD',
 'GEHC',
 'MCHP',
 'HUM',
 'EQT',
 'NRG',
 'TRGP',
 'FIS',
 'STT',
 'HPE',
 'VICI',
 'ACGL',
 'LEN',
 'KEYS',
 'RJF',
 'IBKR',
 'SMCI',
 'VRSK',
 'UAL',
 'IRM',
 'EME',
 'IR',
 'WTW',
 'EXR',
 'ODFL',
 'KHC',
 'MTD',
 'CSGP',
 'ADM',
 'TER',
 'K',
 'FOXA',
 'TSCO',
 'FSLR',
 'MTB',
 'DTE',
 'ROL',
 'AEE',
 'KVUE',
 'ATO',
 'FITB',
 'ES',
 'FOX',
 'BRO',
 'EXPE',
 'WRB',
 'PPL',
 'SYF',
 'FE',
 'HPQ',
 'EFX',
 'BR',
 'CBOE',
 'AWK',
 'HUBB',
 'CNP',
 'DOV',
 'GIS',
 'AVB',
 'TDY',
 'EXE',
 'TTD',
 'VLTO',
 'LDOS',
 'NTRS',
 'HBAN',
 'CINF',
 'PTC',
 'WSM',
 'JBL',
 'NTAP',
 'PHM',
 'ULTA',
 'STE',
 'EQR',
 'STZ',
 'STLD',
 'TPR',
 'DXCM',
 'BIIB',
 'HAL',
 'TROW',
 'VRSN',
 'PODD',
 'CMS',
 'CFG',
 'PPG',
 'DG',
 'TPL',
 'RF',
 'CHD',
 'EIX',
 'LH',
 'DRI',
 'CDW',
 'WAT',
 'L',
 'NVR',
 'DVN',
 'SBAC',
 'TYL',
 'ON',
 'IP',
 'WST',
 'LULU',
 'NI',
 'DLTR',
 'ZBH',
 'KEY',
 'DGX',
 'RL',
 'SW',
 'TRMB',
 'BG',
 'GPN',
 'IT',
 'J',
 'PFG',
 'CPAY',
 'TSN',
 'INCY',
 'AMCR',
 'CHRW',
 'CTRA',
 'GDDY',
 'LII',
 'GPC',
 'EVRG',
 'APTV',
 'PKG',
 'SNA',
 'PNR',
 'CNC',
 'INVH',
 'BBY',
 'MKC',
 'LNT',
 'DOW',
 'PSKY',
 'ESS',
 'WY',
 'EXPD',
 'HOLX',
 'GEN',
 'IFF',
 'JBHT',
 'FTV',
 'LUV',
 'NWS',
 'MAA',
 'ERIE',
 'LYB',
 'NWSA',
 'FFIV',
 'OMC',
 'ALLE',
 'TXT',
 'KIM',
 'COO',
 'UHS',
 'CLX',
 'ZBRA',
 'AVY',
 'CF',
 'DPZ',
 'MAS',
 'EG',
 'NDSN',
 'BF.B',
 'BLDR',
 'IEX',
 'BALL',
 'DOC',
 'HII',
 'BXP',
 'REG',
 'WYNN',
 'UDR',
 'DECK',
 'VTRS',
 'SOLV',
 'HRL',
 'BEN',
 'ALB',
 'SWKS',
 'HST',
 'SJM',
 'DAY',
 'RVTY',
 'JKHY',
 'CPT',
 'AKAM',
 'HAS',
 'AIZ',
 'MRNA',
 'PNW',
 'GL',
 'IVZ',
 'PAYC',
 'SWK',
 'NCLH',
 'ARE',
 'ALGN',
 'FDS',
 'POOL',
 'AES',
 'GNRC',
 'TECH',
 'BAX']

MARKET_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_ETFS = ["SMH", "XLK", "XLC", "XLY", "XLP", "XLF", "XLI", "XLE", "XLV", "XLU", "XLB", "XLRE"]
CORE_ETFS = MARKET_ETFS + SECTOR_ETFS

LIQUID_OPTIONS_BOOST = set([
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLK", "XLF", "XLI", "XLE", "XLV", "XLU",
    "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "GOOG", "AVGO", "AMD", "TSLA",
    "NFLX", "CRM", "ORCL", "ADBE", "NOW", "INTU", "PANW", "CRWD", "PLTR", "COIN",
    "MSTR", "HOOD", "SHOP", "SNOW", "DDOG", "NET", "ROKU", "AFRM", "RBLX", "U",
    "MU", "QCOM", "LRCX", "KLAC", "AMAT", "TSM", "ARM", "MRVL", "TXN", "ON", "INTC", "SMCI",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "AXP", "V", "MA", "BLK", "COF", "PYPL",
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "EOG", "MPC", "VLO", "FCX", "NEM", "ALB",
    "CAT", "DE", "GE", "ETN", "PH", "CMI", "URI", "HON", "BA", "RTX", "LMT", "NOC", "GD",
    "LLY", "UNH", "ABBV", "MRK", "AMGN", "ISRG", "TMO", "DHR", "PFE", "JNJ",
    "COST", "WMT", "MCD", "HD", "LOW", "NKE", "DIS", "UBER", "SBUX", "CMG", "TGT",
])
EXTRA_LIQUID_OPTIONS = sorted(LIQUID_OPTIONS_BOOST - set(SP500_TICKERS))

SECTOR_BUCKETS = {
    "Semiconductors": ["NVDA", "AVGO", "AMD", "MU", "QCOM", "LRCX", "KLAC", "AMAT", "TSM", "ARM", "MRVL", "TXN", "ON", "INTC", "SMCI", "MCHP", "MPWR", "NXPI", "ADI", "SWKS"],
    "Technology": ["AAPL", "MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "IBM", "PANW", "CRWD", "FTNT", "DDOG", "SNOW", "NET", "ZS", "MDB", "CDNS", "SNPS", "ANET", "HPE", "DELL", "HPQ", "PLTR"],
    "Communication": ["META", "GOOG", "GOOGL", "NFLX", "DIS", "TMUS", "VZ", "T", "CMCSA", "CHTR", "WBD", "TTD", "ROKU"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "LOW", "NKE", "SBUX", "MCD", "CMG", "BKNG", "ABNB", "RCL", "CCL", "LVS", "WYNN", "MAR", "HLT", "GM", "F", "UBER", "DASH", "SHOP", "RBLX"],
    "Consumer Staples": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "KHC", "KR", "TGT", "CL", "KMB", "GIS", "HSY", "KDP", "SYY"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK", "SCHW", "COF", "PYPL", "BX", "KKR", "CME", "ICE", "NDAQ", "USB", "PNC", "TFC", "FITB", "RF", "HBAN", "COIN", "MSTR", "HOOD"],
    "Healthcare": ["LLY", "UNH", "ABBV", "MRK", "JNJ", "AMGN", "ISRG", "TMO", "DHR", "PFE", "ABT", "GILD", "VRTX", "REGN", "CVS", "HCA", "CI", "ELV", "SYK", "BSX", "MDT", "BMY", "BIIB", "MRNA", "HUM"],
    "Industrials": ["CAT", "DE", "GE", "ETN", "PH", "CMI", "URI", "HON", "BA", "RTX", "LMT", "NOC", "GD", "EMR", "MMM", "UNP", "CSX", "NSC", "UPS", "FDX", "WM", "RSG", "PWR", "TT", "PCAR", "LHX", "HWM", "AXON", "JCI", "AME"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "HAL", "OXY", "EOG", "MPC", "VLO", "PSX", "FANG", "BKR", "OKE", "KMI", "WMB", "TRGP", "EQT", "DVN"],
    "Materials": ["FCX", "NEM", "ALB", "NUE", "STLD", "CLF", "LIN", "APD", "ECL", "SHW", "DD", "DOW", "PPG", "CF", "VMC", "MLM", "PKG"],
    "Utilities": ["NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "EXC", "XEL", "ETR", "PEG", "ED", "PCG", "WEC", "EIX", "PPL", "FE", "AES", "NRG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "DLR", "PSA", "O", "WELL", "VICI", "CBRE", "CCI", "VTR", "EQR", "AVB", "EXR", "INVH", "SBAC", "BXP", "ARE", "HST"],
}
TICKER_TO_SECTOR = {ticker: sector for sector, tickers in SECTOR_BUCKETS.items() for ticker in tickers}
SECTOR_TO_ETF = {
    "Semiconductors": "SMH", "Technology": "XLK", "Communication": "XLC",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Financials": "XLF",
    "Healthcare": "XLV", "Industrials": "XLI", "Energy": "XLE", "Materials": "XLB",
    "Utilities": "XLU", "Real Estate": "XLRE", "ETF": "SPY", "Other": "SPY",
}

MODE_CONFIG = {
    "Recovery": {
        "label": "Strict. Capital-protection first.",
        "min_proximity": 70, "min_volume_ratio": 0.80, "min_rs_spy_20": 0.00,
        "min_stock_vs_sector_20": 0.00, "allow_hostile_market": False, "max_recent_cross_bars": 5,
    },
    "Normal": {
        "label": "Balanced. Still confirmation-based.",
        "min_proximity": 60, "min_volume_ratio": 0.70, "min_rs_spy_20": -0.01,
        "min_stock_vs_sector_20": -0.01, "allow_hostile_market": False, "max_recent_cross_bars": 6,
    },
    "Aggressive": {
        "label": "More alerts, more fakeout risk.",
        "min_proximity": 50, "min_volume_ratio": 0.60, "min_rs_spy_20": -0.03,
        "min_stock_vs_sector_20": -0.03, "allow_hostile_market": True, "max_recent_cross_bars": 8,
    },
}

DISPLAY_COLUMNS = [
    "Grade", "Ticker", "Sector", "Price", "MACD Status", "Score", "Cross Proximity %",
    "Hist Trend", "RS vs SPY 20D", "Stock vs Sector 20D", "Sector State", "Above 50SMA",
    "Above 200EMA", "Vol Ratio", "Trigger", "Invalidation", "Risk Flags", "Action", "Chart",
]

# ============================================================
# HELPERS
# ============================================================

def yf_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").upper().strip()


def tv_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").upper().strip()


@st.cache_data(show_spinner=False, ttl=60 * 20)
def download_history(tickers, period="1y", interval="1d", batch_size=50):
    clean = []
    seen = set()
    for t in tickers:
        t = str(t).upper().strip()
        if t and t not in seen:
            clean.append(t)
            seen.add(t)
    symbol_to_ticker = {yf_symbol(t): t for t in clean}
    all_data = {}
    for i in range(0, len(clean), batch_size):
        batch_tickers = clean[i:i + batch_size]
        batch_symbols = [yf_symbol(t) for t in batch_tickers]
        try:
            df = yf.download(
                batch_symbols, period=period, interval=interval, auto_adjust=False,
                progress=False, group_by="ticker", threads=True,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            top = list(df.columns.get_level_values(0).unique())
            for sym in batch_symbols:
                if sym in top:
                    sub = df[sym].dropna(how="all")
                    if not sub.empty:
                        all_data[symbol_to_ticker.get(sym, sym)] = sub
        else:
            sym = batch_symbols[0]
            sub = df.dropna(how="all")
            if not sub.empty:
                all_data[symbol_to_ticker.get(sym, sym)] = sub
    return all_data


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def pct_return(close: pd.Series, bars: int):
    if close is None or len(close) <= bars:
        return np.nan
    start = close.iloc[-bars - 1]
    end = close.iloc[-1]
    if pd.isna(start) or start == 0:
        return np.nan
    return float(end / start - 1.0)


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def last_bull_cross_info(macd: pd.Series, sig: pd.Series, high: pd.Series, low: pd.Series, lookback=80):
    common = macd.dropna().index.intersection(sig.dropna().index)
    if len(common) < 3:
        return None
    macd = macd.loc[common]
    sig = sig.loc[common]
    recent = macd.iloc[-lookback:]
    recent_sig = sig.loc[recent.index]
    last_idx = None
    for i in range(1, len(recent)):
        if recent.iloc[i - 1] <= recent_sig.iloc[i - 1] and recent.iloc[i] > recent_sig.iloc[i]:
            last_idx = recent.index[i]
    if last_idx is None:
        return None
    bars_since = len(macd.loc[last_idx:]) - 1
    return {
        "date": last_idx,
        "bars_since": int(bars_since),
        "high": float(high.loc[last_idx]) if last_idx in high.index else np.nan,
        "low": float(low.loc[last_idx]) if last_idx in low.index else np.nan,
    }


def consecutive_hist_trend(hist: pd.Series):
    if len(hist) < 4:
        return "Flat/Unknown", 0
    h0, h1, h2, h3 = hist.iloc[-1], hist.iloc[-2], hist.iloc[-3], hist.iloc[-4]
    if h0 > h1 > h2 > h3:
        return "4D Improving", 4
    if h0 > h1 > h2:
        return "3D Improving", 3
    if h0 > h1:
        return "2D Improving", 2
    if h0 < h1 < h2:
        return "Fading", -2
    return "Mixed", 0


def cross_proximity(macd_curr, sig_curr, hist: pd.Series):
    if pd.isna(macd_curr) or pd.isna(sig_curr):
        return 0.0
    if macd_curr >= sig_curr:
        return 100.0
    gap = abs(macd_curr - sig_curr)
    ref = hist.abs().rolling(60).median().iloc[-1] if len(hist) >= 60 else hist.abs().median()
    if pd.isna(ref) or ref == 0:
        ref = max(abs(sig_curr), 1.0) * 0.01
    return float(max(0.0, min(100.0, 100.0 * (1.0 - min(gap / ref, 1.0)))))


def get_sector(ticker: str):
    t = ticker.upper().strip()
    if t in CORE_ETFS:
        return "ETF"
    return TICKER_TO_SECTOR.get(t, "Other")


def sector_etf_for(ticker: str):
    if ticker in CORE_ETFS:
        return ticker
    return SECTOR_TO_ETF.get(get_sector(ticker), "SPY")


def sector_state(sector_etf: str, data_map: dict, spy_ret20: float):
    df = data_map.get(sector_etf)
    if df is None or df.empty:
        return "Unknown", np.nan
    close = get_series(df, "Close")
    if len(close) < 55:
        return "Unknown", np.nan
    ret20 = pct_return(close, 20)
    sma50 = close.rolling(50).mean().iloc[-1]
    above50 = close.iloc[-1] > sma50 if not pd.isna(sma50) else False
    rs = ret20 - spy_ret20 if not pd.isna(ret20) and not pd.isna(spy_ret20) else np.nan
    if not pd.isna(rs) and rs >= 0.02 and above50:
        return "Strong", float(rs)
    if (not pd.isna(rs) and rs >= -0.02) or above50:
        return "Neutral", float(rs) if not pd.isna(rs) else np.nan
    return "Weak", float(rs) if not pd.isna(rs) else np.nan


def analyze_market_regime(data_map: dict):
    rows = []
    points = 0
    for ticker in ["SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLI", "XLE", "XLV"]:
        df = data_map.get(ticker)
        close = get_series(df, "Close")
        if len(close) < 60:
            continue
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        ret1 = pct_return(close, 1)
        ret5 = pct_return(close, 5)
        ret20 = pct_return(close, 20)
        above20 = close.iloc[-1] > sma20
        above50 = close.iloc[-1] > sma50
        above200 = close.iloc[-1] > ema200
        if ticker in ["SPY", "QQQ"]:
            points += int(above20) + int(above50) + int(above200) + int(ret5 > 0)
            if ticker == "SPY" and not pd.isna(ret1) and ret1 < -0.012:
                points -= 1
            if ticker == "QQQ" and not pd.isna(ret1) and ret1 < -0.015:
                points -= 1
        rows.append({
            "Ticker": ticker,
            "Price": round(float(close.iloc[-1]), 2),
            "1D %": round(ret1 * 100, 2) if not pd.isna(ret1) else np.nan,
            "5D %": round(ret5 * 100, 2) if not pd.isna(ret5) else np.nan,
            "20D %": round(ret20 * 100, 2) if not pd.isna(ret20) else np.nan,
            "Above 20SMA": "Yes" if above20 else "No",
            "Above 50SMA": "Yes" if above50 else "No",
            "Above 200EMA": "Yes" if above200 else "No",
        })
    if points >= 6:
        return "BULLISH", "ALLOWED", "Trade only A/A+ setups", pd.DataFrame(rows), points
    if points >= 3:
        return "NEUTRAL", "CAUTION", "Watch first, confirm entries", pd.DataFrame(rows), points
    return "HOSTILE", "OFF", "Cash / watch only unless hedging", pd.DataFrame(rows), points


def estimate_options_tradability(ticker, price, avg_dollar_vol20, avg_volume20):
    t = ticker.upper().strip()
    if t in CORE_ETFS:
        return True, "ETF liquid"
    if t in LIQUID_OPTIONS_BOOST:
        return True, "Liquid options list"
    if price >= 20 and avg_volume20 >= 1_500_000 and avg_dollar_vol20 >= 150_000_000:
        return True, "High stock liquidity proxy"
    if price >= 20 and avg_dollar_vol20 >= 500_000_000:
        return True, "High dollar-volume proxy"
    return False, "Weak options proxy"


def rank_universe(data_map: dict, candidates: list, include_etfs=True):
    rows = []
    for t in candidates:
        if not include_etfs and t in CORE_ETFS:
            continue
        df = data_map.get(t)
        close = get_series(df, "Close")
        volume = get_series(df, "Volume")
        if len(close) < 30 or len(volume) < 20:
            continue
        price = float(close.iloc[-1])
        avg_vol20 = float(volume.tail(20).mean())
        avg_dollar_vol20 = price * avg_vol20
        boost = 1.50 if t in LIQUID_OPTIONS_BOOST else 0.0
        etf_boost = 0.75 if t in CORE_ETFS else 0.0
        price_ok = 0.30 if 20 <= price <= 900 else -0.50
        score = math.log10(max(avg_dollar_vol20, 1.0)) + boost + etf_boost + price_ok
        rows.append({"Ticker": t, "Tradability Score": round(score, 3), "Avg $Vol 20D": avg_dollar_vol20})
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Tradability Score", "Avg $Vol 20D"])
    return pd.DataFrame(rows).sort_values("Tradability Score", ascending=False).reset_index(drop=True)


def analyze_ticker(ticker: str, data_map: dict, market_regime: str, mode: str, ignore_market_gate=False):
    cfg = MODE_CONFIG[mode]
    df = data_map.get(ticker)
    close = get_series(df, "Close")
    high = get_series(df, "High")
    low = get_series(df, "Low")
    volume = get_series(df, "Volume")
    common = close.index.intersection(high.index).intersection(low.index).intersection(volume.index)
    if len(common) < 80:
        return None
    close, high, low, volume = close.loc[common], high.loc[common], low.loc[common], volume.loc[common]

    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    prior_high = float(high.iloc[-2])
    prior_low = float(low.iloc[-2])
    macd, sig, hist = compute_macd(close)
    macd_curr, sig_curr = float(macd.iloc[-1]), float(sig.iloc[-1])
    macd_prev, sig_prev = float(macd.iloc[-2]), float(sig.iloc[-2])
    hist_curr = float(hist.iloc[-1])

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    avg_vol20 = float(volume.tail(20).mean())
    avg_dollar_vol20 = price * avg_vol20
    vol_ratio = float(volume.iloc[-1] / avg_vol20) if avg_vol20 > 0 else np.nan
    high_252 = float(close.tail(252).max()) if len(close) >= 100 else np.nan
    pct_from_52w_high = price / high_252 - 1 if high_252 and high_252 > 0 else np.nan

    above20 = price > sma20 if not pd.isna(sma20) else False
    above50 = price > sma50 if not pd.isna(sma50) else False
    above200 = price > ema200 if not pd.isna(ema200) else False
    trend_ok = above50 and above200

    hist_label, hist_count = consecutive_hist_trend(hist)
    proximity = cross_proximity(macd_curr, sig_curr, hist)
    cross_info = last_bull_cross_info(macd, sig, high, low, lookback=80)
    bars_since_cross = cross_info["bars_since"] if cross_info else None
    cross_day_high = cross_info["high"] if cross_info else np.nan
    cross_day_low = cross_info["low"] if cross_info else np.nan

    crossed_today = macd_prev <= sig_prev and macd_curr > sig_curr
    recently_crossed = cross_info is not None and bars_since_cross <= cfg["max_recent_cross_bars"]
    about_to_cross = (
        macd_curr < sig_curr and hist_curr < 0 and hist_count >= 2
        and macd_curr > macd_prev and proximity >= cfg["min_proximity"]
    )
    confirmed = (
        recently_crossed and bars_since_cross is not None and bars_since_cross >= 1
        and macd_curr > sig_curr and price > cross_day_high and price > prev_close
        and (pd.isna(vol_ratio) or vol_ratio >= cfg["min_volume_ratio"])
    )
    failed_cross = (
        recently_crossed and (
            macd_curr < sig_curr or hist_curr < 0
            or (not pd.isna(cross_day_low) and price < cross_day_low)
            or (not pd.isna(sma50) and price < sma50)
        )
    )

    spy_close = get_series(data_map.get("SPY", pd.DataFrame()), "Close")
    qqq_close = get_series(data_map.get("QQQ", pd.DataFrame()), "Close")
    spy_ret20 = pct_return(spy_close, 20) if not spy_close.empty else np.nan
    qqq_ret20 = pct_return(qqq_close, 20) if not qqq_close.empty else np.nan
    ret20 = pct_return(close, 20)
    ret63 = pct_return(close, 63)
    rs_spy_20 = ret20 - spy_ret20 if not pd.isna(ret20) and not pd.isna(spy_ret20) else np.nan
    rs_qqq_20 = ret20 - qqq_ret20 if not pd.isna(ret20) and not pd.isna(qqq_ret20) else np.nan

    sector = get_sector(ticker)
    sec_etf = sector_etf_for(ticker)
    sec_close = get_series(data_map.get(sec_etf), "Close")
    sec_ret20 = pct_return(sec_close, 20) if not sec_close.empty else np.nan
    stock_vs_sector_20 = ret20 - sec_ret20 if not pd.isna(ret20) and not pd.isna(sec_ret20) else np.nan
    sec_state, sec_rs_spy = sector_state(sec_etf, data_map, spy_ret20)

    liquid_ok, liquid_reason = estimate_options_tradability(ticker, price, avg_dollar_vol20, avg_vol20)
    market_ok = (market_regime != "HOSTILE") or cfg["allow_hostile_market"] or ignore_market_gate
    rs_ok = pd.isna(rs_spy_20) or rs_spy_20 >= cfg["min_rs_spy_20"]
    stock_vs_sector_ok = pd.isna(stock_vs_sector_20) or stock_vs_sector_20 >= cfg["min_stock_vs_sector_20"]
    sector_ok = sec_state in ["Strong", "Neutral", "Unknown"]
    volume_ok = pd.isna(vol_ratio) or vol_ratio >= cfg["min_volume_ratio"]

    score = 0
    score += 20 if trend_ok else 0
    score += 25 if confirmed else 20 if about_to_cross else 10 if crossed_today or recently_crossed else 0
    score += 15 if (not pd.isna(rs_spy_20) and rs_spy_20 > 0) else 5 if rs_ok else 0
    score += 10 if (not pd.isna(stock_vs_sector_20) and stock_vs_sector_20 > 0) else 5 if stock_vs_sector_ok else 0
    score += 10 if sec_state == "Strong" else 6 if sec_state == "Neutral" else 2 if sec_state == "Unknown" else 0
    score += 10 if liquid_ok else 0
    score += 5 if volume_ok else 0
    score += 5 if market_ok else 0
    if not pd.isna(pct_from_52w_high) and pct_from_52w_high >= -0.25:
        score += 5
    if failed_cross:
        score = 0
    score = int(max(0, min(100, round(score))))

    strict_good = all([market_ok, trend_ok, rs_ok, stock_vs_sector_ok, sector_ok, liquid_ok, volume_ok])
    if failed_cross:
        grade, bucket, action, macd_status = "F Failed Cross", "Failed / Exit Warnings", "EXIT / AVOID — bullish setup failed. No averaging down.", f"Failed cross ({bars_since_cross} bars ago)"
    elif confirmed and strict_good:
        grade, bucket, action, macd_status = "A Confirmed", "Confirmed Trade Candidates", "TRADE CANDIDATE — only with small size and hard invalidation.", f"Confirmed cross ({bars_since_cross} bars ago)"
    elif confirmed:
        grade, bucket, action, macd_status = "B Confirmed Caution", "Watchlist / Caution", "WATCH ONLY — confirmation exists but quality filters are not clean.", f"Confirmed but imperfect ({bars_since_cross} bars ago)"
    elif about_to_cross and strict_good and score >= 75:
        grade, bucket, action, macd_status = "A+ Pre-Cross", "A+ Pre-Cross Setups", "WATCH — buy only after trigger + actual MACD cross + market holds.", "About to cross"
    elif about_to_cross and trend_ok and liquid_ok:
        grade, bucket, action, macd_status = "A Watch", "Watchlist / Caution", "WATCH — close, but needs missing filters to clean up.", "About to cross, imperfect"
    elif about_to_cross:
        grade, bucket, action, macd_status = "B Early", "Watchlist / Caution", "EARLY ONLY — not tradable yet.", "Early curl"
    elif crossed_today:
        grade, bucket, action, macd_status = "B Fresh Cross", "Watchlist / Caution", "DO NOT CHASE — wait for next-day confirmation above signal-day high.", "Crossed today"
    elif recently_crossed:
        grade, bucket, action, macd_status = "C Unconfirmed Cross", "Rejected / Full Scan", "NO TRADE — crossed but not confirmed.", f"Unconfirmed cross ({bars_since_cross} bars ago)"
    elif macd_curr > sig_curr:
        grade, bucket, action, macd_status = "C Already Crossed", "Rejected / Full Scan", "NO NEW ENTRY — move already underway.", "Already crossed"
    else:
        grade, bucket, action, macd_status = "C No Setup", "Rejected / Full Scan", "NO TRADE.", "No clean bullish setup"

    flags = []
    if failed_cross: flags.append("FAILED CROSS")
    if not market_ok: flags.append("Market hostile")
    if sec_state == "Weak": flags.append("Weak sector")
    if not above50: flags.append("Below 50SMA")
    if not above200: flags.append("Below 200EMA")
    if not rs_ok: flags.append("Weak RS vs SPY")
    if not stock_vs_sector_ok: flags.append("Lagging sector")
    if not volume_ok: flags.append("Weak volume")
    if not liquid_ok: flags.append(liquid_reason)
    if not pd.isna(pct_from_52w_high) and pct_from_52w_high < -0.30: flags.append("Far from 52W high")

    if grade.startswith("A+"):
        trigger = f"Break > prior high {prior_high:.2f} + MACD cross"
        invalidation = f"Close < prior low {prior_low:.2f} or lose 50SMA"
    elif grade.startswith("A Confirmed") or grade.startswith("B Confirmed"):
        trigger = f"Hold > cross high {cross_day_high:.2f}"
        invalidation = f"Close < cross low {cross_day_low:.2f} or MACD fails"
    elif crossed_today:
        trigger = f"Wait: next close > signal-day high {float(high.iloc[-1]):.2f}"
        invalidation = f"Close < signal-day low {float(low.iloc[-1]):.2f}"
    else:
        trigger = "No entry trigger"
        invalidation = "No trade"

    return {
        "Bucket": bucket, "Grade": grade, "Ticker": ticker, "Sector": sector, "Sector ETF": sec_etf,
        "Price": round(price, 2), "MACD Status": macd_status, "Score": score,
        "Cross Proximity %": round(proximity, 1), "Hist Trend": hist_label,
        "Bars Since Cross": bars_since_cross if bars_since_cross is not None else np.nan,
        "RS vs SPY 20D": round(rs_spy_20 * 100, 2) if not pd.isna(rs_spy_20) else np.nan,
        "RS vs QQQ 20D": round(rs_qqq_20 * 100, 2) if not pd.isna(rs_qqq_20) else np.nan,
        "Stock vs Sector 20D": round(stock_vs_sector_20 * 100, 2) if not pd.isna(stock_vs_sector_20) else np.nan,
        "Sector State": sec_state,
        "Sector RS vs SPY 20D": round(sec_rs_spy * 100, 2) if not pd.isna(sec_rs_spy) else np.nan,
        "Above 20SMA": "Yes" if above20 else "No", "Above 50SMA": "Yes" if above50 else "No",
        "Above 200EMA": "Yes" if above200 else "No", "Vol Ratio": round(vol_ratio, 2) if not pd.isna(vol_ratio) else np.nan,
        "20D Return %": round(ret20 * 100, 2) if not pd.isna(ret20) else np.nan,
        "63D Return %": round(ret63 * 100, 2) if not pd.isna(ret63) else np.nan,
        "% From 52W High": round(pct_from_52w_high * 100, 2) if not pd.isna(pct_from_52w_high) else np.nan,
        "Options Proxy": liquid_reason, "Trigger": trigger, "Invalidation": invalidation,
        "Risk Flags": ", ".join(flags[:5]) if flags else "Clean", "Action": action,
        "Chart": f"https://www.tradingview.com/symbols/{tv_symbol(ticker)}/",
    }


def make_download(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def render_table(df: pd.DataFrame, height=420):
    if df.empty:
        st.info("Nothing in this section under current filters.")
        return
    columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    st.dataframe(
        df[columns], use_container_width=True, hide_index=True, height=height,
        column_config={
            "Chart": st.column_config.LinkColumn("Chart", display_text="Open"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
        },
    )

# ============================================================
# UI
# ============================================================

st.title("🦊 SilverFoxFlow Mach 8.0")
st.caption("Recovery MACD scanner: pre-cross quality, confirmed entries, failed-cross warnings, sector rotation, and market-regime control.")

with st.sidebar:
    st.header("Mach 8.0 Controls")
    mode = st.radio("Trading Mode", list(MODE_CONFIG.keys()), index=0)
    st.caption(MODE_CONFIG[mode]["label"])
    universe_choice = st.selectbox("Tradable Universe", ["Top 100", "Top 150", "Top 200", "Full S&P 500", "Liquid Core Only"], index=1)
    include_etfs = st.checkbox("Include ETFs as tradable symbols", value=True)
    ignore_market_gate = st.checkbox("Research only: ignore hostile market gate", value=False)
    period = st.selectbox("History Period", ["6mo", "1y", "2y"], index=1)
    custom_text = st.text_area("Add custom tickers", placeholder="Example: TSM, ARM, MSTR", height=80)
    st.caption("Earnings dates are not auto-blocked in this build. Check earnings manually before entry.")

st.warning("Mach 8.0 is a decision-support scanner, not a buy/sell guarantee. In Recovery Mode, pre-cross names are watchlist alerts, not automatic entries.")

custom_tickers = []
if custom_text.strip():
    custom_tickers = [x.strip().upper() for x in custom_text.replace("\n", ",").split(",") if x.strip()]

if universe_choice == "Liquid Core Only":
    candidate_base = sorted(set(LIQUID_OPTIONS_BOOST) | set(custom_tickers))
else:
    candidate_base = sorted(set(SP500_TICKERS) | set(EXTRA_LIQUID_OPTIONS) | set(custom_tickers))

all_download_tickers = sorted(set(candidate_base) | set(CORE_ETFS))
with st.spinner(f"Downloading history for {len(all_download_tickers)} symbols and ranking the tradable universe..."):
    history = download_history(all_download_tickers, period=period, interval="1d")

regime, call_trading, best_action, market_df, market_points = analyze_market_regime(history)
spy_close = get_series(history.get("SPY", pd.DataFrame()), "Close")
spy_ret20 = pct_return(spy_close, 20) if not spy_close.empty else np.nan
ranked_universe = rank_universe(history, candidate_base, include_etfs=include_etfs)

if universe_choice.startswith("Top"):
    n = int(universe_choice.split()[1])
    scan_tickers = ranked_universe.head(n)["Ticker"].tolist()
elif universe_choice == "Liquid Core Only":
    scan_tickers = ranked_universe["Ticker"].tolist()
else:
    scan_tickers = [t for t in candidate_base if include_etfs or t not in CORE_ETFS]
scan_tickers = [t for t in scan_tickers if t in history]

rows = []
with st.spinner(f"Running Mach 8.0 logic on {len(scan_tickers)} ranked symbols..."):
    for ticker in scan_tickers:
        row = analyze_ticker(ticker, history, regime, mode, ignore_market_gate=ignore_market_gate)
        if row:
            rows.append(row)
scan_df = pd.DataFrame(rows)

if not scan_df.empty:
    bucket_order = {"A+ Pre-Cross Setups": 0, "Confirmed Trade Candidates": 1, "Failed / Exit Warnings": 2, "Watchlist / Caution": 3, "Rejected / Full Scan": 4}
    scan_df["_bucket_order"] = scan_df["Bucket"].map(bucket_order).fillna(9)
    scan_df = scan_df.sort_values(["_bucket_order", "Score", "Cross Proximity %"], ascending=[True, False, False]).drop(columns=["_bucket_order"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Market Regime", regime)
col2.metric("Call Trading", call_trading)
col3.metric("Mode", mode)
col4.metric("Scanned", len(scan_tickers))
col5.metric("Market Score", f"{market_points} / 8")
st.caption(f"Best action: **{best_action}**")

if regime == "HOSTILE" and mode == "Recovery" and not ignore_market_gate:
    st.error("Recovery Mode says call trading is OFF in a hostile market. Pre-cross alerts should be watched, not chased.")
elif regime == "NEUTRAL":
    st.warning("Neutral market: require confirmation and smaller size.")
else:
    st.success("Market gate is not blocking A/A+ candidates under current mode.")

if scan_df.empty:
    st.info("No scan rows were produced. Check yfinance connection, symbols, or reduce filters.")
    st.stop()

pre_df = scan_df[scan_df["Bucket"] == "A+ Pre-Cross Setups"].copy()
confirmed_df = scan_df[scan_df["Bucket"] == "Confirmed Trade Candidates"].copy()
failed_df = scan_df[scan_df["Bucket"] == "Failed / Exit Warnings"].copy()
watch_df = scan_df[scan_df["Bucket"] == "Watchlist / Caution"].copy()
rejected_df = scan_df[scan_df["Bucket"] == "Rejected / Full Scan"].copy()

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("A+ Pre-Cross", len(pre_df))
s2.metric("Confirmed", len(confirmed_df))
s3.metric("Failed / Exit", len(failed_df))
s4.metric("Watch/Caution", len(watch_df))
s5.metric("Rejected", len(rejected_df))

st.markdown("---")
st.subheader("Daily Command Center")
left, right = st.columns([2, 1])
with left:
    st.markdown("#### Best Names First")
    best_df = pd.concat([pre_df, confirmed_df], ignore_index=True).sort_values("Score", ascending=False)
    render_table(best_df.head(12), height=360)
with right:
    st.markdown("#### Rules for Today")
    st.write(f"**Market:** {regime}")
    st.write(f"**Call trading:** {call_trading}")
    st.write("**Pre-cross = alert only.**")
    st.write("**Entry requires:** trigger break + MACD cross/confirmation + market holding.")
    st.write("**Exit requires no debate:** failed cross, loss of signal-day low, or MACD back under signal.")
    st.write("**Options:** avoid weeklies; prefer liquid 30–60 DTE; keep loss cap fixed.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["A+ Pre-Cross", "Confirmed", "Failed / Exit", "Watchlist", "Sector / Market", "Full Scan", "Exports / Journal"])

with tab1:
    st.header("A+ Pre-Cross Setups")
    st.caption("Best early alerts. These are NOT automatic buys. Wait for trigger + actual cross/confirmation.")
    render_table(pre_df, height=520)

with tab2:
    st.header("Confirmed Trade Candidates")
    st.caption("MACD crossed and price confirmed. Still use small size and invalidation rules.")
    render_table(confirmed_df, height=520)

with tab3:
    st.header("Failed Cross / Exit Warnings")
    st.caption("This section is designed to prevent ALB-style hold-and-hope traps.")
    render_table(failed_df, height=520)

with tab4:
    st.header("Watchlist / Caution")
    st.caption("Interesting but imperfect. In Recovery Mode, these should not be traded until missing filters improve.")
    render_table(watch_df.sort_values("Score", ascending=False), height=620)

with tab5:
    st.header("Sector Leadership + Market Regime")
    st.subheader("Market ETFs")
    st.dataframe(market_df, use_container_width=True, hide_index=True)
    sec_rows = []
    for sec, etf in SECTOR_TO_ETF.items():
        if sec in ["ETF", "Other"]:
            continue
        state, rs = sector_state(etf, history, spy_ret20)
        sec_rows.append({"Sector": sec, "ETF": etf, "State": state, "RS vs SPY 20D %": round(rs * 100, 2) if not pd.isna(rs) else np.nan})
    sec_df = pd.DataFrame(sec_rows).sort_values("RS vs SPY 20D %", ascending=False, na_position="last")
    st.subheader("Sector Rotation")
    st.dataframe(sec_df, use_container_width=True, hide_index=True)
    st.subheader("Tradability Ranking")
    st.caption("Top names selected by dollar volume + liquid-options boost. This keeps the full S&P 500 list but scans the best 100–200 names first.")
    rank_display = ranked_universe.head(250).copy()
    if not rank_display.empty:
        rank_display["Avg $Vol 20D"] = rank_display["Avg $Vol 20D"].round(0).astype("int64")
    st.dataframe(rank_display, use_container_width=True, hide_index=True, height=520)

with tab6:
    st.header("Full Scan")
    st.caption("Everything scanned, sorted by bucket and score. Use this for research, not impulse entries.")
    render_table(scan_df, height=720)

with tab7:
    st.header("Exports / Trade Journal")
    st.download_button("Download Full Mach 8.0 Scan CSV", make_download(scan_df), "silverfoxflow_mach8_full_scan.csv", "text/csv")
    st.download_button("Download A+ Pre-Cross CSV", make_download(pre_df), "silverfoxflow_mach8_a_plus_precross.csv", "text/csv")
    st.download_button("Download Confirmed Candidates CSV", make_download(confirmed_df), "silverfoxflow_mach8_confirmed.csv", "text/csv")
    st.download_button("Download Failed Cross Warnings CSV", make_download(failed_df), "silverfoxflow_mach8_failed_cross.csv", "text/csv")
    journal_cols = ["Date", "Ticker", "Grade At Entry", "Entry Type", "Option Contract", "Entry Price", "Stop Rule", "Target", "Exit Price", "Result %", "Reason Entered", "Reason Exited", "Screenshot Link", "Notes"]
    journal_template = pd.DataFrame(columns=journal_cols)
    st.download_button("Download Trade Journal Template CSV", make_download(journal_template), "silverfoxflow_trade_journal_template.csv", "text/csv")
    st.markdown("#### Non-negotiable Recovery Rules")
    st.write("1. A+ Pre-Cross is only a watchlist alert.")
    st.write("2. No entry if market regime is HOSTILE in Recovery Mode.")
    st.write("3. No entry before earnings unless that is the specific plan.")
    st.write("4. Exit if MACD falls back under signal after entry or price closes below signal-day low.")
    st.write("5. Options loss cap must be set before entry, not after pain starts.")

st.markdown("---")
st.caption("SilverFoxFlow Mach 8.0 · MACD(12,26,9) · Recovery Mode default · built to reject bad trades first.")
