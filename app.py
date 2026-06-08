# SilverFoxFlow Mach 8.1.1 — Balanced Lifecycle MACD Scanner
# Run: streamlit run app.py
# Purpose: fewer trades, stronger setups, faster failed-cross warnings.

import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SilverFoxFlow Mach 8.1.1", page_icon="🦊", layout="wide")


# ============================================================
# VISUAL THEME — dark nature / earth tones
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --sf-bg: #111812;
        --sf-bg-2: #172118;
        --sf-panel: rgba(31, 45, 32, 0.92);
        --sf-panel-2: rgba(39, 55, 39, 0.88);
        --sf-border: rgba(196, 143, 84, 0.28);
        --sf-text: #f3ead7;
        --sf-muted: #b9c0a8;
        --sf-sage: #93a47d;
        --sf-moss: #60704f;
        --sf-copper: #c88f55;
        --sf-amber: #e0b06e;
        --sf-danger: #cf7d65;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 5%, rgba(120, 91, 55, 0.24), transparent 32%),
            radial-gradient(circle at 85% 12%, rgba(77, 103, 75, 0.25), transparent 30%),
            linear-gradient(135deg, #0f1711 0%, #172119 45%, #211a13 100%);
        color: var(--sf-text);
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141e15 0%, #201a12 100%);
        border-right: 1px solid var(--sf-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--sf-text);
    }

    h1, h2, h3, h4 {
        color: var(--sf-text) !important;
        letter-spacing: 0.01em;
    }

    p, label, span, div {
        color: inherit;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--sf-muted) !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(36, 53, 37, 0.95), rgba(31, 39, 27, 0.92));
        border: 1px solid var(--sf-border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--sf-muted) !important;
        font-size: 0.82rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--sf-amber) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid var(--sf-border);
        background: rgba(36, 48, 33, 0.72);
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(224, 176, 110, 0.45);
        background: linear-gradient(135deg, #765335, #42563a);
        color: #fff7e8;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--sf-amber);
        filter: brightness(1.08);
    }

    div[data-testid="stTabs"] button {
        color: var(--sf-muted) !important;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: rgba(200, 143, 85, 0.18);
        color: var(--sf-amber) !important;
        border: 1px solid rgba(200, 143, 85, 0.35);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--sf-border);
        border-radius: 18px;
        overflow: hidden;
        background: rgba(20, 30, 21, 0.78);
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }

    a {
        color: #e0b06e !important;
        font-weight: 750;
        text-decoration: none;
    }

    a:hover {
        color: #f0c98e !important;
        text-decoration: underline;
    }

    hr {
        border-color: rgba(196, 143, 84, 0.22);
    }

    .sf-hero {
        background: linear-gradient(135deg, rgba(45, 64, 43, 0.94), rgba(44, 32, 21, 0.92));
        border: 1px solid var(--sf-border);
        border-radius: 24px;
        padding: 24px 26px;
        box-shadow: 0 18px 44px rgba(0,0,0,0.28);
        margin-bottom: 18px;
    }

    .sf-hero-title {
        font-size: 2.05rem;
        font-weight: 900;
        color: var(--sf-text);
        margin-bottom: 4px;
    }

    .sf-hero-sub {
        color: var(--sf-muted);
        font-size: 1.02rem;
        line-height: 1.45;
    }

    .sf-chip {
        display: inline-block;
        padding: 6px 10px;
        border: 1px solid rgba(224, 176, 110, 0.32);
        border-radius: 999px;
        color: #f3ead7;
        background: rgba(147, 164, 125, 0.12);
        margin-right: 7px;
        margin-top: 10px;
        font-size: 0.83rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        "label": "Strict.",
        "min_volume_ratio": 0.80,
        "min_rs_spy_20": 0.00,
        "min_stock_vs_sector_20": 0.00,
        "allow_hostile_market": False,
        "fresh_cross_bars": 2,
        "late_cross_bars": 10,
        "max_extension_pct": 3.5,
    },
    "Balanced": {
        "label": "Daily default.",
        "min_volume_ratio": 0.70,
        "min_rs_spy_20": -0.01,
        "min_stock_vs_sector_20": -0.01,
        "allow_hostile_market": False,
        "fresh_cross_bars": 3,
        "late_cross_bars": 12,
        "max_extension_pct": 5.0,
    },
    "Aggressive": {
        "label": "More alerts.",
        "min_volume_ratio": 0.60,
        "min_rs_spy_20": -0.03,
        "min_stock_vs_sector_20": -0.03,
        "allow_hostile_market": True,
        "fresh_cross_bars": 4,
        "late_cross_bars": 15,
        "max_extension_pct": 7.0,
    },
}

PRE_CROSS_CONFIG = {
    "Strict": {"min_proximity": 70, "hist_streak": 3},
    "Balanced": {"min_proximity": 55, "hist_streak": 2},
    "Early": {"min_proximity": 40, "hist_streak": 1},
}

CLEAN_COLUMNS = [
    "Lifecycle", "Grade", "Ticker", "Sector", "Price", "MACD Status", "Age",
    "Score", "Cross Proximity %", "Hist Trend", "Risk Flags", "Trigger", "Invalidation", "Action",
]

FULL_COLUMNS = [
    "Lifecycle", "Grade", "Ticker", "Sector", "Sector ETF", "Price", "MACD Status", "Age", "Data",
    "Score", "Cross Proximity %", "Hist Trend", "Bars Since Cross", "Extension %",
    "RS vs SPY 20D", "RS vs QQQ 20D", "Stock vs Sector 20D",
    "Sector State", "Sector RS vs SPY 20D", "Above 20SMA", "Above 50SMA", "Above 200EMA",
    "Vol Ratio", "20D Return %", "63D Return %", "% From 52W High", "Options Proxy",
    "Why", "Risk Flags", "Trigger", "Invalidation", "Action",
]

# ============================================================
# HELPERS
# ============================================================

def yf_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").upper().strip()


def tv_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").upper().strip()


def tv_chart_url(ticker: str) -> str:
    """TradingView chart URL. The ticker text is extracted by Streamlit LinkColumn display_text regex."""
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol(ticker)}"


@st.cache_data(show_spinner=False, ttl=60 * 5)
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



@st.cache_data(show_spinner=False, ttl=60 * 5)
def download_intraday_latest(tickers, period="5d", interval="15m", batch_size=45):
    clean, seen = [], set()
    for t in tickers:
        t = str(t).upper().strip()
        if t and t not in seen:
            clean.append(t)
            seen.add(t)

    symbol_to_ticker = {yf_symbol(t): t for t in clean}
    latest = {}

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
            items = [(sym, df[sym]) for sym in batch_symbols if sym in top]
        else:
            items = [(batch_symbols[0], df)]

        for sym, sub in items:
            sub = sub.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            last_ts = sub.index[-1]
            last_day = pd.Timestamp(last_ts).date()
            day_df = sub[[c for c in ["High", "Low", "Close", "Volume"] if c in sub.columns]]
            day_df = day_df[[pd.Timestamp(x).date() == last_day for x in day_df.index]]
            if day_df.empty:
                continue
            t = symbol_to_ticker.get(sym, sym)
            latest[t] = {
                "Timestamp": pd.Timestamp(last_ts),
                "Close": float(day_df["Close"].iloc[-1]),
                "High": float(day_df["High"].max()) if "High" in day_df.columns else float(day_df["Close"].iloc[-1]),
                "Low": float(day_df["Low"].min()) if "Low" in day_df.columns else float(day_df["Close"].iloc[-1]),
                "Volume": float(day_df["Volume"].sum()) if "Volume" in day_df.columns else np.nan,
            }
    return latest


def overlay_intraday_on_daily(history: dict, intraday_latest: dict) -> dict:
    if not intraday_latest:
        return history
    out = dict(history)
    for ticker, live in intraday_latest.items():
        df = out.get(ticker)
        if df is None or df.empty:
            continue
        df = df.copy()
        live_ts = pd.Timestamp(live["Timestamp"])
        live_date = live_ts.date()
        last_idx = df.index[-1]
        last_date = pd.Timestamp(last_idx).date()
        if live_date < last_date:
            continue
        if live_date == last_date:
            idx = last_idx
        else:
            idx = pd.Timestamp(live_date)
            df.loc[idx] = df.iloc[-1]
        for col in ["Close", "High", "Low", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        old_high = pd.to_numeric(df.loc[[idx], "High"], errors="coerce").iloc[0]
        old_low = pd.to_numeric(df.loc[[idx], "Low"], errors="coerce").iloc[0]
        old_vol = pd.to_numeric(df.loc[[idx], "Volume"], errors="coerce").iloc[0]
        df.loc[idx, "Close"] = live.get("Close", df.loc[idx, "Close"])
        df.loc[idx, "High"] = max(float(old_high) if not pd.isna(old_high) else live["High"], live["High"])
        df.loc[idx, "Low"] = min(float(old_low) if not pd.isna(old_low) else live["Low"], live["Low"])
        df.loc[idx, "Volume"] = max(float(old_vol) if not pd.isna(old_vol) else 0.0, live.get("Volume", 0.0))
        out[ticker] = df.sort_index()
    return out


def _safe_naive_timestamp(value):
    """Normalize mixed yfinance timestamps so max() never compares tz-aware vs tz-naive."""
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts
    except Exception:
        return None


def latest_data_label(history: dict, intraday_latest: dict | None = None) -> str:
    stamps = []
    for df in history.values():
        if df is not None and not df.empty:
            ts = _safe_naive_timestamp(df.index[-1])
            if ts is not None:
                stamps.append(ts)
    for live in (intraday_latest or {}).values():
        if live and "Timestamp" in live:
            ts = _safe_naive_timestamp(live["Timestamp"])
            if ts is not None:
                stamps.append(ts)
    if not stamps:
        return "No data"
    ts = max(stamps)
    return ts.strftime("%Y-%m-%d %H:%M")

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


def analyze_ticker(
    ticker: str,
    data_map: dict,
    market_regime: str,
    mode: str,
    ignore_market_gate=False,
    pre_sensitivity="Balanced",
    provisional_live=False,
):
    cfg = MODE_CONFIG[mode]
    pcfg = PRE_CROSS_CONFIG.get(pre_sensitivity, PRE_CROSS_CONFIG["Balanced"])
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
    cross_info = last_bull_cross_info(macd, sig, high, low, lookback=100)
    bars_since_cross = cross_info["bars_since"] if cross_info else None
    cross_day_high = cross_info["high"] if cross_info else np.nan
    cross_day_low = cross_info["low"] if cross_info else np.nan
    age = "—" if bars_since_cross is None else f"{bars_since_cross} bars"

    crossed_today = macd_prev <= sig_prev and macd_curr > sig_curr
    active_cross = cross_info is not None and macd_curr > sig_curr
    fresh_confirmed_age = active_cross and bars_since_cross is not None and 1 <= bars_since_cross <= cfg["fresh_cross_bars"]
    late_confirmed_age = active_cross and bars_since_cross is not None and cfg["fresh_cross_bars"] < bars_since_cross <= cfg["late_cross_bars"]

    extension_pct = np.nan
    if cross_info and not pd.isna(cross_day_high) and cross_day_high > 0:
        extension_pct = (price / cross_day_high - 1.0) * 100
    extended = active_cross and not pd.isna(extension_pct) and extension_pct > cfg["max_extension_pct"]

    about_to_cross = (
        macd_curr < sig_curr
        and hist_curr < 0
        and hist_count >= pcfg["hist_streak"]
        and macd_curr > macd_prev
        and proximity >= pcfg["min_proximity"]
    )
    early_curl = (
        macd_curr < sig_curr
        and hist_curr < 0
        and hist_count >= 1
        and macd_curr > macd_prev
        and proximity >= max(25, pcfg["min_proximity"] - 20)
    )
    provisional_cross = crossed_today and macd_curr > sig_curr
    confirmed = (
        fresh_confirmed_age
        and price > cross_day_high
        and price > prev_close
        and (pd.isna(vol_ratio) or vol_ratio >= cfg["min_volume_ratio"])
    )

    # Hard failure only when the cross actually rolls over or price loses the signal low.
    # Being below the 50SMA is a risk flag, not an automatic failed-cross.
    recent_window = cross_info is not None and bars_since_cross is not None and bars_since_cross <= cfg["late_cross_bars"]
    hard_failed = recent_window and (
        (macd_curr < sig_curr and hist_curr < 0 and hist_count <= 0)
        or (not pd.isna(cross_day_low) and price < cross_day_low)
    )
    recovering = recent_window and not active_cross and not hard_failed and hist_count >= 2 and proximity >= 45

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
    score += 20 if trend_ok else 8 if above200 else 0
    score += 25 if confirmed else 22 if about_to_cross else 18 if provisional_cross else 12 if active_cross else 8 if early_curl else 0
    score += 15 if (not pd.isna(rs_spy_20) and rs_spy_20 > 0) else 5 if rs_ok else 0
    score += 10 if (not pd.isna(stock_vs_sector_20) and stock_vs_sector_20 > 0) else 5 if stock_vs_sector_ok else 0
    score += 10 if sec_state == "Strong" else 6 if sec_state == "Neutral" else 2 if sec_state == "Unknown" else 0
    score += 10 if liquid_ok else 0
    score += 5 if volume_ok else 0
    score += 5 if market_ok else 0
    if not pd.isna(pct_from_52w_high) and pct_from_52w_high >= -0.25:
        score += 5
    if hard_failed:
        score = min(score, 18)
    if not above200:
        score = min(score, 25)
    if extended:
        score = min(score, 72)
    score = int(max(0, min(100, round(score))))

    strict_good = all([market_ok, trend_ok, rs_ok, stock_vs_sector_ok, sector_ok, liquid_ok, volume_ok])

    if hard_failed:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Failed", "F Failed Cross", "Failed / Exit Warnings",
            "EXIT / AVOID", f"Failed cross ({bars_since_cross} bars ago)", "Cross rolled over or lost signal low."
        )
    elif not above200:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Under 200", "X Below 200EMA", "No Trade / Below 200EMA",
            "NO CALL TRADE", "Below 200EMA", "Trend gate failed."
        )
    elif provisional_cross:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Provisional", "P Provisional Cross", "Provisional / Near Close",
            "WATCH NEAR CLOSE", "Provisional cross", "Live/daily candle is not final."
        )
    elif confirmed and strict_good and not extended:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Fresh", "A Fresh Confirmed", "Confirmed Trade Candidates",
            "TRADE CANDIDATE", f"Fresh confirmed ({bars_since_cross} bars)", "Fresh cross with quality filters."
        )
    elif confirmed and not extended:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Fresh Caution", "B Confirmed Caution", "Watchlist / Caution",
            "WATCH ONLY", f"Confirmed but imperfect ({bars_since_cross} bars)", "Confirmed, but filters are not clean."
        )
    elif late_confirmed_age or (active_cross and extended):
        late_label = "Extended" if extended else "Late"
        lifecycle, grade, bucket, action, macd_status, why = (
            late_label, "B Late Confirmed", "Late / Learning",
            "NO CHASE", f"Late confirmed ({bars_since_cross} bars)", "Was better earlier; wait for pullback/retest."
        )
    elif about_to_cross and strict_good and score >= 75:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Pre-Cross", "A+ Pre-Cross", "A+ Pre-Cross Setups",
            "WATCH TRIGGER", "About to cross", "Strong curl before the cross."
        )
    elif about_to_cross and trend_ok and liquid_ok:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Pre-Cross", "A Watch", "Watchlist / Caution",
            "WATCH", "About to cross, imperfect", "Close, but needs filters to clean up."
        )
    elif early_curl:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Early Curl", "B Early Curl", "Watchlist / Caution",
            "EARLY ONLY", "Early curl", "Developing, not ready."
        )
    elif recovering:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Recovering", "R Recovering", "Watchlist / Caution",
            "WATCH", "Recovering from failed cross", "Improving, but no clean cross yet."
        )
    elif active_cross:
        lifecycle, grade, bucket, action, macd_status, why = (
            "Old Cross", "C Already Crossed", "Rejected / Full Scan",
            "NO NEW ENTRY", "Already crossed", "Move is not fresh."
        )
    else:
        lifecycle, grade, bucket, action, macd_status, why = (
            "No Setup", "C No Setup", "Rejected / Full Scan",
            "NO TRADE", "No clean bullish setup", "No setup."
        )

    flags = []
    if hard_failed: flags.append("FAILED CROSS")
    if not market_ok: flags.append("Market hostile")
    if sec_state == "Weak": flags.append("Weak sector")
    if not above50: flags.append("Below 50SMA")
    if not above200: flags.append("Below 200EMA")
    if not rs_ok: flags.append("Weak RS vs SPY")
    if not stock_vs_sector_ok: flags.append("Lagging sector")
    if not volume_ok: flags.append("Weak volume")
    if not liquid_ok: flags.append(liquid_reason)
    if extended: flags.append("Extended")
    if not pd.isna(pct_from_52w_high) and pct_from_52w_high < -0.30: flags.append("Far from 52W high")

    if lifecycle == "Pre-Cross":
        trigger = f"> prior high {prior_high:.2f} + cross"
        invalidation = f"< prior low {prior_low:.2f}"
    elif lifecycle in ["Fresh", "Fresh Caution"]:
        trigger = f"Hold > cross high {cross_day_high:.2f}"
        invalidation = f"< cross low {cross_day_low:.2f} or MACD fail"
    elif lifecycle == "Provisional":
        trigger = f"Near close > {float(high.iloc[-1]):.2f}"
        invalidation = f"< signal low {float(low.iloc[-1]):.2f}"
    elif lifecycle in ["Late", "Extended", "Old Cross"]:
        trigger = "Wait pullback/retest"
        invalidation = f"< cross low {cross_day_low:.2f}" if not pd.isna(cross_day_low) else "No trade"
    else:
        trigger = "No entry trigger"
        invalidation = "No trade"

    return {
        "Bucket": bucket, "Lifecycle": lifecycle, "Grade": grade, "Ticker": ticker, "Sector": sector, "Sector ETF": sec_etf,
        "Price": round(price, 2), "MACD Status": macd_status, "Age": age, "Data": "Live overlay" if provisional_live else "Daily",
        "Score": score, "Cross Proximity %": round(proximity, 1), "Hist Trend": hist_label,
        "Bars Since Cross": bars_since_cross if bars_since_cross is not None else np.nan,
        "Extension %": round(extension_pct, 2) if not pd.isna(extension_pct) else np.nan,
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
        "Options Proxy": liquid_reason, "Why": why, "Trigger": trigger, "Invalidation": invalidation,
        "Risk Flags": ", ".join(flags[:5]) if flags else "Clean", "Action": action,
        "Chart": tv_chart_url(ticker),
    }


def make_download(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def linked_display_df(df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """Return a display copy where ticker-like columns become TradingView links."""
    if columns is not None:
        out = df[[c for c in columns if c in df.columns]].copy()
    else:
        out = df.copy()
    for col in ["Ticker", "ETF", "Sector ETF"]:
        if col in out.columns:
            out[col] = out[col].astype(str).apply(tv_chart_url)
    return out


def linked_column_config(extra=None):
    cfg = {
        "Ticker": st.column_config.LinkColumn("Ticker", display_text=r"symbol=([^&]+)", width="small"),
        "ETF": st.column_config.LinkColumn("ETF", display_text=r"symbol=([^&]+)", width="small"),
        "Sector ETF": st.column_config.LinkColumn("Sector ETF", display_text=r"symbol=([^&]+)", width="small"),
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
    }
    if extra:
        cfg.update(extra)
    return cfg


def render_table(df: pd.DataFrame, height=420):
    if df.empty:
        st.info("Nothing in this section under current filters.")
        return
    detail = globals().get("table_detail", "Clean")
    base_cols = FULL_COLUMNS if detail == "Full" else CLEAN_COLUMNS
    columns = [c for c in base_cols if c in df.columns]
    display_df = linked_display_df(df, columns)
    st.dataframe(
        display_df, use_container_width=True, hide_index=True, height=height,
        column_config=linked_column_config(),
    )

# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="sf-hero sf-hero-compact">
        <div class="sf-hero-title">🦊 SilverFoxFlow Mach 8.1.1</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    mode = st.radio("Mode", list(MODE_CONFIG.keys()), index=1)
    universe_choice = st.selectbox("Universe", ["Top 100", "Top 150", "Top 200", "Full S&P 500", "Liquid Core Only"], index=1)
    table_detail = st.radio("Detail", ["Clean", "Full"], index=0, horizontal=True)
    include_etfs = st.checkbox("Include ETFs", value=True)
    ignore_market_gate = st.checkbox("Research: ignore market gate", value=False)
    period = st.selectbox("History", ["6mo", "1y", "2y"], index=1)
    data_mode = st.selectbox("Data", ["Daily + live overlay", "Daily only"], index=0)
    pre_sensitivity = st.selectbox("Pre-cross", ["Balanced", "Strict", "Early"], index=0)
    auto_refresh = st.selectbox("Auto refresh", ["Off", "5 min", "15 min"], index=0)
    custom_text = st.text_area("Custom tickers", placeholder="TSM, ARM, MSTR", height=70)
    st.caption("Earnings: check manually.")

if auto_refresh != "Off":
    ms = 300000 if auto_refresh == "5 min" else 900000
    st.markdown(f"<script>setTimeout(() => window.location.reload(), {ms});</script>", unsafe_allow_html=True)

custom_tickers = []
if custom_text.strip():
    custom_tickers = [x.strip().upper() for x in custom_text.replace("\n", ",").split(",") if x.strip()]

if universe_choice == "Liquid Core Only":
    candidate_base = sorted(set(LIQUID_OPTIONS_BOOST) | set(custom_tickers))
else:
    candidate_base = sorted(set(SP500_TICKERS) | set(EXTRA_LIQUID_OPTIONS) | set(custom_tickers))

all_download_tickers = sorted(set(candidate_base) | set(CORE_ETFS))
with st.spinner(f"Downloading history for {len(all_download_tickers)} symbols..."):
    history_raw = download_history(all_download_tickers, period=period, interval="1d")

intraday_latest = {}
use_live_overlay = data_mode == "Daily + live overlay"
if use_live_overlay:
    with st.spinner("Updating live overlay..."):
        intraday_latest = download_intraday_latest(all_download_tickers, period="5d", interval="15m")
    history = overlay_intraday_on_daily(history_raw, intraday_latest)
else:
    history = history_raw

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
        row = analyze_ticker(ticker, history, regime, mode, ignore_market_gate=ignore_market_gate, pre_sensitivity=pre_sensitivity, provisional_live=use_live_overlay)
        if row:
            rows.append(row)
scan_df = pd.DataFrame(rows)

if not scan_df.empty:
    bucket_order = {"A+ Pre-Cross Setups": 0, "Confirmed Trade Candidates": 1, "Provisional / Near Close": 2, "Late / Learning": 3, "Failed / Exit Warnings": 4, "No Trade / Below 200EMA": 5, "Watchlist / Caution": 6, "Rejected / Full Scan": 7}
    scan_df["_bucket_order"] = scan_df["Bucket"].map(bucket_order).fillna(9)
    scan_df = scan_df.sort_values(["_bucket_order", "Score", "Cross Proximity %"], ascending=[True, False, False]).drop(columns=["_bucket_order"])

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Market", regime)
col2.metric("Calls", call_trading)
col3.metric("Mode", mode)
col4.metric("Scanned", len(scan_tickers))
col5.metric("Score", f"{market_points} / 8")
col6.metric("Data", latest_data_label(history, intraday_latest))
if regime == "HOSTILE" and mode == "Recovery" and not ignore_market_gate:
    st.error("Call trading OFF.")
elif regime == "NEUTRAL":
    st.warning("Caution: confirm first.")
else:
    st.success("Market gate open.")

if scan_df.empty:
    st.info("No scan rows produced.")
    st.stop()

pre_df = scan_df[scan_df["Bucket"] == "A+ Pre-Cross Setups"].copy()
confirmed_df = scan_df[scan_df["Bucket"] == "Confirmed Trade Candidates"].copy()
provisional_df = scan_df[scan_df["Bucket"] == "Provisional / Near Close"].copy()
late_df = scan_df[scan_df["Bucket"] == "Late / Learning"].copy()
failed_df = scan_df[scan_df["Bucket"] == "Failed / Exit Warnings"].copy()
under200_df = scan_df[scan_df["Bucket"] == "No Trade / Below 200EMA"].copy()
watch_df = scan_df[scan_df["Bucket"] == "Watchlist / Caution"].copy()
rejected_df = scan_df[scan_df["Bucket"] == "Rejected / Full Scan"].copy()

s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
s1.metric("Pre-Cross", len(pre_df))
s2.metric("Fresh", len(confirmed_df))
s3.metric("Provisional", len(provisional_df))
s4.metric("Late", len(late_df))
s5.metric("Failed", len(failed_df))
s6.metric("Under 200", len(under200_df))
s7.metric("Watch", len(watch_df))

st.markdown("---")
st.subheader("Command Center")
left, right = st.columns([2, 1])
with left:
    st.markdown("#### Best Names")
    best_df = pd.concat([pre_df, confirmed_df, provisional_df], ignore_index=True)
    if not best_df.empty and "Above 200EMA" in best_df.columns:
        best_df = best_df[best_df["Above 200EMA"] == "Yes"]
    best_df = best_df.sort_values("Score", ascending=False)
    render_table(best_df.head(8), height=320)
with right:
    st.markdown("#### Rules")
    st.write(f"**Market:** {regime}")
    st.write(f"**Calls:** {call_trading}")
    st.write("**Entry:** trigger + confirmation")
    st.write("**Exit:** failed cross / invalidation")
    st.write("**Options:** liquid 30–60 DTE")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Pre-Cross", "Fresh", "Provisional", "Late", "Failed", "Under 200", "Watchlist", "Audit", "Market", "Exports"])

with tab1:
    st.header("Pre-Cross")
    render_table(pre_df, height=520)

with tab2:
    st.header("Fresh Confirmed")
    render_table(confirmed_df, height=520)

with tab3:
    st.header("Provisional")
    render_table(provisional_df, height=520)

with tab4:
    st.header("Late / Learning")
    render_table(late_df.sort_values("Score", ascending=False), height=560)

with tab5:
    st.header("Failed")
    render_table(failed_df, height=560)

with tab6:
    st.header("Under 200")
    render_table(under200_df.sort_values("Score", ascending=False), height=560)

with tab7:
    st.header("Watchlist")
    render_table(watch_df.sort_values("Score", ascending=False), height=620)

with tab8:
    st.header("Ticker Audit")
    audit_default = "CVX" if "CVX" in scan_df["Ticker"].values else str(scan_df["Ticker"].iloc[0])
    audit_ticker = st.text_input("Ticker", value=audit_default).upper().strip()
    if audit_ticker:
        if audit_ticker not in scan_df["Ticker"].values:
            audit_row = analyze_ticker(audit_ticker, history, regime, mode, ignore_market_gate=ignore_market_gate, pre_sensitivity=pre_sensitivity, provisional_live=use_live_overlay)
            audit_df = pd.DataFrame([audit_row]) if audit_row else pd.DataFrame()
        else:
            audit_df = scan_df[scan_df["Ticker"] == audit_ticker].copy()
        if audit_df.empty:
            st.info("No data.")
        else:
            r = audit_df.iloc[0]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Lifecycle", r.get("Lifecycle", "—"))
            a2.metric("Grade", r.get("Grade", "—"))
            a3.metric("Age", r.get("Age", "—"))
            a4.metric("Score", int(r.get("Score", 0)))
            st.write(f"**Why:** {r.get('Why', '—')}")
            st.write(f"**Trigger:** {r.get('Trigger', '—')}")
            st.write(f"**Invalidation:** {r.get('Invalidation', '—')}")
            render_table(audit_df, height=180)

with tab9:
    st.header("Market")
    st.subheader("ETFs")
    st.dataframe(linked_display_df(market_df), use_container_width=True, hide_index=True, column_config=linked_column_config())
    sec_rows = []
    for sec, etf in SECTOR_TO_ETF.items():
        if sec in ["ETF", "Other"]:
            continue
        state, rs = sector_state(etf, history, spy_ret20)
        sec_rows.append({"Sector": sec, "ETF": etf, "State": state, "RS vs SPY 20D %": round(rs * 100, 2) if not pd.isna(rs) else np.nan})
    sec_df = pd.DataFrame(sec_rows).sort_values("RS vs SPY 20D %", ascending=False, na_position="last")
    st.subheader("Sectors")
    st.dataframe(linked_display_df(sec_df), use_container_width=True, hide_index=True, column_config=linked_column_config())
    st.subheader("Universe Rank")
    rank_display = ranked_universe.head(250).copy()
    if not rank_display.empty:
        rank_display["Avg $Vol 20D"] = rank_display["Avg $Vol 20D"].round(0).astype("int64")
    st.dataframe(linked_display_df(rank_display), use_container_width=True, hide_index=True, height=520, column_config=linked_column_config())
    st.subheader("Full Scan")
    render_table(scan_df, height=720)

with tab10:
    st.header("Exports")
    st.download_button("Full Scan CSV", make_download(scan_df), "silverfoxflow_mach8_1_full_scan.csv", "text/csv")
    st.download_button("Pre-Cross CSV", make_download(pre_df), "silverfoxflow_mach8_1_precross.csv", "text/csv")
    st.download_button("Fresh CSV", make_download(confirmed_df), "silverfoxflow_mach8_1_fresh.csv", "text/csv")
    st.download_button("Provisional CSV", make_download(provisional_df), "silverfoxflow_mach8_1_provisional.csv", "text/csv")
    st.download_button("Late CSV", make_download(late_df), "silverfoxflow_mach8_1_late.csv", "text/csv")
    st.download_button("Failed CSV", make_download(failed_df), "silverfoxflow_mach8_1_failed.csv", "text/csv")
    journal_cols = ["Date", "Ticker", "Grade At Entry", "Entry Type", "Option Contract", "Entry Price", "Stop Rule", "Target", "Exit Price", "Result %", "Reason Entered", "Reason Exited", "Screenshot Link", "Notes"]
    journal_template = pd.DataFrame(columns=journal_cols)
    st.download_button("Journal Template CSV", make_download(journal_template), "silverfoxflow_trade_journal_template.csv", "text/csv")
    with st.expander("Rules", expanded=False):
        st.write("Pre-Cross = watch only.")
        st.write("Provisional = wait near close.")
        st.write("Fresh = possible trade.")
        st.write("Late = no chase.")
        st.write("Failed = exit/avoid.")
