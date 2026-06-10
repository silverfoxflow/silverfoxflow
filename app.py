# SilverFoxFlow Mach 8.5.1 — Clean Trade Filter
# Run: streamlit run app.py
# Purpose: simple category-first home, clean actions, and clear MACD history score.

import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SilverFoxFlow Mach 8.5.0", page_icon="🦊", layout="wide")


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
    "Decision", "Ticker", "Sector", "Price", "Lifecycle", "Score", "Historical Edge", "Edge Verdict",
    "Signals Tested", "10D Win %", "Avg 10D %", "Risk Tier", "Risk Flags", "Trigger", "Invalidation",
    "Next Check", "Action",
]

FULL_COLUMNS = [
    "Decision", "Lifecycle", "Grade", "Ticker", "Sector", "Sector ETF", "Price", "MACD Status", "Age", "Data",
    "Score", "Historical Edge", "Edge Verdict", "Signals Tested", "10D Win %", "Avg 10D %",
    "Target First %", "Stop First %", "Failure %", "Worst DD %", "Cross Proximity %", "Hist Trend",
    "Bars Since Cross", "Extension %", "RS vs SPY 20D", "RS vs QQQ 20D", "Stock vs Sector 20D",
    "Sector State", "Sector RS vs SPY 20D", "Above 20SMA", "Above 50SMA", "Above 200EMA",
    "Vol Ratio", "20D Return %", "63D Return %", "% From 52W High", "Options Proxy",
    "Why", "Decision Reason", "Risk Tier", "Risk Flags", "Trigger", "Invalidation",
    "Next Check", "Option Plan", "Position Size", "Action",
]

DECISION_COLUMNS = [
    "Decision", "Ticker", "Sector", "Price", "Lifecycle", "Score", "Historical Edge", "Edge Verdict",
    "Signals Tested", "10D Win %", "Avg 10D %", "Risk Tier", "Trigger", "Invalidation",
    "Decision Reason", "Next Check", "Option Plan", "Action",
]

EDGE_COLUMNS = [
    "Ticker", "Sector", "Historical Edge", "Edge Verdict", "Signals Tested", "10D Win %", "Avg 10D %",
    "Avg 5D %", "Target First %", "Stop First %", "Failure %", "Worst DD %", "Avg Max DD %",
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
        "Price": round(price, 2), "MACD Value": round(macd_curr, 4), "MACD Signal": round(sig_curr, 4),
        "MACD %": round((macd_curr / price) * 100.0, 3) if price else np.nan,
        "MACD Status": macd_status, "Age": age, "Data": "Live overlay" if provisional_live else "Daily",
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
        "Historical Edge": st.column_config.ProgressColumn("Historical Edge", min_value=0, max_value=100),
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
# HISTORICAL EDGE ENGINE — equity backtest proxy
# ============================================================

BACKTEST_STOP_PCT = 0.06
BACKTEST_TARGET_PCT = 0.08
BACKTEST_MAX_LOOKBACK_BARS = 1000


def _pct(value):
    if value is None or pd.isna(value):
        return np.nan
    return round(float(value), 2)


def _edge_verdict(score, signals):
    if signals < 3 or pd.isna(score):
        return "INSUFFICIENT"
    if score >= 70:
        return "STRONG"
    if score >= 55:
        return "OK"
    if score >= 45:
        return "WEAK"
    return "BAD"


def compute_historical_edge(
    ticker: str,
    data_map: dict,
    horizon: int = 10,
    stop_pct: float = BACKTEST_STOP_PCT,
    target_pct: float = BACKTEST_TARGET_PCT,
    max_lookback_bars: int = BACKTEST_MAX_LOOKBACK_BARS,
):
    """Backtests the core equity signal: bullish MACD cross while price is above 200EMA.
    This is not options P/L. It is a stock-move edge proxy used to block weak setups.
    """
    empty = {
        "Ticker": ticker, "Signals Tested": 0, "Historical Edge": np.nan, "Edge Verdict": "INSUFFICIENT",
        "1D Win %": np.nan, "3D Win %": np.nan, "5D Win %": np.nan, "10D Win %": np.nan,
        "Avg 1D %": np.nan, "Avg 3D %": np.nan, "Avg 5D %": np.nan, "Avg 10D %": np.nan,
        "Target First %": np.nan, "Stop First %": np.nan, "Failure %": np.nan,
        "Worst DD %": np.nan, "Avg Max DD %": np.nan, "Best Runup %": np.nan,
    }
    df = data_map.get(ticker)
    close = get_series(df, "Close")
    high = get_series(df, "High")
    low = get_series(df, "Low")
    common = close.index.intersection(high.index).intersection(low.index)
    if len(common) < 240:
        return empty
    close, high, low = close.loc[common], high.loc[common], low.loc[common]
    macd, sig, hist = compute_macd(close)
    ema200 = close.ewm(span=200, adjust=False).mean()
    n = len(close)
    start_i = max(200, n - max_lookback_bars)
    end_i = n - max(horizon, 10) - 1
    if end_i <= start_i:
        return empty

    trades = []
    for i in range(start_i, end_i + 1):
        if any(pd.isna(x) for x in [macd.iloc[i - 1], sig.iloc[i - 1], macd.iloc[i], sig.iloc[i], ema200.iloc[i], close.iloc[i]]):
            continue
        crossed = macd.iloc[i - 1] <= sig.iloc[i - 1] and macd.iloc[i] > sig.iloc[i]
        above200 = close.iloc[i] > ema200.iloc[i]
        if not (crossed and above200):
            continue

        entry = float(close.iloc[i])
        if entry <= 0:
            continue
        max_h = min(horizon, n - i - 1)
        if max_h < 5:
            continue
        future_close = close.iloc[i + 1:i + max_h + 1]
        future_high = high.iloc[i + 1:i + max_h + 1]
        future_low = low.iloc[i + 1:i + max_h + 1]
        if future_close.empty:
            continue

        def fwd_ret(days):
            if len(future_close) >= days:
                return (float(future_close.iloc[days - 1]) / entry - 1.0) * 100.0
            return np.nan

        stop_level = entry * (1.0 - stop_pct)
        target_level = entry * (1.0 + target_pct)
        outcome = "open"
        for hh, ll in zip(future_high, future_low):
            # Conservative daily-bar assumption: if both happen the same day, stop is counted first.
            if float(ll) <= stop_level:
                outcome = "stop"
                break
            if float(hh) >= target_level:
                outcome = "target"
                break

        fail_window = min(5, n - i - 1)
        failed = False
        signal_low = float(low.iloc[i])
        for j in range(i + 1, i + fail_window + 1):
            if (macd.iloc[j] < sig.iloc[j] and hist.iloc[j] < 0) or float(close.iloc[j]) < signal_low:
                failed = True
                break

        max_dd = (float(future_low.min()) / entry - 1.0) * 100.0
        best_runup = (float(future_high.max()) / entry - 1.0) * 100.0
        trades.append({
            "ret1": fwd_ret(1), "ret3": fwd_ret(3), "ret5": fwd_ret(5), "ret10": fwd_ret(min(10, max_h)),
            "target": outcome == "target", "stop": outcome == "stop", "failed": failed,
            "max_dd": max_dd, "best_runup": best_runup,
        })

    if not trades:
        return empty
    tdf = pd.DataFrame(trades)
    signals = int(len(tdf))
    win10 = float((tdf["ret10"] > 0).mean() * 100.0)
    win5 = float((tdf["ret5"] > 0).mean() * 100.0) if "ret5" in tdf else np.nan
    avg10 = float(tdf["ret10"].mean())
    avg5 = float(tdf["ret5"].mean()) if "ret5" in tdf else np.nan
    target_rate = float(tdf["target"].mean() * 100.0)
    stop_rate = float(tdf["stop"].mean() * 100.0)
    failure_rate = float(tdf["failed"].mean() * 100.0)
    avg_dd = float(tdf["max_dd"].mean())

    # Conservative score. Bad history blocks a pretty-looking current cross.
    win_score = np.clip((win10 - 40.0) / 30.0, 0, 1) * 35
    avg_score = np.clip((avg10 + 1.0) / 5.0, 0, 1) * 25
    target_score = np.clip((target_rate - stop_rate + 20.0) / 60.0, 0, 1) * 15
    dd_score = np.clip((avg_dd + 8.0) / 5.0, 0, 1) * 10
    failure_score = np.clip((50.0 - failure_rate) / 50.0, 0, 1) * 10
    sample_score = np.clip(signals / 8.0, 0, 1) * 5
    edge_score = round(float(win_score + avg_score + target_score + dd_score + failure_score + sample_score))
    edge_score = int(max(0, min(100, edge_score)))

    return {
        "Ticker": ticker,
        "Signals Tested": signals,
        "Historical Edge": edge_score,
        "Edge Verdict": _edge_verdict(edge_score, signals),
        "1D Win %": _pct((tdf["ret1"] > 0).mean() * 100.0),
        "3D Win %": _pct((tdf["ret3"] > 0).mean() * 100.0),
        "5D Win %": _pct(win5),
        "10D Win %": _pct(win10),
        "Avg 1D %": _pct(tdf["ret1"].mean()),
        "Avg 3D %": _pct(tdf["ret3"].mean()),
        "Avg 5D %": _pct(avg5),
        "Avg 10D %": _pct(avg10),
        "Target First %": _pct(target_rate),
        "Stop First %": _pct(stop_rate),
        "Failure %": _pct(failure_rate),
        "Worst DD %": _pct(tdf["max_dd"].min()),
        "Avg Max DD %": _pct(avg_dd),
        "Best Runup %": _pct(tdf["best_runup"].max()),
    }


def build_historical_edge_table(data_map: dict, tickers: list, horizon: int = 10):
    rows = []
    seen = set()
    for t in tickers:
        t = str(t).upper().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        row = compute_historical_edge(t, data_map, horizon=horizon)
        row["Sector"] = get_sector(t)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if "Historical Edge" in out.columns:
        out = out.sort_values(["Historical Edge", "Signals Tested"], ascending=[False, False], na_position="last")
    return out


def merge_historical_edge(scan_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    if scan_df is None or scan_df.empty:
        return pd.DataFrame()
    if edge_df is None or edge_df.empty:
        return scan_df.copy()
    drop_cols = [c for c in edge_df.columns if c in scan_df.columns and c not in ["Ticker", "Sector"]]
    base = scan_df.drop(columns=drop_cols, errors="ignore")
    edge_merge = edge_df.drop(columns=["Sector"], errors="ignore")
    return base.merge(edge_merge, on="Ticker", how="left")


# ============================================================
# SIMPLE MACD ZONE ENGINE — category first, low-click view
# ============================================================

SECTION_ORDER = [
    "BELOW ZERO CURL",
    "BELOW ZERO FRESH CROSS",
    "ZERO LINE",
    "ABOVE ZERO CURL",
    "ABOVE ZERO FRESH CROSS",
    "AVOID / NO CHASE",
]

HOME_COLUMNS = ["Ticker", "MACD HIST", "SETUP", "Price", "Trigger", "Action", "Risk"]
CARD_COLUMNS = [
    "Section", "Ticker", "Sector", "Price", "MACD HIST", "SETUP", "Timing",
    "Trigger", "Invalidation", "Action", "Risk", "Reason"
]
SLEEP_COLUMNS = ["Ticker", "MACD HIST", "SETUP", "Price", "Action", "Risk", "Reason"]


def _row_value(row, key, default="—"):
    try:
        value = row.get(key, default)
    except Exception:
        return default
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _as_float(value, default=np.nan):
    try:
        if value in ["—", "", None]:
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value, default=0):
    try:
        if value in ["—", "", None]:
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _edge_bad(row):
    verdict = str(_row_value(row, "Edge Verdict", ""))
    if verdict == "FILTER OFF":
        return False
    edge = _as_float(_row_value(row, "Historical Edge", np.nan))
    signals = _as_int(_row_value(row, "Signals Tested", 0))
    return signals >= 3 and not pd.isna(edge) and edge < 45


def _macd_score(row):
    edge = _as_float(_row_value(row, "Historical Edge", np.nan))
    if pd.isna(edge):
        return np.nan
    return int(round(edge))


def timing_label_for(row):
    lifecycle = str(_row_value(row, "Lifecycle", ""))
    bars = _as_int(_row_value(row, "Bars Since Cross", np.nan), default=-1)
    proximity = _as_float(_row_value(row, "Cross Proximity %", np.nan))

    if lifecycle in ["Pre-Cross", "Early Curl"]:
        if pd.isna(proximity):
            return "Curling"
        return f"{int(round(proximity))}% close"
    if lifecycle == "Provisional":
        return "Crossing today"
    if lifecycle in ["Fresh", "Fresh Caution"]:
        return f"{max(bars, 0)} bars fresh"
    if lifecycle in ["Late", "Extended", "Old Cross"]:
        return "Late"
    if lifecycle == "Failed":
        return "Failed"
    if lifecycle == "Under 200":
        return "Under 200"
    return "No edge"


def is_hard_block(row):
    lifecycle = str(_row_value(row, "Lifecycle", ""))
    flags = str(_row_value(row, "Risk Flags", ""))
    above200 = str(_row_value(row, "Above 200EMA", "No")) == "Yes"
    hard_flags = ["FAILED CROSS", "Market hostile", "Below 200EMA", "Weak options proxy"]
    return lifecycle in ["Failed", "Under 200"] or any(x in flags for x in hard_flags) or not above200 or _edge_bad(row)


def current_section(row):
    lifecycle = str(_row_value(row, "Lifecycle", ""))
    macd_pct = _as_float(_row_value(row, "MACD %", np.nan))
    macd_val = _as_float(_row_value(row, "MACD Value", np.nan))
    bars = _as_int(_row_value(row, "Bars Since Cross", np.nan), default=99)
    above50 = str(_row_value(row, "Above 50SMA", "No")) == "Yes"
    above200 = str(_row_value(row, "Above 200EMA", "No")) == "Yes"

    if is_hard_block(row) or lifecycle in ["Late", "Extended", "Old Cross", "No Setup"]:
        return "AVOID / NO CHASE"

    # Fresh means day-of cross or first few bars only.
    is_fresh = lifecycle == "Provisional" or (lifecycle in ["Fresh", "Fresh Caution"] and bars <= 3)
    is_curl = lifecycle in ["Pre-Cross", "Early Curl", "Recovering"]
    is_live_setup = lifecycle in ["Pre-Cross", "Early Curl", "Recovering", "Provisional", "Fresh", "Fresh Caution"]

    # Zero-line reset: uptrend held, MACD cooled near zero, then starts again.
    near_zero = (not pd.isna(macd_pct)) and abs(macd_pct) <= 0.25
    if is_live_setup and near_zero and above50 and above200:
        return "ZERO LINE"

    if pd.isna(macd_val):
        return "AVOID / NO CHASE"

    if macd_val < 0:
        if is_fresh:
            return "BELOW ZERO FRESH CROSS"
        if is_curl:
            return "BELOW ZERO CURL"

    if macd_val >= 0:
        if is_fresh:
            return "ABOVE ZERO FRESH CROSS"
        if is_curl:
            return "ABOVE ZERO CURL"

    return "AVOID / NO CHASE"


def section_rank(section):
    try:
        return SECTION_ORDER.index(str(section))
    except Exception:
        return 99


def clean_trade_ok(row):
    section = str(_row_value(row, "Section", ""))
    score = _macd_score(row)
    setup = _as_int(_row_value(row, "Setup Score", _row_value(row, "Score", 0)))
    flags = str(_row_value(row, "Risk Flags", "Clean"))
    above50 = str(_row_value(row, "Above 50SMA", "No")) == "Yes"
    above200 = str(_row_value(row, "Above 200EMA", "No")) == "Yes"
    sector = str(_row_value(row, "Sector State", "Unknown"))
    is_clean = flags == "Clean" or flags.strip() == ""
    strong_hist = (not pd.isna(score)) and score >= 75
    strong_now = setup >= 70
    good_section = section in ["BELOW ZERO FRESH CROSS", "BELOW ZERO CURL", "ZERO LINE"]
    return all([good_section, strong_hist, strong_now, above50, above200, sector != "Weak", is_clean])


def quick_action(row):
    section = str(_row_value(row, "Section", ""))
    score = _macd_score(row)
    above50 = str(_row_value(row, "Above 50SMA", "No")) == "Yes"

    if section == "AVOID / NO CHASE" or is_hard_block(row):
        return "AVOID"

    if clean_trade_ok(row):
        if section == "BELOW ZERO FRESH CROSS":
            return "TRADE"
        return "WATCH"

    if section in ["BELOW ZERO FRESH CROSS", "BELOW ZERO CURL", "ZERO LINE"]:
        if not above50:
            return "WAIT"
        if not pd.isna(score) and score >= 60:
            return "WATCH"
        return "WAIT"

    if section in ["ABOVE ZERO FRESH CROSS", "ABOVE ZERO CURL"]:
        if not pd.isna(score) and score >= 75 and above50:
            return "WATCH"
        return "WAIT"

    return "WAIT"


def risk_short(row):
    flags = str(_row_value(row, "Risk Flags", "Clean"))
    section = str(_row_value(row, "Section", ""))
    if flags == "Clean" or flags.strip() == "":
        return "Clean"
    first = flags.split(",")[0].strip()
    replacements = {
        "FAILED CROSS": "Failed",
        "Market hostile": "Market",
        "Weak sector": "Sector",
        "Below 50SMA": "50SMA",
        "Below 200EMA": "200EMA",
        "Weak RS vs SPY": "Weak RS",
        "Lagging sector": "Lagging",
        "Weak volume": "Volume",
        "Extended": "Late",
        "Far from 52W high": "Far high",
    }
    if section == "AVOID / NO CHASE" and first == "Clean":
        return "Late"
    return replacements.get(first, first[:14])


def simple_reason(row):
    section = str(_row_value(row, "Section", ""))
    timing = str(_row_value(row, "Timing", ""))
    score = _macd_score(row)
    score_txt = "No score" if pd.isna(score) else f"MACD {score}"
    if section == "BELOW ZERO CURL":
        return f"Early. Below zero. {score_txt}."
    if section == "BELOW ZERO FRESH CROSS":
        return f"Fresh. Below zero. {score_txt}."
    if section == "ZERO LINE":
        return f"Trend reset. {score_txt}."
    if section == "ABOVE ZERO CURL":
        return f"Curl. Above zero. {score_txt}."
    if section == "ABOVE ZERO FRESH CROSS":
        return f"Fresh. Above zero. {score_txt}."
    if timing == "Late":
        return "Late. Do not chase."
    return f"Skip. {risk_short(row)}."


def short_trigger(row):
    trig = str(_row_value(row, "Trigger", "—"))
    trig = trig.replace("prior high", "high")
    trig = trig.replace("cross high", "cross high")
    trig = trig.replace("Near close", "Close")
    trig = trig.replace("Wait pullback/retest", "Wait reset")
    trig = trig.replace("No entry trigger", "None")
    return trig


def enrich_decisions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["Timing"] = out.apply(timing_label_for, axis=1)
    out["Section"] = out.apply(current_section, axis=1)
    out["Signal"] = out["Section"]
    out["Decision"] = out["Section"]
    out["MACD Score"] = out.apply(_macd_score, axis=1)
    out["Setup Score"] = pd.to_numeric(out.get("Score", 0), errors="coerce").fillna(0).astype(int)
    out["MACD HIST"] = out["MACD Score"]
    out["SETUP"] = out["Setup Score"]
    out["Action"] = out.apply(quick_action, axis=1)
    out["Risk"] = out.apply(risk_short, axis=1)
    out["Reason"] = out.apply(simple_reason, axis=1)
    out["Trigger"] = out.apply(short_trigger, axis=1)
    out["_section_order"] = out["Section"].map(section_rank).fillna(99)
    sort_cols = ["_section_order", "MACD Score", "Setup Score", "Cross Proximity %"]
    sort_cols = [c for c in sort_cols if c in out.columns]
    out = out.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1), na_position="last")
    return out.drop(columns=["_section_order"], errors="ignore")


def top_section(df: pd.DataFrame, section: str, limit: int = 8) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df[df["Section"] == section].copy()
    if out.empty:
        return out
    return out.sort_values(["MACD Score", "Setup Score", "Cross Proximity %"], ascending=[False, False, False], na_position="last").head(limit)


def linked_column_config(extra=None):
    cfg = {
        "Ticker": st.column_config.LinkColumn("Ticker", display_text=r"symbol=([^&]+)", width="small"),
        "ETF": st.column_config.LinkColumn("ETF", display_text=r"symbol=([^&]+)", width="small"),
        "Sector ETF": st.column_config.LinkColumn("Sector ETF", display_text=r"symbol=([^&]+)", width="small"),
        "Score": st.column_config.NumberColumn("Score", min_value=0, max_value=100, format="%d"),
        "Setup Score": st.column_config.NumberColumn("Setup", min_value=0, max_value=100, format="%d"),
        "SETUP": st.column_config.NumberColumn("SETUP", min_value=0, max_value=100, format="%d"),
        "Historical Edge": st.column_config.NumberColumn("History", min_value=0, max_value=100, format="%d"),
        "MACD Score": st.column_config.NumberColumn("MACD HIST", min_value=0, max_value=100, format="%d"),
        "MACD HIST": st.column_config.NumberColumn("MACD HIST", min_value=0, max_value=100, format="%d"),
    }
    if extra:
        cfg.update(extra)
    return cfg


def render_home_table(df: pd.DataFrame, height=260):
    if df is None or df.empty:
        st.info("Empty")
        return
    cols = [c for c in HOME_COLUMNS if c in df.columns]
    st.dataframe(
        linked_display_df(df, cols),
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=linked_column_config(),
    )


def render_sleep_table(df: pd.DataFrame, height=360):
    if df is None or df.empty:
        st.info("Empty")
        return
    cols = [c for c in SLEEP_COLUMNS if c in df.columns]
    st.dataframe(
        linked_display_df(df, cols),
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=linked_column_config(),
    )


def render_trade_card(row):
    ticker = str(_row_value(row, "Ticker", "—"))
    section = str(_row_value(row, "Section", "—"))
    timing = str(_row_value(row, "Timing", "—"))
    score = _row_value(row, "MACD HIST", _row_value(row, "MACD Score", "—"))
    setup = _row_value(row, "SETUP", _row_value(row, "Setup Score", _row_value(row, "Score", "—")))
    trigger = str(_row_value(row, "Trigger", "—"))
    invalidation = str(_row_value(row, "Invalidation", "—"))
    action = str(_row_value(row, "Action", "—"))
    risk = str(_row_value(row, "Risk", "—"))
    reason = str(_row_value(row, "Reason", "—"))
    chart = tv_chart_url(ticker)

    st.markdown(f"### [{ticker}]({chart})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Section", section)
    c2.metric("MACD HIST", score)
    c3.metric("SETUP", setup)
    c4.metric("Action", action)
    st.write(f"**Why:** {reason}")
    st.write(f"**Timing:** {timing}")
    st.write(f"**Trigger:** {trigger}")
    st.write(f"**Stop:** {invalidation}")
    st.write(f"**Risk:** {risk}")


def today_call(regime, call_trading, sections):
    if regime == "HOSTILE" or call_trading == "OFF":
        return "CASH", "Market off"

    pool = []
    for section in ["BELOW ZERO FRESH CROSS", "BELOW ZERO CURL", "ZERO LINE", "ABOVE ZERO FRESH CROSS", "ABOVE ZERO CURL"]:
        df = sections.get(section, pd.DataFrame())
        if df is not None and not df.empty:
            pool.append(df)
    if not pool:
        return "CASH", "None"

    all_df = pd.concat(pool, ignore_index=True)
    trade_df = all_df[all_df["Action"] == "TRADE"].copy()
    if not trade_df.empty:
        top = trade_df.sort_values(["MACD Score", "Setup Score"], ascending=[False, False], na_position="last").iloc[0]
        return "TRADE", str(top.get("Ticker", "None"))

    watch_df = all_df[all_df["Action"] == "WATCH"].copy()
    if not watch_df.empty:
        top = watch_df.sort_values(["MACD Score", "Setup Score"], ascending=[False, False], na_position="last").iloc[0]
        return "WATCH", str(top.get("Ticker", "None"))

    return "WAIT", "None"

# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .sf-clean-note {
        border: 1px solid rgba(224, 176, 110, 0.28);
        background: rgba(31, 45, 32, 0.60);
        border-radius: 18px;
        padding: 14px 16px;
        margin: 8px 0 18px 0;
        color: #f3ead7;
    }
    </style>
    <div class="sf-hero sf-hero-compact">
        <div class="sf-hero-title">🦊 SilverFoxFlow Mach 8.5.1</div>
        <div class="sf-hero-sub">Clean MACD Filter</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    universe_choice = st.selectbox("Universe", ["Top 100", "Top 150", "Top 200", "Liquid Core Only", "Full S&P 500"], index=1)
    data_mode = st.selectbox("Data", ["Daily + live overlay", "Daily only"], index=0)
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    custom_text = st.text_area("Add tickers", placeholder="TSM, ARM, MSTR", height=70)

    with st.expander("Advanced", expanded=False):
        mode = st.radio("Mode", list(MODE_CONFIG.keys()), index=1)
        pre_sensitivity = st.selectbox("Curl filter", ["Balanced", "Strict", "Early"], index=0)
        backtest_horizon = st.selectbox("MACD score days", [10, 15, 20], index=0)
        include_etfs = st.checkbox("ETFs", value=True)
        edge_filter = st.checkbox("Use MACD score", value=True)
        ignore_market_gate = st.checkbox("Ignore market", value=False)
        table_detail = st.radio("Audit", ["Clean", "Full"], index=0, horizontal=True)
        auto_refresh = st.selectbox("Auto refresh", ["Off", "5 min", "15 min"], index=0)
    st.caption("Simple. Fast. No chase.")

if "mode" not in globals():
    mode = "Balanced"
if "pre_sensitivity" not in globals():
    pre_sensitivity = "Balanced"
if "backtest_horizon" not in globals():
    backtest_horizon = 10
if "include_etfs" not in globals():
    include_etfs = True
if "edge_filter" not in globals():
    edge_filter = True
if "ignore_market_gate" not in globals():
    ignore_market_gate = False
if "table_detail" not in globals():
    table_detail = "Clean"
if "auto_refresh" not in globals():
    auto_refresh = "Off"

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
with st.spinner(f"Loading {len(all_download_tickers)} symbols..."):
    history_raw = download_history(all_download_tickers, period=period, interval="1d")

intraday_latest = {}
use_live_overlay = data_mode == "Daily + live overlay"
if use_live_overlay:
    with st.spinner("Live update..."):
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
with st.spinner(f"Scanning {len(scan_tickers)} symbols..."):
    for ticker in scan_tickers:
        row = analyze_ticker(
            ticker, history, regime, mode,
            ignore_market_gate=ignore_market_gate,
            pre_sensitivity=pre_sensitivity,
            provisional_live=use_live_overlay,
        )
        if row:
            rows.append(row)
scan_df = pd.DataFrame(rows)

if scan_df.empty:
    st.info("No rows.")
    st.stop()

with st.spinner("MACD score..."):
    edge_df = build_historical_edge_table(history_raw, scan_tickers, horizon=int(backtest_horizon))
scan_df = merge_historical_edge(scan_df, edge_df)
if not edge_filter:
    scan_df["Historical Edge"] = np.nan
    scan_df["Edge Verdict"] = "FILTER OFF"
scan_df = enrich_decisions(scan_df)

sections = {name: top_section(scan_df, name, limit=8) for name in SECTION_ORDER}
today_action, best_setup = today_call(regime, call_trading, sections)

st.subheader("TODAY")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Action", today_action)
c2.metric("Best", best_setup)
c3.metric("Market", regime)
c4.metric("Calls", call_trading)

st.markdown(
    f"""
    <div class="sf-clean-note">
        <b>Scanned:</b> {len(scan_tickers)} &nbsp; | &nbsp;
        <b>Data:</b> {latest_data_label(history, intraday_latest)} &nbsp; | &nbsp;
        <b>Rule:</b> Below zero first. Clean trades only.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    st.markdown("### BELOW ZERO CURL")
    render_home_table(sections["BELOW ZERO CURL"], height=285)
with right:
    st.markdown("### BELOW ZERO FRESH CROSS")
    render_home_table(sections["BELOW ZERO FRESH CROSS"], height=285)

st.markdown("### ZERO LINE")
render_home_table(sections["ZERO LINE"], height=260)

left, right = st.columns(2)
with left:
    st.markdown("### ABOVE ZERO CURL")
    render_home_table(sections["ABOVE ZERO CURL"], height=285)
with right:
    st.markdown("### ABOVE ZERO FRESH CROSS")
    render_home_table(sections["ABOVE ZERO FRESH CROSS"], height=285)

with st.expander("AVOID / NO CHASE", expanded=False):
    render_sleep_table(sections["AVOID / NO CHASE"], height=420)

watch_pool = pd.concat(
    [sections[x] for x in SECTION_ORDER if x != "AVOID / NO CHASE"],
    ignore_index=True,
)
watch_pool = watch_pool.drop_duplicates("Ticker") if not watch_pool.empty else pd.DataFrame()

with st.expander("Ticker Card", expanded=False):
    if watch_pool.empty:
        st.info("No names.")
    else:
        labels = []
        for _, r in watch_pool.iterrows():
            labels.append(f"{r.get('Ticker')} — {r.get('Section')} — HIST {r.get('MACD HIST')}")
        selected_label = st.selectbox("Ticker", labels)
        selected_ticker = selected_label.split(" — ")[0]
        selected_row = watch_pool[watch_pool["Ticker"] == selected_ticker].iloc[0]
        render_trade_card(selected_row)

with st.expander("Advanced", expanded=False):
    st.markdown("#### Full Scan")
    render_table(scan_df, height=520)
    st.markdown("#### MACD Score Detail")
    if edge_df.empty:
        st.info("No score table.")
    else:
        render_table(edge_df.head(60), height=420)
    st.markdown("#### Market")
    st.dataframe(linked_display_df(market_df), use_container_width=True, hide_index=True, column_config=linked_column_config())

with st.expander("Exports", expanded=False):
    st.download_button("Full Scan CSV", make_download(scan_df), "silverfoxflow_mach8_5_1_full_scan.csv", "text/csv")
    for name in SECTION_ORDER:
        file_name = name.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")
        st.download_button(f"{name} CSV", make_download(sections[name]), f"silverfoxflow_mach8_5_1_{file_name}.csv", "text/csv")
    st.download_button("MACD Score CSV", make_download(edge_df), "silverfoxflow_mach8_5_1_macd_score.csv", "text/csv")
    journal_cols = [
        "Date", "Ticker", "Section", "MACD HIST", "SETUP", "Entry Type",
        "Option Contract", "Entry Price", "Stop Rule", "Target", "Exit Price", "Result %",
        "Reason Entered", "Reason Exited", "Mistake Tag", "Screenshot Link", "Notes"
    ]
    journal_template = pd.DataFrame(columns=journal_cols)
    st.download_button("Journal CSV", make_download(journal_template), "silverfoxflow_trade_journal_template.csv", "text/csv")
