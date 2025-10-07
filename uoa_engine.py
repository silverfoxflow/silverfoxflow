"""
UOA 2.0 Scoring Engine for SilverFoxFlow
- Implements Smart Money Score (0-100)
- Filters: volume>OI, aggression, premium size, short-dated, clustering
- Context checks: VWAP/price, RSI/momentum, EMA trend, catalyst flag
This file is vendor-agnostic. Feed it normalized trade data.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math
import statistics as stats
from datetime import datetime, timedelta

@dataclass
class Trade:
    ticker: str
    type: str         # "CALL" or "PUT"
    strike: float
    expiry: str       # "YYYY-MM-DD"
    premium_usd: float
    vol: int          # option volume on that line
    oi: int           # open interest at time of trade
    side: str         # "AT_ASK", "AT_BID", "MID"
    dte: int          # days to expiry
    timestamp: str    # ISO
    ref_price: float  # stock price at trade time

@dataclass
class Context:
    price: float
    vwap_ok: bool         # price > VWAP for calls OR price < VWAP for puts
    rsi: float
    above_ema20: bool
    catalyst: bool        # earnings/FDA/news within window
    sector_bias: bool     # sector ETF confirmation

@dataclass
class ScoredIdea:
    ticker: str
    expiry: str
    strike: float
    type: str
    premium_usd: float
    score: int
    verdict: str
    reasons: List[str]

DEFAULT_WEIGHTS = {
    "aggression": 25,
    "vol_vs_oi": 15,
    "premium": 10,
    "cluster": 15,
    "vwap_confluence": 15,
    "catalyst": 10,
    "trend": 10,
}

def _cap01(x: float) -> float:
    return max(0.0, min(1.0, x))

def score_single(trades: List[Trade], ctx: Context, weights: Dict[str, int] = None) -> ScoredIdea:
    """Score a *group* of trades for the same ticker/expiry/type (cluster)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    reasons = []

    # --- Flow Aggression ---
    aggression_ratio = sum(1 for t in trades if (t.side == "AT_ASK" and t.type=="CALL") or (t.side == "AT_BID" and t.type=="PUT")) / max(1,len(trades))
    s_aggr = _cap01(aggression_ratio) * weights["aggression"]
    reasons.append(f"Aggression {aggression_ratio:.0%} of prints")

    # --- Volume vs OI (freshness) ---
    vol = sum(t.vol for t in trades)
    oi = max(1, sum(t.oi for t in trades))  # coarse: sum OI; real impl use chain-level OI
    freshness = _cap01((vol - oi) / max(1.0, oi)) if vol > oi else 0.0
    s_fresh = freshness * weights["vol_vs_oi"]
    reasons.append(f"Vol/OI freshness={freshness:.2f}")

    # --- Premium size ---
    prem = sum(t.premium_usd for t in trades)
    # scale: 0 at 100k, 1 at >= 5M
    prem_score = _cap01((prem - 100_000) / (5_000_000 - 100_000))
    s_prem = prem_score * weights["premium"]
    reasons.append(f"Premium ${prem:,.0f}")

    # --- Cluster detection ---
    # naive: count trades and tight time span
    times = [datetime.fromisoformat(t.timestamp) for t in trades]
    span_min = (max(times) - min(times)).total_seconds() / 60 if times else 999
    cluster_ok = (len(trades) >= 3 and span_min <= 10)
    s_cluster = (1.0 if cluster_ok else 0.0) * weights["cluster"]
    reasons.append(f"Cluster {'yes' if cluster_ok else 'no'} ({len(trades)} trades/{span_min:.1f}m)")

    # --- VWAP confluence ---
    vwap_ok = ctx.vwap_ok
    s_vwap = (1.0 if vwap_ok else 0.0) * weights["vwap_confluence"]
    reasons.append(f"VWAP confluence={'yes' if vwap_ok else 'no'}")

    # --- Catalyst ---
    s_catalyst = (1.0 if ctx.catalyst else 0.0) * weights["catalyst"]
    reasons.append(f"Catalyst={'yes' if ctx.catalyst else 'no'}")

    # --- Trend ---
    trend_ok = (ctx.above_ema20 and trades[0].type=="CALL") or ((not ctx.above_ema20) and trades[0].type=="PUT")
    s_trend = (1.0 if trend_ok else 0.0) * weights["trend"]
    reasons.append(f"Trend OK={'yes' if trend_ok else 'no'}")

    total = int(round(s_aggr + s_fresh + s_prem + s_cluster + s_vwap + s_catalyst + s_trend))

    # Verdict logic
    if total >= 75 and ctx.price > 0:
        verdict = "BUY CALLS" if trades[0].type=="CALL" and ctx.vwap_ok else ("BUY PUTS" if trades[0].type=="PUT" and ctx.vwap_ok else "NO TRADE")
    elif 60 <= total < 75:
        verdict = "WATCHLIST"
    else:
        verdict = "NO TRADE"

    # Representative contract details
    rep = max(trades, key=lambda t: t.premium_usd)
    return ScoredIdea(
        ticker=rep.ticker,
        expiry=rep.expiry,
        strike=rep.strike,
        type=rep.type,
        premium_usd=prem,
        score=total,
        verdict=verdict,
        reasons=reasons
    )

def score_market(trade_groups: List[List[Trade]], contexts: Dict[str, Context], weights: Dict[str,int]=None) -> List[ScoredIdea]:
    ideas = []
    for group in trade_groups:
        if not group:
            continue
        tkr = group[0].ticker
        ctx = contexts.get(tkr)
        if not ctx:
            continue
        # Basic noise filters (UOA 2.0): DTE <= 56, ignore deep long-dated, ignore non-aggressive
        if any(t.dte > 56 for t in group):
            continue
        if all(t.side == "MID" for t in group):
            continue
        # Require some sign of freshness
        if sum(t.vol for t in group) <= sum(t.oi for t in group):
            continue
        idea = score_single(group, ctx, weights)
        ideas.append(idea)
    # Sort by score desc
    ideas.sort(key=lambda i: i.score, reverse=True)
    return ideas
