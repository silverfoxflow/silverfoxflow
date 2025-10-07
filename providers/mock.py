"""
Mock provider: generates sample option flow + context so the app works out-of-the-box.
Replace with a real provider (Polygon/Tradier/etc.) later.
"""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta
import random

from uoa_engine import Trade, Context

def generate_mock_market(now=None):
    if now is None:
        now = datetime.utcnow()
    tickers = ["NVDA","META","TSLA","AMD","MRVL","NOC"]
    market = []
    contexts: Dict[str, Context] = {}
    for t in tickers:
        # Random bullish/bearish tilt
        bullish = random.random() > 0.4
        price = round(random.uniform(50, 1200), 2)
        vwap_ok = bullish  # pretend price>VWAP if bullish
        above_ema20 = bullish
        rsi = 58 if bullish else 42
        catalyst = random.random() < 0.3
        sector_bias = random.random() < 0.5

        contexts[t] = Context(
            price=price,
            vwap_ok=vwap_ok,
            rsi=rsi,
            above_ema20=above_ema20,
            catalyst=catalyst,
            sector_bias=sector_bias
        )

        # Create 2-5 trades as a "cluster"
        n = random.randint(2,5)
        group: List[Trade] = []
        for i in range(n):
            is_call = bullish
            side = "AT_ASK" if is_call else "AT_BID"
            dte = random.choice([7,14,21,28,35,42,49,56])
            strike = round(price * (1.05 if is_call else 0.95),2)
            premium = random.choice([150000, 350000, 750000, 1200000, 3000000])
            vol = random.choice([100, 250, 500, 1000, 1500])
            oi = int(vol * random.uniform(0.2, 0.6))  # keep vol>oi
            ts = (now + timedelta(minutes=i*2)).isoformat()
            group.append(Trade(
                ticker=t,
                type="CALL" if is_call else "PUT",
                strike=strike,
                expiry=(now + timedelta(days=dte)).date().isoformat(),
                premium_usd=premium,
                vol=vol,
                oi=oi,
                side=side,
                dte=dte,
                timestamp=ts,
                ref_price=price
            ))
        market.append(group)
    return market, contexts
