"""
CLI to run a market scan and write a JSON report.
By default uses the mock provider so you can deploy immediately.
Later, swap `providers.mock` with a real data source.
"""
import json, os
from datetime import datetime
from providers.mock import generate_mock_market
from uoa_engine import score_market

def main():
    market, contexts = generate_mock_market()
    ideas = score_market(market, contexts)
    report = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "verdicts": [idea.__dict__ for idea in ideas[:5]]
    }
    os.makedirs("data", exist_ok=True)
    with open("data/report.json","w") as f:
        json.dump(report, f, indent=2)
    print("Wrote data/report.json")

if __name__ == "__main__":
    main()
