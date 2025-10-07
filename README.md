# 🦊 SilverFoxFlow — Streamlit Starter (UOA 2.0)

This repo deploys a **Streamlit web app** that implements your **UOA 2.0** scoring and verdicts.
It ships with a **mock data provider** so you can deploy instantly, then swap in a real options-flow API later.

## Quickstart (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501

## Deploy on Streamlit Cloud (fastest)
1. Push this folder to a **GitHub** repo (e.g., `silverfoxflow`).
2. Go to https://share.streamlit.io → **New app** → connect your repo → pick `app.py`.
3. After deploy, open **Settings → Custom Domain** and follow Streamlit’s instructions.
   - You’ll add a **CNAME** in GoDaddy DNS pointing `www.silverfoxflow.com` to the value they give you.
   - Then redirect the apex (`silverfoxflow.com`) to `www` via GoDaddy "Forwarding" OR A-record to Streamlit (if supported).

## Daily 6:40 AM PT Scan (optional, via GitHub Actions)
- By default, the app can create a report with **mock data** via the “Run Fresh Scan” button.
- To automate a daily report:
  1. Keep using mock initially, or wire a real provider in `providers/`.
  2. GitHub Action (`.github/workflows/daily_scan.yml`) runs `python scan.py` and commits `data/report.json`.
  3. The Streamlit app reads `data/report.json` from the repo on each page load (served with the app).

> Tip: When you integrate a real data vendor (Polygon/Tradier/etc.), store API keys in **Streamlit Secrets** and/or GitHub Actions secrets.

## Swap in a Real Provider
- Implement a new module in `providers/yourvendor.py` that returns:
  - `List[List[Trade]]` (groups per ticker) and `contexts: Dict[str, Context]`.
- Replace imports in `scan.py` and `app.py` to your provider.

## Files
- `app.py` — Streamlit UI: shows Top signals and Morning Verdict
- `uoa_engine.py` — UOA 2.0 scoring engine and filters
- `providers/mock.py` — mock data generator (works out of the box)
- `scan.py` — CLI that writes `data/report.json` (used by the scheduled job)
- `.github/workflows/daily_scan.yml` — optional daily scan
- `.streamlit/config.toml` — dark theme
