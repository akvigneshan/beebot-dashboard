# BeeBot Dashboard

📊 A Streamlit-based dashboard to visualize BeeBot trading runs, signals, trades, and performance metrics.

## Features
- Run analytics (buy/sell/skip trends, success rate)
- Trade history with scatterplots
- Skipped reasons analysis
- Symbol-level analytics and leaderboard
- PnL simulation (per trade, cumulative, win rate)
- Portfolio equity curve + drawdown analysis
- Performance metrics (Sharpe, expectancy, avg win/loss)
- Date and symbol filters

## 🚀 Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app → select this repo → choose `dashboard_app_with_leaderboard.py` as the main file.
4. Deploy!

You’ll get a public URL like:
```
https://your-app-name.streamlit.app
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard_app_with_leaderboard.py
```
