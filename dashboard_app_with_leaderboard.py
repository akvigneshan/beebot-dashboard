
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "beebot.db"

# --- DB Loader ---
def get_tables():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return {r[0] for r in rows}

def load_table(name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    conn.close()
    return df

st.set_page_config(page_title="BeeBot Dashboard", layout="wide")

st.title("📊 BeeBot Trading Dashboard")

# --- Load Data ---
tables = get_tables()
trades = load_table("trades") if "trades" in tables else pd.DataFrame()
skipped = load_table("skipped") if "skipped" in tables else pd.DataFrame()
runs = load_table("runs") if "runs" in tables else pd.DataFrame()
perf = load_table("performance") if "performance" in tables else pd.DataFrame()
run_details = load_table("run_details") if "run_details" in tables else pd.DataFrame()


# --- Global Date Filter ---
st.sidebar.header("🔎 Filters")
date_range = st.sidebar.date_input("Select Date Range", [])

def filter_by_date(df, col="timestamp"):
    if df.empty or not date_range or len(date_range) != 2:
        return df
    df[col] = pd.to_datetime(df[col])
    return df[(df[col].dt.date >= date_range[0]) & (df[col].dt.date <= date_range[1])]

# Apply global filter

# --- Symbol Filter ---
symbols_available = trades["symbol"].unique().tolist() if not trades.empty else []
selected_symbols = st.sidebar.multiselect("Filter by Symbols", symbols_available, default=symbols_available)

def filter_by_symbols(df, col="symbol"):
    if df.empty or not selected_symbols:
        return df
    return df[df[col].isin(selected_symbols)]

# Apply symbol filter to trades
trades = filter_by_symbols(trades, "symbol")

# Apply symbol filter to run_details (check buy/sell symbol strings)
if not run_details.empty and "buy_symbols" in run_details.columns:
    run_details = run_details[
        run_details["buy_symbols"].apply(lambda x: any(sym in x for sym in selected_symbols) if isinstance(x, str) else False) |
        run_details["sell_symbols"].apply(lambda x: any(sym in x for sym in selected_symbols) if isinstance(x, str) else False)
    ]

trades = filter_by_date(trades, "timestamp")
skipped = filter_by_date(skipped, "timestamp")
runs = filter_by_date(runs, "timestamp")
perf = filter_by_date(perf, "timestamp")
run_details = filter_by_date(run_details, "timestamp")


# --- KPIs ---
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
if not runs.empty:
    col1.metric("Total Runs", len(runs))
    col2.metric("Total Buys", runs["buy_count"].sum())
    col3.metric("Total Sells", runs["sell_count"].sum())
    col4.metric("Total Skips", runs["skip_count"].sum())
if not perf.empty:
    avg_sr = perf["success_rate"].mean()
    st.metric("Avg Success Rate", f"{avg_sr:.2f}%")

# --- Last Run ---
if not run_details.empty:
    st.subheader("🕒 Last Run Summary")
    last = run_details.iloc[-1]
    st.write(f"**Timestamp:** {last['timestamp']}")
    st.write(f"**Buys:** {last['buy_symbols']}")
    st.write(f"**Sells:** {last['sell_symbols']}")
    if last['skipped_reasons']:
        with st.expander("Skipped Reasons"):
            for reason in last['skipped_reasons'].split('\n'):
                st.write(f"- {reason}")

# --- Run Analytics ---
if not runs.empty and not perf.empty:
    st.header("📊 Run Analytics")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(runs["timestamp"], runs["buy_count"], label="Buys", marker="o")
    ax.plot(runs["timestamp"], runs["sell_count"], label="Sells", marker="o")
    ax.plot(runs["timestamp"], runs["skip_count"], label="Skips", marker="o")
    ax2 = ax.twinx()
    ax2.plot(perf["timestamp"], perf["success_rate"], label="Success Rate", linestyle="--")
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Run Counts and Success Rate Over Time")
    ax.legend(loc="upper left")
    st.pyplot(fig)

# --- Trades ---
if not trades.empty:
    st.header("💹 Trades")
    st.dataframe(trades.tail(20))
    fig, ax = plt.subplots(figsize=(10,5))
    buy = trades[trades["action"]=="BUY"]
    sell = trades[trades["action"]=="SELL"]
    ax.scatter(buy["timestamp"], buy["price"], marker="^", color="green", label="BUY")
    ax.scatter(sell["timestamp"], sell["price"], marker="v", label="SELL")
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Trade Prices Over Time")
    ax.legend()
    st.pyplot(fig)

# --- Skipped Reasons ---
if not run_details.empty:
    st.header("⏭️ Skipped Reasons")
    all_reasons = []
    for reasons in run_details["skipped_reasons"].dropna():
        all_reasons.extend(reasons.split("\n"))
    if all_reasons:
        reason_counts = pd.Series(all_reasons).value_counts().head(10)
        st.bar_chart(reason_counts)

        # Trend of skipped reasons per run
        skipped_counts = run_details.copy()
        skipped_counts["skip_count"] = skipped_counts["skipped_reasons"].apply(lambda x: len(x.split("\n")) if x else 0)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(skipped_counts["timestamp"], skipped_counts["skip_count"], marker="o")
        ax.set_title("Skipped Reasons Trend Over Runs")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# --- Symbol Analytics ---
if not trades.empty:
    st.header("📌 Symbol Analytics")
    symbol_counts = trades["symbol"].value_counts().head(10)
    st.bar_chart(symbol_counts)
    symbol_filter = st.selectbox("Filter trades by symbol", trades["symbol"].unique())
    st.dataframe(trades[trades["symbol"] == symbol_filter])


# --- PnL Simulation ---
if not trades.empty:
    st.header("💰 PnL Simulation")

    # Sort trades by timestamp
    trades_sorted = trades.sort_values("timestamp")

    pnl_records = []
    positions = {}

    for _, row in trades_sorted.iterrows():
        sym = row["symbol"]
        if row["action"] == "BUY":
            positions[sym] = {"buy_price": row["price"], "timestamp": row["timestamp"]}
        elif row["action"] == "SELL" and sym in positions:
            buy_info = positions.pop(sym)
            buy_price = buy_info["buy_price"]
            sell_price = row["price"]
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            pnl_records.append({
                "symbol": sym,
                "buy_time": buy_info["timestamp"],
                "sell_time": row["timestamp"],
                "buy_price": buy_price,
                "sell_price": sell_price,
                "pnl_pct": pnl_pct
            })

    if pnl_records:
        pnl_df = pd.DataFrame(pnl_records)
        st.subheader("Trade PnL Records")
        st.dataframe(pnl_df)

        # Cumulative PnL
        pnl_df["cum_pnl"] = pnl_df["pnl_pct"].cumsum()
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(pnl_df["sell_time"], pnl_df["cum_pnl"], marker="o", )
        ax.set_title("Cumulative PnL Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative PnL (%)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # Win rate
        win_rate = (pnl_df["pnl_pct"] > 0).mean() * 100
        st.metric("Win Rate", f"{win_rate:.2f}%")

        # --- Performance Metrics ---
        st.subheader("📊 Performance Metrics")

        returns = pnl_df["pnl_pct"]

        # Sharpe Ratio (risk-free=0 assumption)
        sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # Avg Win / Avg Loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0
        st.metric("Average Win (%)", f"{avg_win:.2f}")
        st.metric("Average Loss (%)", f"{avg_loss:.2f}")

        # Expectancy per Trade
        win_rate = (returns > 0).mean()
        loss_rate = (returns < 0).mean()
        expectancy = win_rate * avg_win + loss_rate * avg_loss
        st.metric("Expectancy per Trade (%)", f"{expectancy:.2f}")

    else:
        st.write("No complete BUY/SELL pairs to calculate PnL yet.")





# --- Top Symbols Leaderboard ---
if not trades.empty and 'pnl_df' in locals():
    st.header("🏆 Top Symbols Leaderboard")

    pnl_df_grouped = pnl_df.groupby("symbol").agg(
        total_trades=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        cum_pnl=("pnl_pct", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100)
    ).reset_index().sort_values("avg_pnl", ascending=False)

    st.subheader("Symbol Performance Table")
    st.dataframe(pnl_df_grouped)

    if not pnl_df_grouped.empty:
        st.subheader("Top 10 Symbols by Avg PnL")
        st.bar_chart(pnl_df_grouped.set_index("symbol")["avg_pnl"].head(10))

        st.subheader("Bottom 10 Symbols by Avg PnL")
        st.bar_chart(pnl_df_grouped.set_index("symbol")["avg_pnl"].tail(10))


# --- Portfolio Equity Curve & Drawdown ---
if not trades.empty:
    st.header("📉 Portfolio Simulation")

    # User inputs
    initial_balance = st.number_input("Starting Balance ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
    date_range = st.date_input("Filter trades by date range", [])

    # Rebuild pnl_records from trades to ensure availability
    trades_sorted_eq = trades.sort_values("timestamp")
    pnl_records_eq = []
    positions_eq = {}
    for _, row in trades_sorted_eq.iterrows():
        sym = row["symbol"]
        ts = pd.to_datetime(row["timestamp"])
        if date_range and len(date_range) == 2:
            if ts.date() < date_range[0] or ts.date() > date_range[1]:
                continue
        if row["action"] == "BUY":
            positions_eq[sym] = {"buy_price": row["price"], "timestamp": row["timestamp"]}
        elif row["action"] == "SELL" and sym in positions_eq:
            buy_info = positions_eq.pop(sym)
            buy_price = buy_info["buy_price"]
            sell_price = row["price"]
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            pnl_records_eq.append({
                "symbol": sym,
                "buy_time": buy_info["timestamp"],
                "sell_time": row["timestamp"],
                "buy_price": buy_price,
                "sell_price": sell_price,
                "pnl_pct": pnl_pct
            })

    if pnl_records_eq:
        equity_vals = [initial_balance]
        timestamps_eq = []
        for rec in pnl_records_eq:
            new_balance = equity_vals[-1] * (1 + rec["pnl_pct"]/100.0)
            equity_vals.append(new_balance)
            timestamps_eq.append(rec["sell_time"])

        eq_df = pd.DataFrame({"timestamp": timestamps_eq, "equity": equity_vals[1:]})

        # Plot equity curve
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(eq_df["timestamp"], eq_df["equity"], marker="o")
        ax.set_title(f"Portfolio Equity Curve (Start ${initial_balance:,})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity ($)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # Drawdown calculation
        eq_df["cummax"] = eq_df["equity"].cummax()
        eq_df["drawdown"] = (eq_df["equity"] - eq_df["cummax"]) / eq_df["cummax"] * 100.0
        max_dd = float(eq_df["drawdown"].min()) if not eq_df.empty else 0.0

        # Plot drawdown
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(eq_df["timestamp"], eq_df["drawdown"], marker="o")
        ax.set_title("Portfolio Drawdown (%)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown (%)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    else:
        st.write("No completed BUY→SELL trade pairs available to simulate equity.")
