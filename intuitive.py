import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# === Generate Synthetic Portfolio Data ===
np.random.seed(2024)
start_date = pd.to_datetime("2017-06-01")
end_date = pd.to_datetime("2025-06-30")
dates = pd.date_range(start_date, end_date, freq="W")  # weekly data

n = len(dates)
# Simulate log returns for each strategy
# Optimized: moderate mean, low volatility
# EqualWeighted: similar mean, slightly higher vol
# BTCOnly (BuyAndHold): higher mean, highest volatility

# Choose annualized means and vols
ann_mu = {'Optimized': 0.10, 'EqualWeighted': 0.10, 'BTCOnly': 0.16}
ann_sigma = {'Optimized': 0.50, 'EqualWeighted': 0.60, 'BTCOnly': 0.80}

weeks_per_year = 52
mu = {k: v/weeks_per_year for k,v in ann_mu.items()}
sigma = {k: v/np.sqrt(weeks_per_year) for k,v in ann_sigma.items()}

rets = {k: np.random.normal(mu[k], sigma[k], n) for k in mu}
# Add some random big negative shocks for realism
for k in rets:
    shock_dates = np.random.choice(n, size=int(n*0.03), replace=False)
    rets[k][shock_dates] += np.random.uniform(-0.15, -0.05, len(shock_dates))

data = {
    "date": dates,
    "Optimized": 1 + np.cumsum(rets['Optimized']),
    "EqualWeighted": 1 + np.cumsum(rets['EqualWeighted']),
    "BTCOnly": 1 + np.cumsum(rets['BTCOnly'])
}
# Make all series positive
for k in ["Optimized","EqualWeighted","BTCOnly"]:
    minv = np.min(data[k])
    if minv < 0.1:
        data[k] = np.array(data[k]) - minv + 0.1

df = pd.DataFrame(data)
print("First few rows of synthetic data:\n", df.head())
print("Last few rows of synthetic data:\n", df.tail())

# === Save synthetic CSV just like system output ===
csv_path = "synthetic_portfolio_history.csv"
df.to_csv(csv_path, index=False)
print(f"\nSynthetic portfolio data saved to: {csv_path}")

# === Analysis & Visualization Pipeline ===
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Normalize series for growth of $1
for col in ["Optimized","EqualWeighted","BTCOnly"]:
    df[col + "_norm"] = df[col] / df[col].iloc[0]
    print(f"First 5 normalized values for {col}: {df[col + '_norm'].head().values}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['Optimized_norm'], label='Optimized', linewidth=2)
plt.plot(df['date'], df['EqualWeighted_norm'], label='EqualWeighted', linewidth=2)
plt.plot(df['date'], df['BTCOnly_norm'], label='BuyAndHold (BTC-Only)', linewidth=2)
plt.title("Portfolio Performance: Optimized vs Equal Weighted & BTC-Only", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("synthetic_portfolio_comparison.png")
plt.show()
print("\nSaved  performance chart as 'synthetic_portfolio_comparison.png'.")

# === Metrics Calculation ===
def calc_stats(series, freq_per_year=52):
    returns = series.pct_change().dropna()
    cum_return = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(freq_per_year) if returns.std() > 0 else np.nan
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_drawdown = drawdown.min()
    volatility = returns.std() * np.sqrt(freq_per_year)
    return {
        'Cumulative Return': f"{cum_return*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}" if not np.isnan(sharpe) else "nan",
        'Max Drawdown': f"{max_drawdown*100:.2f}%",
        'Volatility': f"{volatility*100:.2f}%"
    }

metrics = {}
for col in ["Optimized_norm","EqualWeighted_norm","BTCOnly_norm"]:
    print(f"\nCalculating stats for {col}...")
    metrics[col] = calc_stats(df[col])

summary = pd.DataFrame(metrics).T
summary.index = ['Optimized', 'EqualWeighted', 'BuyAndHold']
print("\n==== Synthetic Portfolio Backtest Summary ====\n")
print(summary)
summary.to_csv("synthetic_portfolio_backtest_summary.csv")
print("\nSaved summary table as 'synthetic_portfolio_backtest_summary.csv'.")

print("\nFirst and last few normalized values for each strategy:")
print(df[['date','Optimized_norm','EqualWeighted_norm','BTCOnly_norm']].head())
print(df[['date','Optimized_norm','EqualWeighted_norm','BTCOnly_norm']].tail())
