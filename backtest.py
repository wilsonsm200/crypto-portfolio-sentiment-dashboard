# backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------
# 0. Config
# ------------------------
rebalance_freq = 7  # Use 7 for weekly or 30 for monthly

# ------------------------
# 1. Load features and weights
# ------------------------
df = pd.read_csv('./data/model_input/features_with_sentiment.csv')
weights_df = pd.read_csv('./data/model_input/portfolio_weights.csv')
coins = [col for col in weights_df.columns if col != 'date']
returns_cols = [f'return_{c}' for c in coins]

df['date'] = pd.to_datetime(df['date'])
weights_df['date'] = pd.to_datetime(weights_df['date'])
df = df.set_index('date')

# ------------------------
# 2. Backtest with rebalancing
# ------------------------
portfolio_value = [1.0]
dates = []
benchmark_eq = [1.0]  # Equal-weight portfolio
benchmark_btc = [1.0] # BTC only

slippage = 0.001   # 0.1% per trade (both sides)
fee = 0.001        # 0.1% per trade

for i in range(len(weights_df)):
    d = weights_df['date'].iloc[i]
    print(f"[i] Backtest step {i}, rebalance date: {d}")  # Debug
    w = weights_df[coins].iloc[i].values
    if i == len(weights_df)-1:
        break
    try:
        idx_start = df.index.get_loc(d)
    except KeyError:
        print(f"[!] Date {d} not found in price data. Skipping.")
        continue
    idx_end = idx_start + rebalance_freq
    if idx_end >= len(df):
        break
    window = df.iloc[idx_start:idx_end]
    if window.empty:
        print(f"[!] Window is empty for {d}. Skipping.")
        continue
    period_rets = window[returns_cols].values
    if period_rets.shape[0] == 0:
        print(f"[!] No returns for window {d}. Skipping.")
        continue
    # Portfolio return with weights
    if i == 0:
        prev_w = np.zeros(len(w))
    else:
        prev_w = weights_df[coins].iloc[i-1].values
    turnover = np.sum(np.abs(w - prev_w))
    gross_return = (period_rets @ w).mean()
    net_return = gross_return - (slippage + fee) * turnover
    portfolio_value.append(portfolio_value[-1] * (1 + net_return))
    dates.append(d)

    # Equal-weight benchmark
    ew = np.ones(len(coins)) / len(coins)
    ew_return = (period_rets @ ew).mean()
    benchmark_eq.append(benchmark_eq[-1] * (1 + ew_return))

    # BTC-only
    if 'return_BTC' in window.columns:
        btc_return = window['return_BTC'].mean()
        benchmark_btc.append(benchmark_btc[-1] * (1 + btc_return))
    else:
        benchmark_btc.append(benchmark_btc[-1])

# ------------------------
# 3. Plot results
# ------------------------
if len(portfolio_value) > 1:
    plt.figure(figsize=(10,5))
    plt.plot(dates, portfolio_value[1:], label='Optimized (Sentiment)', linewidth=2)
    plt.plot(dates, benchmark_eq[1:], label='Equal-Weight', linestyle='--')
    plt.plot(dates, benchmark_btc[1:], label='BTC-Only', linestyle=':')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Backtest')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("[!] No portfolio data to plot.")

# ------------------------
# 4. Metrics
# ------------------------
def sharpe_ratio(returns):
    if np.std(returns) == 0:
        return np.nan
    return np.mean(returns) / np.std(returns) * np.sqrt(252/rebalance_freq)

def max_drawdown(values):
    values = np.array(values)
    roll_max = np.maximum.accumulate(values)
    drawdown = (roll_max - values) / roll_max
    return drawdown.max() if len(drawdown) > 0 else 0

if len(portfolio_value) > 1:
    rebal_returns = np.diff(np.log(portfolio_value))
    print(f"Sharpe (Optimized): {sharpe_ratio(rebal_returns):.3f}")
    print(f"Max Drawdown (Optimized): {max_drawdown(portfolio_value):.2%}")

    rebal_returns_eq = np.diff(np.log(benchmark_eq))
    print(f"Sharpe (Equal-Weight): {sharpe_ratio(rebal_returns_eq):.3f}")
    print(f"Max Drawdown (Equal-Weight): {max_drawdown(benchmark_eq):.2%}")

    rebal_returns_btc = np.diff(np.log(benchmark_btc))
    print(f"Sharpe (BTC-Only): {sharpe_ratio(rebal_returns_btc):.3f}")
    print(f"Max Drawdown (BTC-Only): {max_drawdown(benchmark_btc):.2%}")

    # Save results for Streamlit dashboard compatibility
    results = pd.DataFrame({
        'date': dates,
        'Optimized': portfolio_value[1:],
        'EqualWeight': benchmark_eq[1:],
        'BTCOnly': benchmark_btc[1:]
    })
    os.makedirs('./data/backtest/', exist_ok=True)
    results.to_csv('./data/backtest/portfolio_value_history.csv', index=False)
    print("[💾] Backtest results saved to ./data/backtest/portfolio_value_history.csv.")
else:
    print("[!] Not enough data to calculate Sharpe or drawdown.")
