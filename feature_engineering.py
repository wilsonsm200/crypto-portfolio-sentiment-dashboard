import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1️⃣ Load data
# -------------------------------
sent = pd.read_csv('./data/merged/daily_sentiment_index.csv')
ohlcv = pd.read_csv('./data/structured/merged_ohlcv.csv')

# Standardize date fields
sent['date'] = pd.to_datetime(sent['date'])
ohlcv['date'] = pd.to_datetime(ohlcv['time']).dt.normalize()

# -------------------------------
# 2️⃣ Compute daily returns for all coins
# -------------------------------
coins = ['BTC', 'ETH', 'ADA', 'BNB', 'XRP', 'SOL', 'DOT', 'DOGE', 'MATIC', 'LTC']
ohlcv = ohlcv.sort_values('date')
for coin in coins:
    price_col = f'close_{coin}'
    return_col = f'return_{coin}'
    if price_col in ohlcv.columns:
        ohlcv[return_col] = ohlcv[price_col].pct_change()
    else:
        print(f"[!] Warning: {price_col} not in OHLCV data. Skipping return calc.")

# -------------------------------
# 3️⃣ Merge sentiment with OHLCV
# -------------------------------
df = ohlcv.merge(sent, how='left', on='date')

# -------------------------------
# 4️⃣ Compute rolling features for BTC (add for others as needed)
# -------------------------------
df['ret5_BTC'] = df['close_BTC'].pct_change(periods=5)
df['vol10_BTC'] = df['return_BTC'].rolling(window=10).std()

# -------------------------------
# 5️⃣ Visualize: Sentiment vs. BTC Return
# -------------------------------
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(df['date'], df['return_BTC'], label='BTC Daily Return')
plt.ylabel('BTC Return')
plt.legend()
plt.subplot(2,1,2)
plt.plot(df['date'], df['mean_sentiment'], label='Daily News Sentiment', color='orange')
plt.ylabel('Mean Sentiment')
plt.legend()
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# Scatter plot to see correlation
plt.figure(figsize=(6,4))
plt.scatter(df['mean_sentiment'], df['return_BTC'], alpha=0.5)
plt.xlabel('Daily Mean Sentiment')
plt.ylabel('BTC Return')
plt.title('Sentiment vs BTC Daily Return')
plt.show()

# -------------------------------
# 6️⃣ Prepare for Modeling/Backtesting
# -------------------------------
min_features = ['mean_sentiment', 'return_BTC', 'return_ETH', 'return_ADA', 'ret5_BTC', 'vol10_BTC']
df_model = df.dropna(subset=min_features)

# Example: Linear regression (BTC as target)
X = df_model[['mean_sentiment', 'ret5_BTC', 'vol10_BTC']]
y = df_model['return_BTC']
model = LinearRegression().fit(X, y)
print("\nModel coefficients:", dict(zip(X.columns, model.coef_)))
print("Model intercept:", model.intercept_)
print("R2 (fit score):", model.score(X, y))

# -------------------------------
# 7️⃣ Save features for portfolio optimization/backtest
# -------------------------------
os.makedirs('./data/model_input/', exist_ok=True)
df_model.to_csv('./data/model_input/features_with_sentiment.csv', index=False)
print("[💾] Features saved for portfolio modeling: ./data/model_input/features_with_sentiment.csv")

