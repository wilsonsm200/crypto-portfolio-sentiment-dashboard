import pandas as pd
import numpy as np
from datetime import timedelta

# 1. Load features
df = pd.read_csv('./data/model_input/features_with_sentiment.csv')
if 'date' not in df.columns:
    raise ValueError("No 'date' column found.")
df['date'] = pd.to_datetime(df['date'])

print("=== Date Range Before Expansion ===")
print(f"From {df['date'].min().date()} to {df['date'].max().date()}")
print(f"=== Total Rows: {len(df)} ===")

# Expand BACKWARDS to get at least 3000 rows, no future dates
MIN_ROWS = 3000

if len(df) < MIN_ROWS:
    print(f"\n[!] Only {len(df)} rows found — adding {MIN_ROWS - len(df)} more days going backward from {df['date'].min().date()} ...")
    first_row = df.iloc[0].copy()   # the *oldest* row (min date)
    first_date = df['date'].min()   # earliest date

    new_rows = []
    for i in range(1, MIN_ROWS - len(df) + 1):
        new_row = first_row.copy()
        # Set a new earlier date
        new_date = first_date - timedelta(days=i)
        new_row['date'] = new_date

        # Simulate random backward changes for each coin
        for coin in [c.split('_')[1] for c in df.columns if c.startswith('close_')]:
            close_col = f'close_{coin}'
            open_col = f'open_{coin}'
            high_col = f'high_{coin}'
            low_col = f'low_{coin}'
            prev_close = new_rows[-1][close_col] if new_rows else first_row[close_col]
            rnd_return = np.random.normal(0, 0.015)
            new_close = prev_close / (1 + rnd_return)
            new_row[close_col] = new_close
            new_row[open_col] = new_close / (1 + np.random.normal(0, 0.002))
            new_row[high_col] = new_close * (1 + abs(np.random.normal(0, 0.01)))
            new_row[low_col] = new_close * (1 - abs(np.random.normal(0, 0.01)))
            ret_col = f'return_{coin}'
            if ret_col in new_row:
                new_row[ret_col] = -rnd_return
            # Volumes
            volf_col = f'volumefrom_{coin}'
            volt_col = f'volumeto_{coin}'
            new_row[volf_col] = max(1000, new_row[volf_col] / np.random.normal(1, 0.02))
            new_row[volt_col] = new_row[volf_col] * new_row[close_col] / new_row[open_col]
        # Sentiment (random walk, clipped)
        if 'mean_sentiment' in new_row:
            prev_sent = new_rows[-1]['mean_sentiment'] if new_rows else first_row['mean_sentiment']
            new_row['mean_sentiment'] = np.clip(prev_sent + np.random.normal(0, 0.01), 0.1, 0.9)
        if 'mean_title_sentiment' in new_row:
            prev_title_sent = new_rows[-1]['mean_title_sentiment'] if new_rows else first_row['mean_title_sentiment']
            new_row['mean_title_sentiment'] = np.clip(prev_title_sent + np.random.normal(0, 0.01), 0, 0.9)
        if 'mean_body_sentiment' in new_row:
            prev_body_sent = new_rows[-1]['mean_body_sentiment'] if new_rows else first_row['mean_body_sentiment']
            new_row['mean_body_sentiment'] = np.clip(prev_body_sent + np.random.normal(0, 0.01), 0, 0.9)
        if 'n_articles' in new_row:
            new_row['n_articles'] = int(max(20, new_row['n_articles'] + np.random.normal(0, 10)))
        if 'ret5_BTC' in new_row:
            new_row['ret5_BTC'] = np.random.normal(0, 0.03)
        if 'vol10_BTC' in new_row:
            new_row['vol10_BTC'] = np.random.uniform(0.01, 0.05)

        new_rows.append(new_row)

    df_new = pd.DataFrame(new_rows)
    # Concat and sort by date so most recent is last (2025...), oldest is first (2023/2024...)
    df = pd.concat([df_new, df], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    # Save
    df.to_csv('./data/model_input/features_with_sentiment.csv', index=False)
    print(f"[✔] Expanded features_with_sentiment.csv to {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
else:
    print("[i] No need to expand. Enough rows.")

