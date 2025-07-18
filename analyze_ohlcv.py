import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ CONFIG
# ------------------------------
INPUT_DIR = './data/structured'
SYMBOLS = ['BTC', 'ETH', 'ADA', 'BNB', 'XRP', 'SOL', 'DOT', 'DOGE', 'MATIC', 'LTC']

# ------------------------------
# 2️⃣ CLEANING FUNCTION
# ------------------------------
def clean_ohlcv(df):
    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Drop rows with invalid time
    df = df.dropna(subset=['time'])

    # Replace zeros with NaN for prices
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].replace(0, pd.NA)

    # Forward fill missing prices, then backfill as needed
    df[price_cols] = df[price_cols].fillna(method='ffill').fillna(method='bfill')

    # Fill volume NaNs with zero
    df['volumefrom'] = df['volumefrom'].fillna(0)
    df['volumeto'] = df['volumeto'].fillna(0)

    return df.reset_index(drop=True)

# ------------------------------
# 3️⃣ MERGE MULTIPLE SYMBOLS
# ------------------------------
def merge_ohlcv_dfs(dfs, symbols):
    merged = None
    for df, sym in zip(dfs, symbols):
        # Rename columns with symbol
        rename_cols = {
            'open': f'open_{sym}',
            'high': f'high_{sym}',
            'low': f'low_{sym}',
            'close': f'close_{sym}',
            'volumefrom': f'volumefrom_{sym}',
            'volumeto': f'volumeto_{sym}',
        }
        df_renamed = df.rename(columns=rename_cols)[['time'] + list(rename_cols.values())]
        if merged is None:
            merged = df_renamed
        else:
            merged = pd.merge(merged, df_renamed, on='time', how='outer')
    merged = merged.sort_values('time').reset_index(drop=True)
    return merged

# ------------------------------
# 4️⃣ HANDLE MISSING OR ZERO ROWS
# ------------------------------
def filter_empty_rows(df, symbols):
    keep_mask = pd.Series([False] * len(df))
    for sym in symbols:
        price_cols = [f'open_{sym}', f'high_{sym}', f'low_{sym}', f'close_{sym}']
        # Keep rows where at least one price column is not null or zero
        keep_mask |= df[price_cols].notnull().any(axis=1) & ~(df[price_cols] == 0).all(axis=1)
    df = df[keep_mask]
    return df.reset_index(drop=True)

# ------------------------------
# 5️⃣ PLOT PRICE + VOLUME FOR SYMBOL
# ------------------------------
def plot_symbol(df, sym):
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df[f'close_{sym}'], label=f'{sym} Close')
    plt.title(f'{sym} Close Price')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.bar(df['time'], df[f'volumefrom_{sym}'], label=f'{sym} Volume', color='orange')
    plt.title(f'{sym} Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# ------------------------------
# 6️⃣ MAIN SCRIPT
# ------------------------------
if __name__ == '__main__':
    all_cleaned = []
    for sym in SYMBOLS:
        path = os.path.join(INPUT_DIR, f'{sym}_ohlcv.csv')
        if not os.path.exists(path):
            print(f'[!] File not found: {path}')
            continue
        df = pd.read_csv(path)
        df_clean = clean_ohlcv(df)
        all_cleaned.append(df_clean)
        print(f'[✔] Cleaned: {sym}  —  {df_clean.shape[0]} rows')

    merged = merge_ohlcv_dfs(all_cleaned, SYMBOLS)
    merged = filter_empty_rows(merged, SYMBOLS)

    print(f'[✓] Merged shape: {merged.shape}')
    print(merged.head())

    # Save cleaned + merged version
    merged.to_csv('./data/structured/merged_ohlcv.csv', index=False)
    print('[💾] Saved merged file: merged_ohlcv.csv')

    # Plot example
    plot_symbol(merged, 'BTC')
