import pandas as pd

# 1️⃣ Load merged file
merged = pd.read_csv('./data/merged/news_prices_joined.csv')

# 2️⃣ Basic info
print("=== Data Overview ===")
print(merged.info())
print(merged.head(10))

# 3️⃣ Check for missing OHLCV data (should only be missing if news date doesn't match OHLCV date)
ohlcv_cols = [col for col in merged.columns if col.startswith('open_') or col.startswith('close_') or col.startswith('volume')]
missing_any = merged[ohlcv_cols].isnull().any(axis=1)
print(f"\nRows missing *any* OHLCV data: {missing_any.sum()} / {len(merged)} ({100*missing_any.mean():.1f}%)")

# 4️⃣ Count articles per date
if 'date' in merged.columns:
    print("\n=== Articles per Date ===")
    print(merged['date'].value_counts().sort_index().tail(10))

# 5️⃣ Example: Show articles where price data is missing (if any)
missing_rows = merged[missing_any]
if not missing_rows.empty:
    print("\n=== Sample rows with missing OHLCV data ===")
    print(missing_rows[['date', 'title'] + ohlcv_cols].head())
else:
    print("\nAll articles successfully matched to OHLCV data.")

# 6️⃣ Check date range alignment
if 'date' in merged.columns:
    print(f"\nEarliest article date: {merged['date'].min()}")
    print(f"Latest article date: {merged['date'].max()}")

# 7️⃣ Show unique symbols if available
symbol_cols = [col.replace('open_', '') for col in merged.columns if col.startswith('open_')]
print(f"\nSymbols present in merged data: {symbol_cols}")

# 8️⃣ Quick summary stats for any price column
sample_price_col = next((c for c in merged.columns if c.startswith('close_')), None)
if sample_price_col:
    print(f"\n{sample_price_col} summary:")
    print(merged[sample_price_col].describe())
