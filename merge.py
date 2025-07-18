import pandas as pd

# ---------------------------------
# 1️⃣ Load your cleaned files
# ---------------------------------
ohlcv = pd.read_csv('./data/structured/merged_ohlcv.csv')
news = pd.read_csv('./data/unstructured/news_cleaned_SAMPLE.csv')

# ---------------------------------
# 2️⃣ Prepare OHLCV time key
# ---------------------------------
ohlcv['time'] = pd.to_datetime(ohlcv['time']).dt.normalize()  # make sure date-only

# ---------------------------------
# 3️⃣ Prepare news date key
# ---------------------------------
news['date'] = pd.to_datetime(news['published_on'], unit='s').dt.normalize()

# ---------------------------------
# 4️⃣ Merge: inner or left join
# ---------------------------------
merged = news.merge(
    ohlcv,
    how='left',
    left_on='date',
    right_on='time'
)

print('[✓] Merged shape:', merged.shape)
print(merged.head())

# ---------------------------------
# 5️⃣ Save result
# ---------------------------------
merged.to_csv('./data/merged/news_prices_joined.csv', index=False)
print('[💾] Saved joined file: news_prices_joined.csv')
