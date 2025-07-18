# sentiment_pipeline.py

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# ----------------------------------
# 1️⃣ Load merged data
# ----------------------------------
INPUT_PATH = './data/merged/news_prices_joined.csv'
OUTPUT_ARTICLE_PATH = './data/merged/news_prices_with_sentiment.csv'
OUTPUT_AGG_PATH = './data/merged/daily_sentiment_index.csv'

print("[*] Loading merged news+price data...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} articles.")

# ----------------------------------
# 2️⃣ Initialize VADER
# ----------------------------------
analyzer = SentimentIntensityAnalyzer()

# Optional: add/adjust for crypto slang
custom_words = {
    "hodl": 2.0,   # strong positive
    "moon": 2.5,
    "rekt": -2.5,
    "fud": -2.0,
    "pump": 1.5,
    "dump": -1.5,
    "lambo": 1.7,
    "bullish": 2.0,
    "bearish": -2.0,
    "scam": -2.5,
    "rugpull": -2.0,
}
analyzer.lexicon.update(custom_words)

# ----------------------------------
# 3️⃣ Score sentiment (body & title)
# ----------------------------------
def safe_sentiment(text):
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
        return 0.0
    return analyzer.polarity_scores(text)['compound']

print("[*] Running VADER sentiment scoring...")
df['sentiment_title'] = df['clean_title'].apply(safe_sentiment)
df['sentiment_body']  = df['clean_body'].apply(safe_sentiment)
# Optionally, combine title/body (average or weighted)
df['sentiment_avg'] = df[['sentiment_title', 'sentiment_body']].mean(axis=1)

print("[*] Sentiment columns added:")
print(df[['sentiment_title', 'sentiment_body', 'sentiment_avg']].head())

# ----------------------------------
# 4️⃣ Save per-article sentiment
# ----------------------------------
os.makedirs('./data/merged/', exist_ok=True)
df.to_csv(OUTPUT_ARTICLE_PATH, index=False)
print(f"[💾] Saved with sentiment: {OUTPUT_ARTICLE_PATH}")

# ----------------------------------
# 5️⃣ Aggregate: daily/coin sentiment index
# ----------------------------------
if 'date' in df.columns:
    group_cols = ['date']  # optionally add 'categories' or coin field if you have it
    daily_sentiment = df.groupby(group_cols).agg(
        n_articles=('id', 'count'),
        mean_sentiment=('sentiment_avg', 'mean'),
        mean_title_sentiment=('sentiment_title', 'mean'),
        mean_body_sentiment=('sentiment_body', 'mean'),
    ).reset_index()

    daily_sentiment.to_csv(OUTPUT_AGG_PATH, index=False)
    print(f"[💾] Saved aggregated sentiment index: {OUTPUT_AGG_PATH}")
    print(daily_sentiment.tail())

# ----------------------------------
# 6️⃣ Quick EDA: plot daily sentiment
# ----------------------------------
try:
    import matplotlib.pyplot as plt
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment.set_index('date')['mean_sentiment'].plot(
        marker='o', title='Daily Average News Sentiment', figsize=(10,4))
    plt.xlabel('Date')
    plt.ylabel('Mean Sentiment')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"(Plotting skipped: {e})")

print("[✓] Sentiment analysis pipeline complete.")
