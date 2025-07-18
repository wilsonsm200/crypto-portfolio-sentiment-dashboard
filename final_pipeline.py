import os
import glob
import json
import re
import hashlib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# --------------- Config -----------------
UNSTRUCTURED_DIR = './data/unstructured'
STRUCTURED_DIR = './data/structured'
OUTPUT_DIR = './data/final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------- Helpers -----------------

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = BeautifulSoup(text, 'html.parser').get_text()  # remove HTML tags
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text)  # keep alphanum + space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_and_clean_unstructured():
    files = glob.glob(os.path.join(UNSTRUCTURED_DIR, '*.json'))
    print(f"[+] Found {len(files)} raw JSON files.")
    articles = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            for line_num, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    articles.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[!] Skipped bad line {line_num} in {os.path.basename(f)}: {e}")
                    continue
    df = pd.DataFrame(articles)
    print(f"[+] Raw articles loaded: {df.shape[0]}")

    # Clean text columns
    df['clean_title'] = df['title'].astype(str).apply(clean_text)
    df['clean_body'] = df['body'].astype(str).apply(clean_text)

    # Deduplicate based on hashes of cleaned title + body
    df['title_hash'] = df['clean_title'].apply(hash_text)
    df['body_hash'] = df['clean_body'].apply(hash_text)
    df.drop_duplicates(subset=['title_hash', 'body_hash'], inplace=True)
    print(f"[✓] After deduplication: {df.shape[0]}")

    # Convert published_on (unix timestamp) to datetime.date
    df['published_date'] = pd.to_datetime(df['published_on'], unit='s').dt.date

    return df

def load_structured_data():
    # Load all OHLCV csv files in structured folder and concat them with symbol column
    files = glob.glob(os.path.join(STRUCTURED_DIR, '*_ohlcv.csv'))
    dfs = []
    for f in files:
        symbol = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
        df['symbol'] = symbol
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[+] Loaded structured OHLCV data: {full_df.shape}")
    return full_df

def add_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()

    # Add custom crypto lexicon
    new_words = {
        "hodl": 2.0,
        "fud": -2.0,
        "moon": 1.5,
        "rekt": -2.0,
        "bullish": 2.0,
        "bearish": -2.0,
        "rekt": -2.5,
        "pump": 1.5,
        "dump": -1.5,
        "whale": 0.5,
        "dip": -1.0,
    }
    analyzer.lexicon.update(new_words)

    df['sentiment'] = df['clean_body'].apply(analyzer.polarity_scores)
    df['neg'] = df['sentiment'].apply(lambda x: x['neg'])
    df['neu'] = df['sentiment'].apply(lambda x: x['neu'])
    df['pos'] = df['sentiment'].apply(lambda x: x['pos'])
    df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
    df.drop(columns=['sentiment'], inplace=True)

    print("[✓] Added sentiment scores.")
    return df

def aggregate_daily_sentiment(df):
    # Aggregate daily sentiment scores by published_date
    agg = df.groupby('published_date').agg({
        'neg': 'mean',
        'neu': 'mean',
        'pos': 'mean',
        'compound': 'mean',
        'clean_body': 'count'
    }).rename(columns={'clean_body': 'article_count'}).reset_index()
    print(f"[+] Aggregated daily sentiment: {agg.shape}")
    return agg

def engineer_factors(df):
    df = df.sort_values('time')
    # Calculate returns
    df['return'] = df.groupby('symbol')['close'].pct_change()

    # Rolling windows: 5d and 20d momentum and volatility
    df['mom_5'] = df.groupby('symbol')['return'].rolling(window=5).sum().reset_index(0,drop=True)
    df['mom_20'] = df.groupby('symbol')['return'].rolling(window=20).sum().reset_index(0,drop=True)
    df['vol_5'] = df.groupby('symbol')['return'].rolling(window=5).std().reset_index(0,drop=True)
    df['vol_20'] = df.groupby('symbol')['return'].rolling(window=20).std().reset_index(0,drop=True)

    # Value at Risk (VaR) 5% quantile rolling 20 days
    df['VaR_5'] = df.groupby('symbol')['return'].rolling(window=20).quantile(0.05).reset_index(0,drop=True)

    # Max drawdown function
    def max_drawdown(series):
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        return drawdown.min()

    # Compute max drawdown over rolling 20 days
    df['max_drawdown'] = df.groupby('symbol')['close'].rolling(window=20).apply(max_drawdown, raw=True).reset_index(0,drop=True)

    print("[✓] Factor engineering completed.")
    return df

def align_and_save(news_df, ohlcv_df, daily_sentiment_df):
    # Join daily sentiment with OHLCV by date (time) and symbol (we can merge news sentiment on date)
    # Since news_df is aggregated by date only, merge on dates only

    # First merge OHLCV with daily sentiment (by time and published_date)
    merged = ohlcv_df.merge(daily_sentiment_df, left_on='time', right_on='published_date', how='left')
    merged.drop(columns=['published_date'], inplace=True)

    # Optional: fill NaN sentiment scores with 0 or forward fill
    merged[['neg','neu','pos','compound','article_count']] = merged[['neg','neu','pos','compound','article_count']].fillna(0)

    # Save final merged file
    out_path = os.path.join(OUTPUT_DIR, 'final_factor_matrix.csv')
    merged.to_csv(out_path, index=False)
    print(f"[💾] Saved final factor matrix: {out_path}")

    return merged

def main():
    print("[>] Starting full pipeline...")

    # 1. Load and clean unstructured news data
    news_df = load_and_clean_unstructured()

    # 2. Sentiment analysis
    news_df = add_sentiment(news_df)

    # 3. Aggregate daily sentiment index
    daily_sentiment = aggregate_daily_sentiment(news_df)

    # 4. Load structured OHLCV data
    ohlcv_df = load_structured_data()

    # 5. Factor engineering on OHLCV data
    ohlcv_df = engineer_factors(ohlcv_df)

    # 6. Align data and save final output
    final_df = align_and_save(news_df, ohlcv_df, daily_sentiment)

    print("[✓] Pipeline complete.")

if __name__ == '__main__':
    main()
