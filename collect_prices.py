# collect_prices.py

import requests
import pandas as pd
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# =============================
# 1️⃣ Load API key securely
# =============================
load_dotenv()  # Loads .env file in same folder
API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')

# =============================
# 2️⃣ Config settings
# =============================
BASE_URL = 'https://min-api.cryptocompare.com/data/v2/histoday'
CRYPTO_SYMBOLS =  ['BTC', 'ETH', 'ADA', 'BNB', 'XRP', 'SOL', 'DOT', 'DOGE', 'MATIC', 'LTC']  # 
CURRENCY = 'USD'
LIMIT = 2000  # max days per request for free plan 
OUTPUT_DIR = './data/structured/'

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# 3️⃣ Function to fetch OHLCV
# =============================
def fetch_ohlcv(symbol, limit=LIMIT):
    params = {
        'fsym': symbol,
        'tsym': CURRENCY,
        'limit': limit,
        'api_key': API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['symbol'] = symbol
            return df
        else:
            print(f"[!] API Error for {symbol}: {data.get('Message', 'Unknown error')}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"[!] Request failed for {symbol}: {e}")
        return pd.DataFrame()

# =============================
# 4️⃣ Main logic
# =============================
def main():
    for symbol in CRYPTO_SYMBOLS:
        print(f"[+] Fetching daily OHLCV for {symbol} ...")
        df = fetch_ohlcv(symbol)
        if not df.empty:
            filename = os.path.join(OUTPUT_DIR, f"{symbol}_ohlcv.csv")
            df.to_csv(filename, index=False)
            print(f"[✔] Saved {symbol} data to {filename}")
        else:
            print(f"[x] No data saved for {symbol} due to errors or empty response.")
        # Pause to stay within free plan rate limits
        time.sleep(5)

    print("[✓] All symbols processed.")

# =============================
# 5️⃣ Run script
# =============================
if __name__ == '__main__':
    main()
