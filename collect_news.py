import requests
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# =============================
# 1️⃣ Load API key
# =============================
load_dotenv()
API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')

# =============================
# 2️⃣ Directories & URL
# =============================
OUTPUT_DIR = './data/unstructured/news_raw/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = 'https://min-api.cryptocompare.com/data/v2/news/'

# =============================
# 3️⃣ Base request params
# =============================
PARAMS = {
    'lang': 'EN',
    'api_key': API_KEY,
    'sortOrder': 'latest',
    'limit': 100   # maximum allowed per page
}

# =============================
# 4️⃣ Fetch one batch of news
# =============================
def fetch_news_batch(last_timestamp=None):
    params = PARAMS.copy()
    if last_timestamp:
        params['lTs'] = last_timestamp

    print(f"[+] Requesting news batch, lTs={last_timestamp} ...")

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data['Type'] == 100:
            return data['Data']
        else:
            print(f"[!] API Error: {data.get('Message', 'No message')}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"[!] Request failed: {e}")
        return []

# =============================
# 5️⃣ Save articles by date
# =============================
def save_news_by_date(articles):
    grouped = {}
    for article in articles:
        pub_date = datetime.utcfromtimestamp(article['published_on']).strftime('%Y-%m-%d')
        grouped.setdefault(pub_date, []).append(article)

    for date_str, articles_list in grouped.items():
        filename = os.path.join(OUTPUT_DIR, f'news_{date_str}.json')

        # Load existing
        existing_ids = set()
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    existing_articles = json.load(f)
                    existing_ids = {a['id'] for a in existing_articles}
                except json.JSONDecodeError:
                    existing_articles = []
        else:
            existing_articles = []

        # Deduplicate
        new_articles = [a for a in articles_list if a['id'] not in existing_ids]

        if new_articles:
            all_articles = existing_articles + new_articles
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_articles, f, indent=2, ensure_ascii=False)
            print(f"[✔] Saved {len(new_articles)} new articles to {filename}")
        else:
            print(f"[i] No new articles to save for {date_str}")

# =============================
# 6️⃣ Main loop — keep going!
# =============================
def main():
    last_timestamp = None
    pages_fetched = 0

    while True:
        articles = fetch_news_batch(last_timestamp)
        if not articles:
            print("[✓] No more articles fetched — exiting.")
            break

        save_news_by_date(articles)

        # Update lTs to oldest article in batch to page backward
        oldest_timestamp = min([a['published_on'] for a in articles])
        if oldest_timestamp == last_timestamp:
            print("[✓] Reached end of available news — no further pagination possible.")
            break

        last_timestamp = oldest_timestamp
        pages_fetched += 1

        print(f"[i] Completed page {pages_fetched}.")
        time.sleep(5)  # Pause for API rate limits

    print(f"[✓] Finished fetching all available pages: {pages_fetched} pages done.")

if __name__ == "__main__":
    main()
