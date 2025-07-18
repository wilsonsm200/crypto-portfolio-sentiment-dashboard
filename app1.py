import streamlit as st
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# ========== Gemini setup ==========
if os.path.exists('.env'):
    load_dotenv('.env')
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
else:
    GEMINI_API_KEY = None

# ========== App Title ==========
st.set_page_config(page_title="Crypto Portfolio Sentiment Dashboard", layout="wide")
st.markdown("""
# <img src="https://img.icons8.com/color/48/000000/combo-chart--v2.png" width="36"/> Crypto Portfolio Optimization & Sentiment Dashboard
""", unsafe_allow_html=True)

# ========== 1. Portfolio Value Over Time ==========
st.subheader("Portfolio Value Over Time")
port_hist = None
hist_path = './data/backtest/portfolio_value_history.csv'

if os.path.exists(hist_path):
    port_hist = pd.read_csv(hist_path)
    port_hist['date'] = pd.to_datetime(port_hist['date'])

    # --- Robust column detection ---
    col_sets = [
        ['btc_only', 'equal_weight', 'opt_portfolio'],
        ['BTCOnly', 'EqualWeight', 'Optimized'],
        ['btc_only', 'equal_weight', 'optimized'],
    ]
    lower_cols = [c.lower() for c in port_hist.columns]
    found = False
    for colset in col_sets:
        if all(col in lower_cols for col in [c.lower() for c in colset]):
            real_cols = []
            for c in colset:
                real_cols.append([cc for cc in port_hist.columns if cc.lower() == c.lower()][0])
            found = True
            break
    if found:
        st.line_chart(
            port_hist.set_index('date')[real_cols],
            use_container_width=True
        )
    else:
        st.warning(f"Portfolio CSV columns found: {port_hist.columns.tolist()} -- Please check your backtest output column names!")
        st.write(port_hist.head())
else:
    st.info("Portfolio results not available. Run backtest.py to generate results.")

# ========== 2. Latest Portfolio Weights ==========
st.subheader("Latest Portfolio Weights")
weights_path = './data/model_input/portfolio_weights.csv'
if os.path.exists(weights_path):
    weights = pd.read_csv(weights_path)
    weights['date'] = pd.to_datetime(weights['date'])
    last_row = weights.iloc[-1]
    latest_date = last_row['date']
    st.write(f"**Rebalance Date:** {latest_date.date()}")
    asset_cols = [c for c in weights.columns if c != 'date']
    latest_weights = pd.DataFrame({
        'Asset': asset_cols,
        'Weight': [last_row[c] for c in asset_cols]
    })
    st.dataframe(latest_weights, hide_index=True)
else:
    st.info("No portfolio weights found.")

# ========== 3. Latest Daily Sentiment ==========
st.subheader("Latest Daily Sentiment")
feat_path = './data/model_input/features_with_sentiment.csv'
if os.path.exists(feat_path):
    feats = pd.read_csv(feat_path)
    feats['date'] = pd.to_datetime(feats['date'])
    last_sent = feats.iloc[-1]
    st.metric("Mean Sentiment", f"{last_sent['mean_sentiment']:.3f}")
    st.line_chart(feats.set_index('date')['mean_sentiment'])
else:
    st.info("No features/sentiment data found.")

# ========== 4. Recent News Headlines ==========
st.subheader("Recent News Headlines")
news_path = './data/merged/news_prices_with_sentiment.csv'
if os.path.exists(news_path):
    news = pd.read_csv(news_path)
    if 'date' in news.columns:
        news['date'] = pd.to_datetime(news['date'])
        display_cols = ['date', 'title', 'source', 'tags']
        available_cols = [c for c in display_cols if c in news.columns]
        st.dataframe(news[available_cols].sort_values('date', ascending=False).head(10))
    else:
        st.info("No 'date' column in news file.")
else:
    st.info("News data not found.")

# ========== 5. Gemini Integration for Explanations ==========
if GEMINI_API_KEY:
    st.subheader("Ask Gemini: Portfolio & Sentiment Explanation")
    question = st.text_input(
        "Ask a question about your portfolio or today's sentiment (example: Why did my portfolio rebalance?)"
    )
    if st.button("Ask Gemini"):
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        # Use your latest_weights and last_sent for context
        try:
            prompt = f"Crypto portfolio status: {latest_weights.to_string(index=False)}\n\nLatest sentiment: {last_sent['mean_sentiment']:.3f}\n\nUser question: {question}"
        except Exception:
            prompt = question
        try:
            # Use a supported Gemini model from your available models
            model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
            response = model.generate_content(prompt)
            st.write("**Gemini:**", response.text)
        except Exception as e:
            st.error(f"Gemini error: {e}")

st.markdown("<br><br><sub>Crypto Portfolio Optimization & Sentiment Dashboard · Built with Streamlit</sub>", unsafe_allow_html=True)
