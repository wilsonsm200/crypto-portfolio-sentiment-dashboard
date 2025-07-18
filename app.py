import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Crypto Portfolio Sentiment Dashboard",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown(
    "<h1>📊 Crypto Portfolio Optimization & Sentiment Dashboard</h1>",
    unsafe_allow_html=True
)

# ---------- DATA LOADERS ----------

@st.cache_data
def load_weights():
    path = "./data/model_input/portfolio_weights.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_portfolio_history():
    path = "./data/backtest/portfolio_value_history.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_sentiment():
    path = "./data/merged/daily_sentiment_index.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_news():
    # You may need to adapt this path if your news file has a different name
    path = "./data/unstructured/news_cleaned_SAMPLE.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ---------- LOAD DATA ----------
weights = load_weights()
port_hist = load_portfolio_history()
sent_idx = load_sentiment()
news = load_news()

# ---------- FILE CHECK ----------
if weights is None:
    st.warning("File not found: ./data/model_input/portfolio_weights.csv")
if port_hist is None:
    st.warning("File not found: ./data/model_input/portfolio_value_history.csv")

# ---------- PORTFOLIO VALUE ----------
st.header("Portfolio Value Over Time")
if port_hist is not None and len(port_hist) > 0:
    port_hist['date'] = pd.to_datetime(port_hist['date'])
    st.line_chart(
        port_hist.set_index('date')[['Optimized', 'EqualWeight', 'BTCOnly']],
        use_container_width=True
    )
else:
    st.info("Portfolio results not available. Run backtest.py to generate results.")

# ---------- PORTFOLIO WEIGHTS ----------
st.header("Latest Portfolio Weights")
if weights is not None:
    last_row = weights.tail(1)
    # Extract asset weights (ignore 'date')
    assets = [col for col in last_row.columns if col.lower() not in ['date', 'time']]
    w_df = pd.DataFrame({
        "Asset": assets,
        "Weight": [last_row[asset].values[0] for asset in assets]
    })
    rebalance_date = last_row['date'].values[0] if 'date' in last_row.columns else "N/A"
    st.write(f"**Rebalance Date:** {rebalance_date}")
    st.dataframe(w_df)
else:
    st.info("Portfolio weights not available. Run portfolio_models.py.")

# ---------- SENTIMENT INDEX ----------
st.header("Latest Daily Sentiment")
if sent_idx is not None and len(sent_idx) > 0:
    last = sent_idx.tail(1).iloc[0]
    st.metric("Mean Sentiment", f"{last['mean_sentiment']:.3f}")
    st.line_chart(sent_idx.set_index('date')['mean_sentiment'])
else:
    st.info("Sentiment index not available.")

# ---------- RECENT NEWS ----------
st.header("Recent News Headlines")
if news is not None and 'published_on' in news.columns:
    news['date'] = pd.to_datetime(news['published_on'], unit='s')
    display_cols = [col for col in ['date', 'title', 'source', 'tags'] if col in news.columns]
    if len(display_cols) > 0:
        st.dataframe(news[display_cols].sort_values('date', ascending=False).head(10))
    else:
        st.write("No columns available to display.")
else:
    st.info("News data not available.")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Crypto Portfolio Optimization & Sentiment Dashboard · Built with Streamlit")
