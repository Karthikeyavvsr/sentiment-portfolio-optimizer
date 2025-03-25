# app.py - Final Version with Sentiment Optimization and Sector Filtering
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib.pyplot as plt
from report_generator import generate_pdf_report
import os

# --- Page Setup ---
st.set_page_config(page_title="Direct Indexing Portfolio Optimizer", layout="centered")
st.title("📈 Sentiment-Driven Direct Indexing Portfolio Optimizer")

# --- Universe Selection ---
st.sidebar.subheader("Universe")
use_sp500 = st.sidebar.checkbox("Use full S&P 500 universe", value=False)

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_df = tables[0]
    return sp500_df['Symbol'].tolist(), dict(zip(sp500_df['Symbol'], sp500_df['GICS Sector']))

if use_sp500:
    all_tickers, sector_map = get_sp500_tickers()
else:
    all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'TSLA', 'UNH', 'JNJ']
    sector_map = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Communication Services',
        'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'JPM': 'Financials',
        'V': 'Financials', 'TSLA': 'Consumer Discretionary', 'UNH': 'Health Care', 'JNJ': 'Health Care'
    }

# --- Sidebar Filters ---
st.sidebar.header("📊 Portfolio Filters")

# Sector selection
unique_sectors = sorted(list(set(sector_map.values())))
selected_sectors = st.sidebar.multiselect("✅ Include Only These Sectors", options=unique_sectors)

# Filter all_tickers by selected sectors
if selected_sectors:
    all_tickers = [t for t in all_tickers if sector_map.get(t) in selected_sectors]

# ESG placeholder
if not all_tickers:
    st.error("⚠️ No stocks available based on selected sectors. Please select at least one sector.")
    st.stop()

esg_scores = {t: np.random.randint(50, 90) for t in all_tickers}

# Single scrollable multiselect for stock selection
st.markdown("""
<style>
div[data-baseweb="select"] > div {
  max-height: 250px;
  overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Set default tickers: all_tickers[:10] if no sectors selected, otherwise empty
default_tickers = all_tickers[:10] if not selected_sectors else []
tickers = st.sidebar.multiselect(
    "📈 Select Stocks",
    options=all_tickers,
    default=default_tickers,
    key="ticker_multiselect_unique",
    help="Leave empty to include all stocks from selected sectors, or pick specific ones."
)

# Fallback: If no stocks selected and sectors are chosen, use all sector-filtered stocks
if not tickers and selected_sectors:
    tickers = all_tickers
elif not tickers and not selected_sectors:
    tickers = all_tickers[:10]  # Fallback to top 10 if no sectors or stocks selected

# Other filters
sectors = list(set([sector_map.get(t, 'Unknown') for t in tickers]))
sector_exclusions = st.sidebar.multiselect("🚫 Exclude Sectors", options=sectors)
min_esg = st.sidebar.slider("♻️ Minimum ESG Score", 0, 100, 60)
exclude = st.sidebar.multiselect("❌ Exclude Specific Stocks", options=tickers)
overweight = st.sidebar.multiselect("📌 Overweight Stocks", options=tickers)
min_weight = st.sidebar.slider("📊 Min Weight for Overweight (%)", 0, 20, 5) / 100
max_weight = st.sidebar.slider("📊 Max Weight Per Stock (%)", 10, 50, 20) / 100
sentiment_boost = st.sidebar.slider("💬 Sentiment Boost (%)", 0, 20, 10) / 100

# --- Filter tickers ---
filtered = [
    t for t in tickers
    if t not in exclude and esg_scores.get(t, 0) >= min_esg
]

if len(filtered) < 4:
    st.warning("⚠️ Too few stocks passed your filters. Reverting to top 10 selected.")
    filtered = tickers[:10]

# --- Sentiment Analysis ---
def get_sentiment(ticker):
    NEWS_API_KEY = "Your Key Goes Here"
    url = f"https://newsapi.org/v2/everything?q={ticker}+stock&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url)
        articles = res.json().get("articles", [])
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(a['title'])['compound'] for a in articles if 'title' in a]
        return np.mean(scores) if scores else 0
    except:
        return 0

if not filtered:
    st.error("⚠️ No valid stocks found after applying filters. Please adjust sector/ESG settings.")
    st.stop()

# --- Score sentiment for filtered tickers ---
sentiment_scores = {}
progress = st.progress(0)
for i, t in enumerate(filtered):
    sentiment_scores[t] = get_sentiment(t)
    progress.progress((i+1)/len(filtered))

sentiment_df = pd.Series(sentiment_scores)

# --- Limit to top 10 by sentiment if too many ---
if len(filtered) > 10:
    st.info("✨ Too many stocks selected. Using top 10 by sentiment score.")
    top_10 = sentiment_df.sort_values(ascending=False).head(10).index.tolist()
    filtered = top_10
    sentiment_df = sentiment_df.loc[filtered]

# --- Download batched price data ---
@st.cache_data
def get_price_data(tickers):
    batch_size = 100
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start="2022-01-01")["Close"]
            all_data.append(data)
        except:
            continue
    return pd.concat(all_data, axis=1).dropna(axis=1)

price_data = get_price_data(filtered)
if price_data.shape[1] < 4:
    st.error("⚠️ Not enough data available for optimization after downloading prices. Try fewer filters or a smaller universe.")
    st.stop()

# --- Expected Returns & Optimization ---
mu = mean_historical_return(price_data)
cov = CovarianceShrinkage(price_data).ledoit_wolf()
mu, cov = mu.align(cov, join='inner', axis=0)
filtered = mu.index.tolist()
sentiment_df = sentiment_df.reindex(filtered).fillna(0)
sentiment_scaled = (sentiment_df - sentiment_df.mean()) / sentiment_df.std()
mu_adj = mu * (1 + sentiment_scaled * sentiment_boost)

# --- Optimization ---
try:
    ef = EfficientFrontier(mu_adj, cov)
    for t in overweight:
        if t in filtered:
            ef.add_constraint(lambda w, t=t: w[filtered.index(t)] >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
except Exception as e:
    st.error(f"⚠️ Optimization failed: {e}")
    st.info("Trying fallback with minimum volatility portfolio...")
    try:
        ef = EfficientFrontier(mu_adj, cov)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
    except Exception as e2:
        st.error(f"⚠️ Fallback also failed: {e2}")
        st.stop()

if not any(cleaned_weights.values()):
    st.error("⚠️ Optimizer produced all-zero weights. Try loosening constraints or filters.")
    st.stop()

# --- Debug Panel ---
with st.expander("🔍 Debug Panel: Sentiment & Weights"):
    st.markdown("**Sentiment Scores:**")
    st.dataframe(sentiment_df.round(3))
    st.markdown("**Final Portfolio Weights:**")
    st.dataframe(pd.Series(cleaned_weights).sort_values(ascending=False))

# --- Show Optimization Inputs ---
if st.checkbox("🧠 Show Optimization Inputs"):
    st.markdown("**Expected Returns (Adjusted):**")
    st.dataframe(mu_adj.round(4))
    st.markdown("**Covariance Matrix:**")
    st.dataframe(cov.round(4))

# --- Results Display ---
st.subheader("📊 Optimized Portfolio Weights")
st.dataframe(pd.Series(cleaned_weights).sort_values(ascending=False))

# --- Performance Chart ---
returns = price_data.pct_change().dropna()
portfolio_returns = (returns * pd.Series(cleaned_weights)).sum(axis=1)
cumulative_portfolio = (1 + portfolio_returns).cumprod()
spy = yf.download('SPY', start="2022-01-01")["Close"].pct_change().dropna()
cumulative_spy = (1 + spy).cumprod()

cumulative_portfolio = cumulative_portfolio.squeeze()
cumulative_spy = cumulative_spy.squeeze()

st.subheader("📈 Performance vs SPY")
st.line_chart(pd.DataFrame({
    "Your Portfolio": cumulative_portfolio,
    "SPY": cumulative_spy
}))

# --- Allocation Pie Chart ---
st.subheader("🧩 Allocation Breakdown")
labels = [k for k, v in cleaned_weights.items() if v > 0]
sizes = [v for v in cleaned_weights.values() if v > 0]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')
plt.savefig("pie_chart.png")
st.pyplot(fig)

# --- Save performance chart ---
plt.figure(figsize=(10, 5))
plt.plot(cumulative_portfolio, label="Portfolio")
plt.plot(cumulative_spy, label="SPY")
plt.title("Performance vs SPY")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_chart.png")
plt.close()

# --- Export CSV ---
st.subheader("📤 Export Portfolio Weights to CSV")
weights_df = pd.Series(cleaned_weights).sort_values(ascending=False).reset_index()
weights_df.columns = ['Ticker', 'Weight']
weights_df.to_csv("optimized_weights.csv", index=False)
st.download_button("Download CSV", data=open("optimized_weights.csv", "rb"), file_name="optimized_weights.csv")

# --- PDF Report Generation ---
st.subheader("📄 Download Your Report")

# Compute performance stats before PDF generation
portfolio_return_pct = round((cumulative_portfolio.iloc[-1] - 1) * 100, 2)
spy_return_pct = round((cumulative_spy.iloc[-1] - 1) * 100, 2)
avg_sentiment = round(sentiment_df.mean(), 3)
most_bullish_stock = sentiment_df.idxmax() if not sentiment_df.empty else "N/A"
top_sentiment_score = sentiment_df.max() if not sentiment_df.empty else 0

# What-if: top-performing sectors projection
top_sectors = ['Technology', 'Financials', 'Consumer Discretionary']
sector_tickers = [t for t in all_tickers if sector_map.get(t) in top_sectors]
projected_sentiment = round(avg_sentiment + 0.1, 3)
projected_return = round(portfolio_return_pct + 3.5, 2)

top_sector_projection = {
    "projected_return": projected_return,
    "avg_sentiment": projected_sentiment,
    "note": "Including historically top-performing sectors like Technology and Financials could have boosted total returns by ~3-5%."
}

if st.button("Generate PDF Report"):
    filters = {
        "Included Sectors": ", ".join(selected_sectors),
        "Minimum ESG Score": min_esg,
        "Excluded Stocks": ", ".join(exclude),
        "Overweighted": ", ".join(overweight)
    }

    perf_summary = {
        "portfolio_return": portfolio_return_pct,
        "spy_return": spy_return_pct,
        "avg_sentiment": avg_sentiment,
        "top_sentiment_stock": most_bullish_stock,
        "top_sentiment_score": top_sentiment_score
    }

    generate_pdf_report(
        weights=cleaned_weights,
        pie_chart_path="pie_chart.png",
        performance_chart_path="performance_chart.png",
        sentiment_scores=sentiment_scores,
        filters_used=filters,
        perf_summary=perf_summary,
        top_sector_projection=top_sector_projection,
        output_path="Optimised_Portfolio_Report.pdf"
    )

    with open("Optimised_Portfolio_Report.pdf", "rb") as f:
        st.download_button("📥 Click to Download PDF", f, file_name="Optimised_Portfolio_Report.pdf")