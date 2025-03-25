# 📈 Sentiment-Driven Direct Indexing Portfolio Optimizer

## Description
This project is a web-based application built with Streamlit that optimizes investment portfolios using direct indexing, enhanced by sentiment analysis from news articles. Users can filter stocks by sectors, ESG scores, and other preferences, then optimize their portfolio based on historical returns adjusted by sentiment scores. The app provides visualizations like performance charts and allocation breakdowns, and generates a detailed PDF report summarizing the results.

The combination of sentiment-driven optimization, user-customizable filters, and reporting makes this project a unique showcase for my GitHub portfolio. As my first project here, it demonstrates skills in financial data analysis, web development, and sentiment processing, built as a hobby to explore these concepts.

## Features
- Select stocks from the S&P 500 or a predefined list, with sector and ESG filtering.
- Sentiment analysis of stock-related news using the News API and VADER.
- Portfolio optimization with PyPortfolioOpt, incorporating sentiment boosts.
- Visualizations including performance vs. SPY and allocation pie charts.
- Exportable CSV of optimized weights and a comprehensive PDF report.

## How to Run

1. **Clone the Repository**:
```bash
git clone https://github.com/your-username/sentiment-driven-portfolio-optimizer.git
cd sentiment-driven-portfolio-optimizer
```

2. **Install Dependencies**:
Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

3. **Set Up API Key**:
- Obtain a free API key from [News API](https://newsapi.org/).
- Replace the `NEWS_API_KEY` in `app.py` with your key (or use environment variables for better security).

4. **Run the Application**:
```bash
streamlit run app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501) to use the app.

## Tech Stack
- **Python**: Core programming language
- **Streamlit**: Web application framework for the user interface
- **yfinance**: Fetches historical stock price data
- **PyPortfolioOpt**: Portfolio optimization using Efficient Frontier
- **vaderSentiment**: Sentiment analysis of news titles
- **News API**: Source for stock-related news articles
- **FPDF**: PDF report generation
- **pandas, numpy, matplotlib**: Data manipulation and visualization

## Installation Notes

Create a `requirements.txt` file with the following:
```
streamlit
yfinance
pandas
numpy
matplotlib
PyPortfolioOpt
vaderSentiment
requests
fpdf
```

Then install:
```bash
pip install -r requirements.txt
```

## Acknowledgments
This project uses open-source libraries and APIs, credited in the Tech Stack section. I’ve worked to make this an original implementation, but if you spot anything here that looks inspired by your work, please reach out! I’d love to give a shoutout and note how your efforts have influenced me.

---

Excited to share my first GitHub project! Feel free to explore, open issues, or suggest improvements.