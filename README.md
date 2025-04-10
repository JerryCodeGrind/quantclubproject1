# Stock Sentiment Analyzer

A Python-based tool that analyzes sentiment for stocks using news from financial sources.

## Features

- Scrapes financial news from multiple sources including FinViz
- Performs sentiment analysis using FinBERT, a financial NLP model
- Calculates average sentiment scores for stocks
- Simple command-line interface

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/JerryCodeGrind/quantclubproject1.git
   cd quantclubproject1
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install beautifulsoup4 requests transformers torch pandas
   ```

## Usage

### FinViz Sentiment Analysis

To analyze a stock using news from FinViz:

```
python Finviz/finviz.py
```

When prompted, enter the stock ticker (e.g., AAPL).

The script will:
1. Fetch recent news articles for the stock from FinViz
2. Display each article with its date and source
3. Calculate and display the average sentiment score

## Project Structure

- `Finviz/finviz.py` - Simplified sentiment analysis using FinViz news
- `main.py` - Multi-source news scraper and sentiment analyzer (optional)

## Requirements

- Python 3.6+
- BeautifulSoup4
- Requests
- Transformers
- PyTorch
- Pandas

## Note

This project is for educational purposes only. Financial decisions should not be based solely on sentiment analysis.
