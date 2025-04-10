from bs4 import BeautifulSoup
import requests
import pandas as pd
from transformers import pipeline
import os
import re
import time
from typing import List, Dict
from urllib.parse import quote_plus

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT}

# Initialize the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ===== NEWS SCRAPING FUNCTIONS =====

def get_news(ticker: str, company_name: str, max_articles: int = 5) -> List[Dict[str, str]]:
    """
    Get news articles from multiple sources.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        max_articles: Maximum number of articles to return
        
    Returns:
        List of article dictionaries with title, description, url, and source
    """
    # Try different sources in sequence, starting with the most reliable
    sources = [
        get_google_news,
        get_finviz_news,
        get_yahoo_finance_news,
        get_benzinga_news
    ]
    
    all_articles = []
    
    for source_func in sources:
        print(f"Trying to get news from {source_func.__name__}")
        articles = source_func(ticker, company_name)
        if articles:
            all_articles.extend(articles)
        
        # Stop if we have enough articles
        if len(all_articles) >= max_articles:
            break
            
        # Be polite to servers
        time.sleep(1)
    
    # Return only the requested number of articles
    return all_articles[:max_articles]

def get_google_news(ticker: str, company_name: str) -> List[Dict[str, str]]:
    """Get news from Google News search"""
    # Construct search query with both ticker and company name for better results
    query = f"{ticker} {company_name} stock news"
    encoded_query = quote_plus(query)
    url = f"https://www.google.com/search?q={encoded_query}&tbm=nws"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = []
        # Find news article divs
        news_divs = soup.find_all('div', class_='SoaBEf')
        
        for div in news_divs[:5]:
            # Extract title and URL
            title_elem = div.find('div', class_='mCBkyc')
            link_elem = div.find('a')
            
            # Extract description
            desc_elem = div.find('div', class_='GI74Re')
            
            if title_elem and link_elem:
                title = title_elem.text.strip()
                url = link_elem.get('href', '')
                
                # Clean up Google redirect URL
                if url.startswith('/url?'):
                    url = url.split('&url=')[1].split('&')[0]
                
                # Get description or use title as fallback
                description = desc_elem.text.strip() if desc_elem else title
                
                articles.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'source': 'Google News'
                })
        
        return articles
        
    except Exception:
        return []

def get_finviz_news(ticker: str, company_name: str) -> List[Dict[str, str]]:
    """Get news from FinViz (great financial news source)"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = []
        
        # FinViz news table has a specific structure
        news_table = soup.find('table', class_='fullview-news-outer')
        if not news_table:
            return []
            
        rows = news_table.find_all('tr')
        
        for row in rows[:5]:
            # Each row has a td for date and a td for the news
            cells = row.find_all('td')
            if len(cells) >= 2:
                link_elem = cells[1].find('a')
                if link_elem:
                    title = link_elem.text.strip()
                    url = link_elem.get('href', '')
                    source = cells[1].find('span').text.strip() if cells[1].find('span') else "FinViz"
                    
                    # FinViz doesn't provide descriptions, use title
                    description = title
                    
                    articles.append({
                        'title': title,
                        'description': description,
                        'url': url,
                        'source': source
                    })
        
        return articles
        
    except Exception:
        return []

def get_yahoo_finance_news(ticker: str, company_name: str) -> List[Dict[str, str]]:
    """Get news from Yahoo Finance"""
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = []
        # Yahoo Finance news articles
        news_items = soup.find_all('div', {'class': 'js-stream-content'})
        
        for item in news_items[:5]:
            title_link = item.find('a', {'data-test': 'title'})
            if title_link:
                title = title_link.text.strip()
                url = 'https://finance.yahoo.com' + title_link.get('href', '') if title_link.get('href', '').startswith('/') else title_link.get('href', '')
                
                # Get description or summary
                summary = item.find('p', {'class': 'Fz(14px)'})
                description = summary.text.strip() if summary else title
                
                # Get source
                source_div = item.find('div', {'class': 'C(#959595)'})
                source = source_div.text.strip().split('·')[0].strip() if source_div else "Yahoo Finance"
                
                articles.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'source': source
                })
        
        return articles
        
    except Exception:
        return []

def get_benzinga_news(ticker: str, company_name: str) -> List[Dict[str, str]]:
    """Get news from Benzinga"""
    url = f"https://www.benzinga.com/stock/{ticker}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = []
        # Benzinga news articles
        news_items = soup.find_all('div', {'class': 'news-right'})
        
        for item in news_items[:5]:
            title_elem = item.find('div', {'class': 'title'})
            if title_elem and title_elem.find('a'):
                title = title_elem.find('a').text.strip()
                url = title_elem.find('a').get('href', '')
                
                # Make URL absolute if it's relative
                if url.startswith('/'):
                    url = f"https://www.benzinga.com{url}"
                
                # Get description
                desc_elem = item.find('div', {'class': 'summary'})
                description = desc_elem.text.strip() if desc_elem else title
                
                articles.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'source': 'Benzinga'
                })
        
        return articles
        
    except Exception:
        return []

# ===== STOCK AND SENTIMENT ANALYSIS FUNCTIONS =====

def get_top_gainers(limit=10):
    """Get the top gaining stocks from TradingView"""
    url = "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    data = []
    all_gainers = soup.find_all('tr', class_='row-RdUXZpkv listRow')
    
    for stock in all_gainers[:limit]:
        # Extract stock information
        change_cell = stock.find('td', class_='cell-RLhfr_y4 right-RLhfr_y4')
        name_cell = stock.find('sup', class_='apply-common-tooltip tickerDescription-GrtoTeat')
        ticker_cell = stock.find('a', class_='apply-common-tooltip tickerName-GrtoTeat')
        
        if change_cell and name_cell:
            # Create stock data dictionary
            stock_data = {
                'Change': change_cell.text.strip(),
                'Name': name_cell.text.strip(),
                'Ticker': ticker_cell.text.strip() if ticker_cell else ''
            }
            
            # Extract ticker from name if not found
            if not stock_data['Ticker']:
                stock_data['Ticker'] = extract_ticker_from_name(stock_data['Name'])
                
            data.append(stock_data)
            
    return pd.DataFrame(data)

def extract_ticker_from_name(name):
    """Extract a likely ticker symbol from a company name"""
    # Try to find ticker in parentheses
    match = re.search(r'\(([A-Z]+)\)', name)
    if match:
        return match.group(1)
    
    # Create abbreviation from capital letters
    words = name.split()
    if len(words) >= 2:
        return ''.join(word[0] for word in words if word[0].isupper())[:4]
    
    # Fallback
    return name[:4].upper()

def analyze_sentiment(text):
    """Analyze the sentiment of a piece of text"""
    result = sentiment_model(text)[0]
    return {
        'label': result['label'],
        'score': result['score']
    }

def get_news_sentiment(ticker, company_name, max_articles=3):
    """Get news articles and analyze their sentiment"""
    # Get news for the stock
    articles = get_news(ticker, company_name, max_articles)
    
    # Analyze sentiment for each article
    sentiments = []
    for article in articles:
        if article.get('description'):
            sentiment = analyze_sentiment(article['description'])
            sentiments.append({
                'article': article,
                'sentiment': sentiment
            })
    
    return sentiments

def analyze_top_gainers(limit=10):
    """Analyze sentiment for top gaining stocks"""
    # Get top gainers
    print("Fetching top gaining stocks...")
    top_gainers = get_top_gainers(limit)
    
    # Store results
    results = []
    
    # Process each stock
    for _, stock in top_gainers.iterrows():
        ticker = stock['Ticker']
        name = stock['Name']
        change = stock['Change']
        
        print(f"\n{'='*50}")
        print(f"Processing {name} ({ticker}) - Change: {change}")
        
        # Get news and analyze sentiment
        sentiments = get_news_sentiment(ticker, name)
        
        if not sentiments:
            print(f"No news found for {name}")
            continue
        
        # Calculate average sentiment
        sentiment_scores = [item['sentiment']['score'] for item in sentiments]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Print articles and sentiments
        print(f"Found {len(sentiments)} articles with average sentiment: {avg_sentiment:.3f}")
        for i, item in enumerate(sentiments, 1):
            article = item['article']
            sentiment = item['sentiment']
            print(f"\n{i}. {article['title']}")
            print(f"Source: {article['source']}")
            print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.3f})")
        
        # Add to results
        results.append({
            'ticker': ticker,
            'name': name,
            'change': change,
            'avg_sentiment': avg_sentiment,
            'article_count': len(sentiments),
            'sentiments': sentiments
        })
    
    return results

def print_summary(results):
    """Print a summary of the sentiment analysis results"""
    # Sort stocks by sentiment score
    sorted_results = sorted(
        [r for r in results if r['article_count'] > 0],
        key=lambda x: x['avg_sentiment'],
        reverse=True
    )
    
    print("\n\n" + "="*50)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*50)
    
    for result in sorted_results:
        print(f"{result['ticker']} - {result['name']} (Change: {result['change']})")
        print(f"Sentiment: {result['avg_sentiment']:.3f} ({result['article_count']} articles)")
        print("-"*30)


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Run the full analysis
    results = analyze_top_gainers()
    print_summary(results) 