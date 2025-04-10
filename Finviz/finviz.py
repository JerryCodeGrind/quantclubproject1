from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from transformers import pipeline

def analyze_stock_sentiment(ticker):
    """Gets sentiment score for a stock from FinViz news"""
    print(f"Analyzing sentiment for {ticker}...")
    
    # Get news from FinViz
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    
    # Find the news table
    news_table = html.find(id='news-table')
    
    if not news_table:
        print(f"No news found for {ticker}")
        return
    
    # Get all news titles and display each article
    titles = []
    print(f"\nArticles for {ticker}:")
    print("-" * 60)
    
    for i, row in enumerate(news_table.find_all('tr'), 1):
        title = row.a.text
        
        # Get date and time
        date_data = row.td.text.split()
        if len(date_data) == 1:
            time = date_data[0]
            date = "Today"
        else:
            date = date_data[0]
            time = date_data[1]
            
        # Get source if available
        source_span = row.find('span', class_='news-link-right')
        source = source_span.text if source_span else "FinViz"
        
        # Print the article details
        print(f"{i}. {title}")
        print(f"   Date: {date} | Time: {time}")
        print(f"   Source: {source}")
        print()
        
        titles.append(title)
    
    print(f"Found {len(titles)} articles")
    
    # Load sentiment model
    sentiment_analyzer = pipeline('text-classification', model='ProsusAI/finbert')
    
    # Get sentiment scores
    total_score = 0
    for title in titles:
        sentiment_score = sentiment_analyzer(title)[0]['score']
        total_score += sentiment_score
    
    # Calculate average
    avg_score = total_score / len(titles)
    
    print(f"Average sentiment score: {avg_score:.3f}")
    return avg_score

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol: ").strip().upper()
    analyze_stock_sentiment(ticker)