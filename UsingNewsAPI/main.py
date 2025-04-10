from bs4 import BeautifulSoup
import requests
import pandas as pd
from transformers import pipeline
import os
import warnings
from newsapi import NewsApiClient

# Add these lines right after the imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def get_top_gainers():
    url = "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    # Find all rows for top gainers (update class name if it changes)
    data = []
    all_gainers = soup.find_all('tr', class_='row-RdUXZpkv listRow')

    for stock in all_gainers[:10]:
        stock_data = {}
        
        # Get the percentage change cell
        change_cell = stock.find('td', class_='cell-RLhfr_y4 right-RLhfr_y4')
        name_cell = stock.find('sup', class_='apply-common-tooltip tickerDescription-GrtoTeat')
        
        if change_cell and name_cell:
            stock_data['Change'] = change_cell.text.strip()
            stock_data['Name'] = name_cell.text.strip()
            data.append(stock_data)

    return pd.DataFrame(data)

def finbert_sentiment(text):
    # Load the Prosus AI FinBERT model
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    result = finbert(text)[0]['score']
    return result

def search_news_api(query):
    """
    Search for news articles using NewsAPI.
    """
    newsapi = NewsApiClient(api_key='d46263b53201426b9c776c43cd951c10')
    
    try:
        response = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        
        articles = []
        if response['status'] == 'ok':
            for article in response['articles']:
                # Only include articles that have a description
                if article.get('description'):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', '')
                    })
                
        return articles
        
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

if __name__ == '__main__':
    top_gainers = get_top_gainers()
    
    # Dictionary to store average sentiment per ticker
    ticker_sentiments = {}
    
    for index, row in top_gainers.head(10).iterrows():
        name = row['Name']
        query = f"{name} Stock Recent News"
        articles = search_news_api(query)
        
        print(f"\n=== Articles for {name} ===")
        sentiments = []
        for article in articles:
            sentiment = finbert_sentiment(article['description'])
            sentiments.append(sentiment)
            print(f"\nSource: {article['source']}")
            print(f"Title: {article['title']}")
            print(f"Description: {article['description']}")
            print(f"Sentiment: {sentiment}")
        
        # Store average sentiment (only if we have valid sentiments)
        ticker_sentiments[name] = sum(sentiments) / len(sentiments) if sentiments else 0
    
    # Print final sentiment summary with percent changes directly from DataFrame
    print("\n=== Overall Sentiment Summary ===")
    for index, row in top_gainers.iterrows():
        name = row['Name']
        sentiment = ticker_sentiments.get(name, 'N/A')
        print(f"{name} ({row['Change']}): {sentiment:.3f}")