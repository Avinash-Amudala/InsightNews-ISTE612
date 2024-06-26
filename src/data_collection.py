import os
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('NEWS_API_KEY')

def fetch_articles(query, from_date, to_date):
    newsapi = NewsApiClient(api_key=API_KEY)
    articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language='en',
        sort_by='publishedAt',
        page=1,
        page_size=100
    )
    return articles['articles']

def save_data(data, filename):
    if not data:
        print(f"No data to save for {filename}")
        return
    df = pd.DataFrame(data)
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"Data saved to {abs_path}")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    topics = ['technology', 'politics', 'health', 'sports']  # Modify topics as needed
    from_date = '2024-06-01'  # Adjusted date to fit within the allowed range
    to_date = '2024-06-25'    # Current date

    for topic in topics:
        print(f"Collecting data for topic: {topic}")
        try:
            articles = fetch_articles(topic, from_date, to_date)
            print(f"Collected {len(articles)} articles for topic: {topic}")
            for article in articles:
                article['topic'] = topic
            save_data(articles, f'data/raw/{topic}_articles.csv')
        except Exception as e:
            print(f"Error collecting data for topic {topic}: {e}")
    print("Data collection complete.")
