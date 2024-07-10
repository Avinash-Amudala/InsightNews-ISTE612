import os
import time
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv('NEWS_API_KEY')

def fetch_articles_newsapi(query, from_date, to_date):
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    all_articles = []
    page = 1
    while True:
        try:
            articles = newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt',
                page=page,
                page_size=100
            )
            if articles['status'] == 'ok' and articles['totalResults'] > 0:
                all_articles.extend(articles['articles'])
                if len(articles['articles']) < 100:
                    break
                page += 1
            else:
                break
        except Exception as e:
            print(f"Error collecting data for topic {query} with NewsAPI: {e}")
            break
        time.sleep(1)
    return all_articles

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
    topics = ['technology', 'politics', 'health', 'sports']
    date_ranges = [('2024-06-01', '2024-06-15'), ('2024-06-16', '2024-06-30')]

    for topic in topics:
        for from_date, to_date in date_ranges:
            print(f"Collecting data for topic: {topic} from {from_date} to {to_date}")
            articles = []
            try:
                newsapi_articles = fetch_articles_newsapi(topic, from_date, to_date)
                articles.extend(newsapi_articles)
            except Exception as e:
                print(f"Skipping NewsAPI due to error: {e}")
            print(f"Collected {len(articles)} articles for topic: {topic}")
            for article in articles:
                article['topic'] = topic
            save_data(articles, f'data/raw/newsapi_{topic}_{from_date}_to_{to_date}_articles.csv')
    print("Data collection complete.")
