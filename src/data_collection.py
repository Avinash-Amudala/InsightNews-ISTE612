import os
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta

API_KEY = '9b4207b8856b4179969a67dfb9994844'

def fetch_articles(query, from_date, to_date, page=1):
    newsapi = NewsApiClient(api_key=API_KEY)
    all_articles = []
    while True:
        articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt',
            page=page,
            page_size=100
        )
        if 'articles' not in articles or not articles['articles']:
            break
        all_articles.extend(articles['articles'])
        if len(articles['articles']) < 100:
            break
        page += 1
    return all_articles

def save_data(data, filename):
    if not data:
        print(f"No data to save for {filename}")
        return
    df = pd.DataFrame(data)
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if os.path.exists(abs_path):
        df.to_csv(abs_path, mode='a', header=False, index=False)
    else:
        df.to_csv(abs_path, index=False)
    print(f"Data saved to {abs_path}")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    topics = ['technology', 'politics', 'health', 'sports']
    from_date = datetime(2024, 6, 1)
    to_date = datetime(2024, 6, 25)

    for topic in topics:
        print(f"Collecting data for topic: {topic}")
        all_articles = []
        current_date = from_date

        while current_date < to_date:
            next_date = current_date + timedelta(days=1)  # Fetch data in 1-day chunks
            if next_date > to_date:
                next_date = to_date

            try:
                articles = fetch_articles(topic, current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d'))
                print(f"Collected {len(articles)} articles for date range {current_date.strftime('%Y-%m-%d')} to {next_date.strftime('%Y-%m-%d')}")
                all_articles.extend(articles)
                save_data(articles, f'data/raw/{topic}_articles.csv')
            except Exception as e:
                print(f"Error collecting data for date range {current_date.strftime('%Y-%m-%d')} to {next_date.strftime('%Y-%m-%d')}: {e}")

            current_date = next_date + timedelta(days=1)

    print("Data collection complete.")
