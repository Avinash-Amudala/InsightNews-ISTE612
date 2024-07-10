import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import http.client, urllib.parse
import json

load_dotenv()

MEDIASTACK_KEY = os.getenv('MEDIASTACK_API_KEY')

def fetch_articles_mediastack(query, from_date, to_date, api_key):
    all_articles = []
    offset = 0
    request_count = 0

    while request_count < 500:
        params = urllib.parse.urlencode({
            'access_key': api_key,
            'keywords': query,
            'date': f"{from_date},{to_date}",
            'languages': 'en',
            'sort': 'published_desc',
            'offset': offset,
            'limit': 100
        })
        conn = http.client.HTTPConnection('api.mediastack.com')
        conn.request('GET', f'/v1/news?{params}')
        res = conn.getresponse()
        data = res.read()
        try:
            articles = json.loads(data.decode('utf-8')).get('data', [])
            print(f"Fetched {len(articles)} articles for query: {query} from {from_date} to {to_date} with offset {offset}")
            request_count += 1
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            break

        if articles:
            all_articles.extend(articles)
            if len(articles) < 100:
                break
            offset += 100
        else:
            break
        time.sleep(1)

    print(f"Total requests made: {request_count}")
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
    date_ranges = [
        ('2024-06-01', '2024-06-05'),
        ('2024-06-06', '2024-06-10'),
        ('2024-06-11', '2024-06-15'),
        ('2024-06-16', '2024-06-20'),
        ('2024-06-21', '2024-06-25'),
        ('2024-06-26', '2024-06-30')
    ]

    for topic in topics:
        for from_date, to_date in date_ranges:
            print(f"Collecting data for topic: {topic} from {from_date} to {to_date}")
            articles = []
            try:
                mediastack_articles = fetch_articles_mediastack(topic, from_date, to_date, MEDIASTACK_KEY)
                articles.extend(mediastack_articles)
            except Exception as e:
                print(f"Error collecting data for topic {topic} with MediaStack: {e}")
            print(f"Collected {len(articles)} articles for topic: {topic}")
            for article in articles:
                article['topic'] = topic
            save_data(articles, f'data/raw/mediastack_{topic}_{from_date}_to_{to_date}_articles.csv')
    print("Data collection complete.")
