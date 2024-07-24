import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

DEFAULT_IMAGES = {
    'technology': 'static/images/technology_default.jpg',
    'politics': 'static/images/politics_default.jpg',
    'health': 'static/images/health_default.jpg',
    'sports': 'static/images/sports_default.jpg'
}

def clean_dataframe(df):
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace(r'\\n', ' ', regex=True)
    df = df.infer_objects(copy=False)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

def load_filtered_data(file_path, search=None, from_date=None, to_date=None):
    # Load the dataset with datetime parsing
    df = pd.read_csv(file_path, parse_dates=['published_at'])
    print(f"Loaded {len(df)} records from {file_path}")

    # Filter based on search term
    if search:
        search = search.lower()
        df = df[
            df['topic'].str.lower().str.contains(search, na=False) |
            df['author'].str.lower().str.contains(search, na=False) |
            df['title'].str.lower().str.contains(search, na=False) |
            df['source'].str.lower().str.contains(search, na=False)
        ]
        print(f"Filtered by search '{search}': {len(df)} records remaining")

    # Filter based on date range
    if from_date and to_date:
        df = df[(df['published_at'] >= from_date) & (df['published_at'] <= to_date)]
        print(f"Filtered by date range {from_date} to {to_date}: {len(df)} records remaining")

    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@cache.cached(timeout=60, query_string=True)
def analyze():
    search = request.form.get('search')
    from_date = request.form.get('from_date')
    to_date = request.form.get('to_date')

    print(f"Search: {search}, From: {from_date}, To: {to_date}")

    data_file = '../data/processed/articles_with_sentiment.csv'
    df = load_filtered_data(data_file, search, from_date, to_date)

    if df.empty:
        return render_template('no_results.html', search=search)

    df = clean_dataframe(df)
    df['image'] = df.apply(lambda row: row['image'] if pd.notna(row['image']) else DEFAULT_IMAGES.get(row['topic'], 'static/images/default.jpg'), axis=1)

    return render_template('results.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == "__main__":
    app.run(debug=True)
