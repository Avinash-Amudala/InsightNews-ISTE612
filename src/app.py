import os
from flask import Flask, render_template, request
import pandas as pd
from flask_caching import Cache

app = Flask(__name__)

data_path = os.path.join('data', 'processed', 'articles_cleaned.csv')
articles = pd.read_csv(data_path)

articles['published_at'] = pd.to_datetime(articles['published_at'], errors='coerce').dt.tz_convert('UTC')
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

DEFAULT_IMAGES = {
    'technology': 'static/images/technology_default.jpg',
    'politics': 'static/images/politics_default.jpg',
    'health': 'static/images/health_default.jpg',
    'sports': 'static/images/sports_default.jpg'
}

FALLBACK_IMAGE = 'static/images/default.jpg'

analyzer = setup_sentiment_analyzer()
summarizer = setup_summarizer()

@app.route('/', methods=['GET', 'POST'])
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
    page = request.args.get('page', 1, type=int)
    per_page = 10

    # Initialize empty variables for query parameters
    query = ''
    from_date = ''
    to_date = ''
    filtered_articles = pd.DataFrame()

    
    if request.method == 'POST':
        query = request.form.get('query', '')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
    else:
        query = request.args.get('query', '')
        from_date = request.args.get('from_date', '')
        to_date = request.args.get('to_date', '')

    if query or from_date or to_date:
        filtered_articles = articles.copy()
        if query:
            filtered_articles = filtered_articles[filtered_articles['content'].str.contains(query, case=False, na=False)]
        if from_date:
            from_date = pd.to_datetime(from_date, errors='coerce')
            if from_date.tzinfo is None:
                from_date = from_date.tz_localize('UTC')
            filtered_articles = filtered_articles[filtered_articles['published_at'] >= from_date]
        if to_date:
            to_date = pd.to_datetime(to_date, errors='coerce')
            if to_date.tzinfo is None:
                to_date = to_date.tz_localize('UTC')
            filtered_articles = filtered_articles[filtered_articles['published_at'] <= to_date]

        total = len(filtered_articles)
        start = (page - 1) * per_page
        end = start + per_page
        current_articles = filtered_articles.iloc[start:end].copy()

        
        current_articles['sentiment'] = current_articles['content'].apply(lambda x: analyze_sentiment(analyzer, x))
        current_articles['summary'] = current_articles['content'].apply(lambda x: summarize_content(summarizer, x))
        current_articles['image'] = current_articles['category'].apply(lambda x: DEFAULT_IMAGES.get(x, FALLBACK_IMAGE))
        
        pagination = {
            'total': total,
            'pages': (total - 1) // per_page + 1,
            'page': page,
            'has_prev': page > 1,
            'has_next': page < (total - 1) // per_page + 1,
        }
    else:
        current_articles = pd.DataFrame()
        pagination = {
            'total': 0,
            'pages': 1,
            'page': 1,
            'has_prev': False,
            'has_next': False,
        }

    return render_template('index.html', articles=current_articles.to_dict(orient='records'), pagination=pagination, query=query, from_date=from_date, to_date=to_date)
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

if __name__ == '__main__':
if __name__ == "__main__":
    app.run(debug=True)
