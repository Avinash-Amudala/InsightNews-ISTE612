import os
from flask import Flask, render_template, request
import pandas as pd
import re
from flask_caching import Cache
from datetime import datetime

app = Flask(__name__)

# Configuring cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Use the preprocessed data with sentiment and summary
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_with_sentiment.csv'))
print(f"Loading data from {data_path}")
try:
    articles = pd.read_csv(data_path)
    articles = articles.drop_duplicates('summary')
    pattern = re.compile(r'image \d+ of \d+', re.IGNORECASE)
    articles = articles[~articles['title'].apply(lambda x: bool(pattern.search(x)) if pd.notnull(x) else False)]
    articles['published_at'] = pd.to_datetime(articles['published_at'], errors='coerce').dt.tz_convert('UTC')
    date_min = articles['published_at'].min().strftime('%Y-%m-%d')
    date_max = articles['published_at'].max().strftime('%Y-%m-%d')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure the articles_with_sentiment.csv file exists.")
    exit(1)

DEFAULT_IMAGES = {
    'technology': 'static/images/technology_default.jpg',
    'politics': 'static/images/politics_default.jpg',
    'health': 'static/images/health_default.jpg',
    'sports': 'static/images/sports_default.jpg'
}

FALLBACK_IMAGE = 'static/images/default.jpg'

def calculate_page_range(current_page, total_pages, delta=2):
    start_page = max(current_page - delta, 1)
    end_page = min(current_page + delta, total_pages) + 1
    return range(start_page, end_page)

@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 12

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

    # Construct cache key including page number
    cache_key = f"{query or ''}_{from_date or ''}_{to_date or ''}_{page}"
    cached_articles = cache.get(cache_key)

    if cached_articles is None:
        if query or from_date or to_date:
            filtered_articles = articles.copy()
            if query:
                filtered_articles = filtered_articles[
                    filtered_articles['content'].str.contains(query, case=False, na=False) |
                    filtered_articles['title'].str.contains(query, case=False, na=False) |
                    filtered_articles['author'].str.contains(query, case=False, na=False) |
                    filtered_articles['source'].str.contains(query, case=False, na=False)
                ]
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

            current_articles['image'] = current_articles.apply(
                lambda row: row['image'] if pd.notna(row['image']) else DEFAULT_IMAGES.get(row['topic'].lower(), FALLBACK_IMAGE),
                axis=1
            )

            pagination = {
                'total': total,
                'pages': (total - 1) // per_page + 1,
                'page': page,
                'has_prev': page > 1,
                'has_next': page < (total - 1) // per_page + 1,
                'page_range': calculate_page_range(page, (total - 1) // per_page + 1)
            }
        else:
            current_articles = articles.sort_values(by='published_at', ascending=False).iloc[(page-1)*per_page:page*per_page].copy()
            current_articles['image'] = current_articles.apply(
                lambda row: row['image'] if pd.notna(row['image']) else DEFAULT_IMAGES.get(row['topic'].lower(), FALLBACK_IMAGE),
                axis=1
            )

            pagination = {
                'total': len(articles),
                'pages': (len(articles) - 1) // per_page + 1,
                'page': page,
                'has_prev': page > 1,
                'has_next': page < (len(articles) - 1) // per_page + 1,
                'page_range': calculate_page_range(page, (len(articles) - 1) // per_page + 1)
            }

        cached_articles = (current_articles.to_dict(orient='records'), pagination)
        cache.set(cache_key, cached_articles, timeout=5 * 60)

    current_articles, pagination = cached_articles

    return render_template('index.html', articles=current_articles, pagination=pagination, query=query, from_date=from_date, to_date=to_date, date_min=date_min, date_max=date_max)

@app.template_filter('dateformat')
def dateformat(value, format='%B %d, %Y'):
    """Format a date string or Timestamp into a more readable format."""
    if isinstance(value, datetime):
        date_obj = value
    else:
        try:
            date_obj = datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            date_obj = value.to_pydatetime()

    return date_obj.strftime(format)

if __name__ == '__main__':
    app.run(debug=True)
