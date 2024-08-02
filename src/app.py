import os
import pandas as pd
from flask import Flask, render_template, request
from flask_caching import Cache
from datetime import datetime
import plotly.express as px
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

app = Flask(__name__)

# Configuring cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Paths to the preprocessed data and models
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_with_sentiments_and_summaries.csv'))
tfidf_vectorizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl'))
tfidf_matrix_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_matrix.pkl'))
classifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl'))
kmeans_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'kmeans.pkl'))

print(f"Loading data from {data_path}")
try:
    articles = pd.read_csv(data_path)
    articles['published_at'] = pd.to_datetime(articles['published_at'], errors='coerce').dt.tz_convert('UTC')
    date_min = articles['published_at'].min().strftime('%Y-%m-%d')
    date_max = articles['published_at'].max().strftime('%Y-%m-%d')

    # Load precomputed models and matrices
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    classifier = joblib.load(classifier_path)
    kmeans = joblib.load(kmeans_path)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure the necessary files exist.")
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
    sentiment = ''
    filtered_articles = pd.DataFrame()

    if request.method == 'POST':
        action = request.form.get('action')
        query = request.form.get('query', '')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        sentiment = request.form.get('sentiment', '')

        if action == 'analytics':
            # Prepare data for sentiment trend visualization
            filtered_articles = articles.copy()

            if query:
                query_vector = tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                most_similar_indices = similarities.argsort()[::-1]
                filtered_articles = filtered_articles.iloc[most_similar_indices]

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

            if sentiment:
                filtered_articles = filtered_articles[filtered_articles['roberta_sentiment'].str.lower() == sentiment.lower()]

            trend_data = filtered_articles.copy()
            trend_data['published_at'] = pd.to_datetime(trend_data['published_at'])
            trend_data.set_index('published_at', inplace=True)
            sentiment_counts = trend_data.resample('D')['roberta_sentiment'].value_counts().unstack().fillna(0)

            fig = px.line(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.columns,
                          labels={'value': 'Sentiment Count', 'published_at': 'Date'},
                          title='Sentiment Trend Over Time')
            graph_html = fig.to_html(full_html=False)

            # Detailed exploration of sentiment across topics
            topic_sentiments = filtered_articles.groupby('topic')['roberta_sentiment'].value_counts().unstack().fillna(0)

            # Statistics on sentiment classification accuracy
            sentiment_stats = {
                'total_articles': len(filtered_articles),
                'positive': (filtered_articles['roberta_sentiment'] == 'positive').sum(),
                'neutral': (filtered_articles['roberta_sentiment'] == 'neutral').sum(),
                'negative': (filtered_articles['roberta_sentiment'] == 'negative').sum()
            }

            fig_bar = px.bar(topic_sentiments, barmode='group', title='Sentiment Distribution Across Topics')
            bar_chart_html = fig_bar.to_html(full_html=False)

            return render_template('analytics.html', query=query, from_date=from_date, to_date=to_date, sentiment=sentiment,
                                   graph_html=graph_html, bar_chart_html=bar_chart_html, sentiment_stats=sentiment_stats)
    else:
        query = request.args.get('query', '')
        from_date = request.args.get('from_date', '')
        to_date = request.args.get('to_date', '')
        sentiment = request.args.get('sentiment', '')

    # Construct cache key including page number
    cache_key = f"{query or ''}_{from_date or ''}_{to_date or ''}_{sentiment or ''}_{page}"
    cached_articles = cache.get(cache_key)

    if cached_articles is None:
        filtered_articles = articles.copy()

        if query or from_date or to_date or sentiment:
            if query:
                query_vector = tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                most_similar_indices = similarities.argsort()[::-1]
                filtered_articles = filtered_articles.iloc[most_similar_indices]

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

            if sentiment:
                filtered_articles = filtered_articles[filtered_articles['roberta_sentiment'].str.lower() == sentiment.lower()]

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

    return render_template('index.html', articles=current_articles, pagination=pagination, query=query, from_date=from_date, to_date=to_date, sentiment=sentiment, date_min=date_min, date_max=date_max)

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
