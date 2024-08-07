import os
import re
import pandas as pd
from flask import Flask, render_template, request, flash
from flask_caching import Cache
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = 'supersecretkey'

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


def filter_articles(query, author, title, source, from_date, to_date, sentiment, sentiment_model, sort_by):
    filtered_articles = articles.copy()
    
    # Apply filters with regex support
    if author:
        filtered_articles = filtered_articles[filtered_articles['author'].str.contains(author, case=False, na=False, regex=True)]

    if title:
        filtered_articles = filtered_articles[filtered_articles['title'].str.contains(title, case=False, na=False, regex=True)]

    if source:
        filtered_articles = filtered_articles[filtered_articles['source'].str.contains(source, case=False, na=False, regex=True)]

    if from_date:
        from_date = pd.to_datetime(from_date, errors='coerce')
        if from_date.tzinfo is None:
            from_date = from_date.tz_localize('UTC')
        filtered_articles = filtered_articles[filtered_articles['published_at'] >= from_date]

    if to_date:
        to_date = pd.to_datetime(to_date, errors='coerce')
        if to_date.tzinfo is None:
            to_date = to_date.tz_localize('UTC')
        filtered_articles = filtered_articles[filtered_articles['published_at'] <= to_date + pd.Timedelta(days=1)]

    if sentiment:
        filtered_articles = filtered_articles[filtered_articles[sentiment_model].str.lower() == sentiment.lower()]

    # Apply query filter with regex support and recompute similarities
    if query:
        filtered_articles = filtered_articles[filtered_articles['content'].str.contains(query, case=False, na=False, regex=True)]
        tfidf_matrix_filtered = tfidf_vectorizer.transform(filtered_articles['content'])
        query_vector = tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix_filtered).flatten()
        threshold = 0.1  # Set a threshold for filtering articles
        article_indices = similarities > threshold
        filtered_articles = filtered_articles.iloc[article_indices]
        similarities = similarities[article_indices]
        filtered_articles['relevance'] = similarities

        # Sort articles
        if sort_by == 'desc':
            filtered_articles = filtered_articles.sort_values(by='published_at', ascending=False)
        elif sort_by == 'asc':
            filtered_articles = filtered_articles.sort_values(by='published_at', ascending=True)
        elif sort_by == 'alphabetical_asc':
            filtered_articles = filtered_articles.sort_values(by='title', ascending=True)
        elif sort_by == 'alphabetical_desc':
            filtered_articles = filtered_articles.sort_values(by='title', ascending=False)
        elif sort_by == 'relevance_desc':
            filtered_articles = filtered_articles.sort_values(by='relevance', ascending=False)
        elif sort_by == 'relevance_asc':
            filtered_articles = filtered_articles.sort_values(by='relevance', ascending=True)

    return filtered_articles


@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 12

    query = ''
    author = ''
    title = ''
    source = ''
    from_date = ''
    to_date = ''
    sentiment = ''
    sentiment_model = 'roberta_sentiment'
    sort_by = 'desc'
    filtered_articles = pd.DataFrame()
    total = len(articles)

    graph_html = None
    bar_chart_html = None
    pie_chart_html = None
    source_html = None
    author_html = None
    geo_html = None
    length_html = None
    topic_html = None
    action = None

    if request.method == 'POST':
        action = request.form.get('action')
        query = request.form.get('query', '')
        author = request.form.get('author', '')
        title = request.form.get('title', '')
        source = request.form.get('source', '')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        sentiment = request.form.get('sentiment', '')
        sentiment_model = request.form.get('sentiment_model', 'roberta_sentiment')
        sort_by = request.form.get('sort_by', 'desc')

        filtered_articles = filter_articles(query, author, title, source, from_date, to_date, sentiment, sentiment_model, sort_by)
        total = len(filtered_articles)
        
        if filtered_articles.empty:
            flash('No articles found matching your search criteria<br>Please try using Advanced Search', 'danger')
            return render_template('index.html', articles=[], pagination=None, query=query, author=author, title=title, source=source, from_date=from_date, to_date=to_date, sentiment=sentiment, sentiment_model=sentiment_model, date_min=date_min, date_max=date_max, graph_html=None, bar_chart_html=None, pie_chart_html=None, source_html=None, author_html=None, geo_html=None, length_html=None, topic_html=None, action=action, total=0)

        if action == 'analytics':
            total = len(filtered_articles)
            # Debugging: Print filtered articles count
            print(f"Filtered Articles Count: {len(filtered_articles)}")
            print(f"Filtered Articles Sentiments: {filtered_articles[sentiment_model].value_counts()}")

            trend_data = filtered_articles.copy()
            trend_data['published_at'] = pd.to_datetime(trend_data['published_at'])
            trend_data.set_index('published_at', inplace=True)
            sentiment_counts = trend_data.resample('D')[sentiment_model].value_counts().unstack().fillna(0)

            fig = px.line(sentiment_counts.reset_index(), x='published_at', y=sentiment_counts.columns,
                          labels={'value': 'Sentiment Count', 'published_at': 'Date'},
                          title='Sentiment Trend Over Time',
                          template='plotly_dark')
            graph_html = fig.to_html(full_html=False)

            # Detailed exploration of sentiment across topics
            topic_sentiments = filtered_articles.groupby('topic')[sentiment_model].value_counts().unstack().fillna(0)

            fig_bar = px.bar(topic_sentiments.reset_index(), x='topic', y=topic_sentiments.columns, barmode='group', title='Sentiment Distribution Across Topics', template='plotly_dark')
            bar_chart_html = fig_bar.to_html(full_html=False)

            # Top 10 sources by sentiment
            source_sentiments = filtered_articles.groupby('source')[sentiment_model].value_counts().unstack().fillna(0)
            top_sources = source_sentiments.sum(axis=1).nlargest(10).index
            fig_source = px.bar(source_sentiments.loc[top_sources].reset_index(), x='source', y=source_sentiments.columns, barmode='group', title='Top 10 Sources by Sentiment', template='plotly_dark')
            source_html = fig_source.to_html(full_html=False)

            # Top authors by sentiment (Top 10)
            author_sentiments = filtered_articles.groupby('author')[sentiment_model].value_counts().unstack().fillna(0)
            top_authors = author_sentiments.sum(axis=1).nlargest(10).index
            fig_author = px.bar(author_sentiments.loc[top_authors].reset_index(), x='author', y=author_sentiments.columns, barmode='group', title='Top Authors by Sentiment (Top 10)', template='plotly_dark')
            author_html = fig_author.to_html(full_html=False)

            # Geographical sentiment distribution
            geo_sentiments = filtered_articles.groupby('country')[sentiment_model].value_counts().unstack().fillna(0)
            geo_sentiments['total'] = geo_sentiments.sum(axis=1)
            fig_geo = px.scatter_geo(geo_sentiments.reset_index(), locationmode='country names', locations='country', size='total', title='Geographical Sentiment Distribution', template='plotly_dark')
            geo_html = fig_geo.to_html(full_html=False)

            # Sentiment by article length
            length_sentiments = filtered_articles.groupby(filtered_articles['content'].str.len())[sentiment_model].value_counts().unstack().fillna(0)
            fig_length = px.scatter(length_sentiments.reset_index(), x='content', y=length_sentiments.columns, title='Sentiment by Article Length', template='plotly_dark')
            length_html = fig_length.to_html(full_html=False)

            # Trending topics with sentiment
            topic_sentiments_trend = filtered_articles.groupby(['topic', 'published_at'])[sentiment_model].value_counts().unstack().fillna(0)
            fig_topic = px.line(topic_sentiments_trend.reset_index(), x='published_at', y=topic_sentiments_trend.columns, color='topic', title='Trending Topics with Sentiment', template='plotly_dark')
            topic_html = fig_topic.to_html(full_html=False)

            # Sentiment classification pie chart
            sentiment_stats = filtered_articles[sentiment_model].value_counts()
            fig_pie = go.Figure(data=[go.Pie(labels=sentiment_stats.index, values=sentiment_stats.values, hole=.3)])
            fig_pie.update_layout(title_text='Sentiment Classification Distribution', template='plotly_dark')
            pie_chart_html = fig_pie.to_html(full_html=False)

            # Check if visualizations are generated
            if not graph_html:
                print("Error: graph_html is empty")
            if not bar_chart_html:
                print("Error: bar_chart_html is empty")
            if not pie_chart_html:
                print("Error: pie_chart_html is empty")
            if not source_html:
                print("Error: source_html is empty")
            if not author_html:
                print("Error: author_html is empty")
            if not geo_html:
                print("Error: geo_html is empty")
            if not length_html:
                print("Error: length_html is empty")
            if not topic_html:
                print("Error: topic_html is empty")

    else:
        query = request.args.get('query', '')
        author = request.args.get('author', '')
        title = request.args.get('title', '')
        source = request.args.get('source', '')
        from_date = request.args.get('from_date', '')
        to_date = request.args.get('to_date', '')
        sentiment = request.args.get('sentiment', '')
        sentiment_model = request.args.get('sentiment_model', 'roberta_sentiment')
        sort_by = request.args.get('sort_by', 'desc')

        filtered_articles = filter_articles(query, author, title, source, from_date, to_date, sentiment, sentiment_model, sort_by)

        if filtered_articles.empty:
            flash('No articles found matching your search criteria<br>Please try using Advanced Search', 'danger')
            return render_template('index.html', articles=[], pagination=None, query=query, author=author, title=title, source=source, from_date=from_date, to_date=to_date, sentiment=sentiment, sentiment_model=sentiment_model, sort_by=sort_by, date_min=date_min, date_max=date_max, graph_html=None, bar_chart_html=None, pie_chart_html=None, source_html=None, author_html=None, geo_html=None, length_html=None, topic_html=None, action=action, total=0)

    # Construct cache key including page number
    cache_key = f"{query or ''}_{author or ''}_{title or ''}_{source or ''}_{from_date or ''}_{to_date or ''}_{sentiment or ''}_{sentiment_model or ''}_{sort_by}_{page}"
    cached_articles = cache.get(cache_key)

    if cached_articles is None:
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

        cached_articles = (current_articles.to_dict(orient='records'), pagination)
        cache.set(cache_key, cached_articles, timeout=5 * 60)

    current_articles, pagination = cached_articles

    # Update heading logic
    heading = "News Articles"
    if query or from_date or to_date:
        heading = "Searched for "
        if query:
            heading += f'"{query}" '
        if from_date:
            heading += f'from {from_date} '
        if to_date:
            heading += f'to {to_date} '

    return render_template('index.html', articles=current_articles, pagination=pagination, query=query, author=author, title=title, source=source, from_date=from_date, to_date=to_date, sentiment=sentiment, sentiment_model=sentiment_model, sort_by=sort_by, date_min=date_min, date_max=date_max, graph_html=graph_html, bar_chart_html=bar_chart_html, pie_chart_html=pie_chart_html, source_html=source_html, author_html=author_html, geo_html=geo_html, length_html=length_html, topic_html=topic_html, action=action, total=total, heading=heading)

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