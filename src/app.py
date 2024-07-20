import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentiment_analysis import analyze_sentiment, setup_sentiment_analyzer, setup_summarizer, summarize_content

app = Flask(__name__)

DEFAULT_IMAGES = {
    'technology': 'static/images/technology_default.jpg',
    'politics': 'static/images/politics_default.jpg',
    'health': 'static/images/health_default.jpg',
    'sports': 'static/images/sports_default.jpg'
}

def clean_dataframe(df):
    df = df.replace(r'\n', ' ', regex=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    search = request.form.get('search')
    from_date = request.form.get('from_date')
    to_date = request.form.get('to_date')

    data_file = '../data/processed/articles_cleaned.csv'
    df = pd.read_csv(data_file)

    if search:
        search = search.lower()
        df = df[
            df['topic'].str.lower().str.contains(search) |
            df['author'].str.lower().str.contains(search, na=False) |
            df['title'].str.lower().str.contains(search, na=False) |
            df['source'].str.lower().str.contains(search, na=False)
            ]

    df['published_at'] = pd.to_datetime(df['published_at'])
    df = df[(df['published_at'] >= from_date) & (df['published_at'] <= to_date)]

    if df.empty:
        return render_template('no_results.html', search=search)

    df = clean_dataframe(df)

    sentiment_analyzer = setup_sentiment_analyzer()
    summarizer = setup_summarizer()

    df = df.head(10)
    df = analyze_sentiment(df, sentiment_analyzer)
    df = summarize_content(df, summarizer)

    df = clean_dataframe(df)
    df['image'] = df.apply(lambda row: row['image'] if pd.notna(row['image']) else DEFAULT_IMAGES.get(row['topic'], 'static/images/default.jpg'), axis=1)

    return render_template('results.html', articles=df.to_dict(orient='records'), search=search)

@app.route('/load_more', methods=['POST'])
def load_more():
    search = request.form['search']
    offset = int(request.form['offset'])

    data_file = '../data/processed/articles_cleaned.csv'
    df = pd.read_csv(data_file)

    search = search.lower()
    df = df[
        df['topic'].str.lower().str.contains(search) |
        df['author'].str.lower().str.contains(search, na=False) |
        df['title'].str.lower().str.contains(search, na=False) |
        df['source'].str.lower().str.contains(search, na=False)
        ]

    if df.empty:
        return jsonify({'status': 'no_more'})

    df = clean_dataframe(df)
    sentiment_analyzer = setup_sentiment_analyzer()
    summarizer = setup_summarizer()

    df = df.iloc[offset:offset + 10]
    if df.empty:
        return jsonify({'status': 'no_more'})

    df = analyze_sentiment(df, sentiment_analyzer)
    df = summarize_content(df, summarizer)

    df = clean_dataframe(df)
    df['image'] = df.apply(lambda row: row['image'] if pd.notna(row['image']) else DEFAULT_IMAGES.get(row['topic'], 'static/images/default.jpg'), axis=1)

    return jsonify({'status': 'success', 'articles': df.to_dict(orient='records')})

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=True)
