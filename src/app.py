import os
from flask import Flask, render_template, request
import pandas as pd
from sentiment_analysis import analyze_sentiment, setup_sentiment_analyzer

app = Flask(__name__)

DEFAULT_IMAGES = {
    'technology': 'static/images/technology_default.jpg',
    'politics': 'static/images/politics_default.jpg',
    'health': 'static/images/health_default.jpg',
    'sports': 'static/images/sports_default.jpg'
}

def clean_dataframe(df):
    df = df.replace(r'\n', ' ', regex=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data_file = '../data/processed/articles_cleaned.csv'
    df = pd.read_csv(data_file)

    df = clean_dataframe(df)

    sentiment_analyzer = setup_sentiment_analyzer()

    df = analyze_sentiment(df, sentiment_analyzer)

    df = clean_dataframe(df)

    df = df[['author', 'title', 'description', 'url', 'source', 'image', 'published_at', 'topic', 'sentiment']]

    df['image'] = df.apply(lambda row: row['image'] if pd.notna(row['image']) else DEFAULT_IMAGES.get(row['topic'], 'static/images/default.jpg'), axis=1)

    return render_template('results.html', articles=df.to_dict(orient='records'))

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=True)
