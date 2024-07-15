import pandas as pd
from transformers import pipeline

def load_cleaned_data(file_path):
    return pd.read_csv(file_path)

def setup_sentiment_analyzer():
    return pipeline('sentiment-analysis')

def analyze_sentiment(df, analyzer):
    df = df.head(10)  # Limit to 10 rows for the demo
    sentiments = df['cleaned_content'].apply(lambda x: analyzer(x)[0]['label'])
    df['sentiment'] = sentiments
    return df

if __name__ == "__main__":
    input_file = 'data/processed/articles_cleaned.csv'
    output_file = 'data/processed/articles_with_sentiment.csv'

    print("Loading cleaned data...")
    df = load_cleaned_data(input_file)

    print("Setting up sentiment analyzer...")
    sentiment_analyzer = setup_sentiment_analyzer()

    print("Analyzing sentiment...")
    df = analyze_sentiment(df, sentiment_analyzer)

    print("Saving results...")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
