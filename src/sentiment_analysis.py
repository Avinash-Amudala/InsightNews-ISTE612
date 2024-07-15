import pandas as pd
from transformers import pipeline

def setup_summarizer():
    return pipeline('summarization')

def summarize_content(df, summarizer):
    df['summary'] = df['cleaned_content'].apply(lambda x: summarizer(x, max_length=50, min_length=25, do_sample=False)[0]['summary_text'])
    return df

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
    output_file = 'data/processed/articles_with_sentiment_and_summary.csv'

    print("Loading cleaned data...")
    df = load_cleaned_data(input_file)

    print("Setting up sentiment analyzer...")
    sentiment_analyzer = setup_sentiment_analyzer()

    print("Analyzing sentiment...")
    df = analyze_sentiment(df, sentiment_analyzer)

    print("Setting up summarizer...")
    summarizer = setup_summarizer()

    print("Summarizing content...")
    df = summarize_content(df, summarizer)

    print("Saving results...")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
