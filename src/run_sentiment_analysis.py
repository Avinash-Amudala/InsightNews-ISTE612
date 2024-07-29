import pandas as pd
from sentiment_analysis import analyze_sentiment, setup_sentiment_analyzer, setup_summarizer, summarize_content

def run_sentiment_analysis(input_file, output_file):
    print("Loading cleaned data...")
    df = pd.read_csv(input_file)

    print("Setting up sentiment analyzer...")
    sentiment_analyzer = setup_sentiment_analyzer(model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    print("Analyzing sentiment...")
    df = analyze_sentiment(df, sentiment_analyzer)

    print("Setting up summarizer...")
    summarizer = setup_summarizer(model_name="sshleifer/distilbart-cnn-12-6")

    print("Summarizing content...")
    df = summarize_content(df, summarizer, max_length=30)

    print("Saving results...")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = '../data/processed/articles_cleaned.csv'
    output_file = '../data/processed/articles_with_sentiment.csv'
    run_sentiment_analysis(input_file, output_file)
