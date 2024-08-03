import pandas as pd
from transformers import pipeline
import gc
from tqdm import tqdm
import torch
import os

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load your data
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_cleaned.csv'))
df = pd.read_csv(file_path)

# Load the sentiment analysis pipelines with RoBERTa and DistilBERT models and GPU support
roberta_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)
distilbert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# Function to process data in batches with progress display
def process_in_batches(data, pipeline, batch_size, model_name):
    sentiments = []
    total_batches = len(data) // batch_size + 1
    for i in tqdm(range(0, len(data), batch_size), total=total_batches, desc=f"Processing {model_name} batches"):
        batch = data[i:i+batch_size]
        batch_sentiments = pipeline(batch)
        sentiments.extend([sent['label'] for sent in batch_sentiments])
        gc.collect()  # Clear memory
        torch.cuda.empty_cache()  # Clear GPU cache if using GPU
    return sentiments

# Apply the RoBERTa sentiment analysis to the cleaned_content column in batches
df['roberta_sentiment'] = process_in_batches(df['cleaned_content'].tolist(), roberta_pipeline, batch_size=10000, model_name="RoBERTa")

# Apply the DistilBERT sentiment analysis to the cleaned_content column in batches
df['distilbert_sentiment'] = process_in_batches(df['cleaned_content'].tolist(), distilbert_pipeline, batch_size=10000, model_name="DistilBERT")

# Map RoBERTa sentiment labels to positive, negative, and neutral
def map_roberta_sentiment(sentiment):
    if sentiment == 'LABEL_0':
        return 'negative'
    elif sentiment == 'LABEL_1':
        return 'neutral'
    elif sentiment == 'LABEL_2':
        return 'positive'

df['roberta_sentiment'] = df['roberta_sentiment'].apply(map_roberta_sentiment)

# Save the updated dataframe
def save_data(df, filename):
    if df.empty:
        print(f"No data to save for {filename}")
        return
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"Data saved to {abs_path}")

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_with_sentiment.csv'))
save_data(df, output_file)

# Display sentiment counts for both models
roberta_sentiment_counts = df['roberta_sentiment'].value_counts()
distilbert_sentiment_counts = df['distilbert_sentiment'].value_counts()

print("RoBERTa Sentiment Counts:")
print(roberta_sentiment_counts)

print("\nDistilBERT Sentiment Counts:")
print(distilbert_sentiment_counts)
