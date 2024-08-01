import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

def analyze_sentiment(df, tokenizer, model):
    print("Starting sentiment analysis...")
    inputs = tokenizer(list(df['cleaned_content']), return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).numpy()
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['sentiment'] = [sentiment_map[pred] for pred in predictions]
    print("Sentiment analysis complete.")
    return df

def summarize_content(df, tokenizer, model, max_length=30):
    print("Starting summarization...")
    summaries = []
    for text in df['cleaned_content']:
        inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate()
        summary = tokenizer.decode()
        summaries.append(summary)
    df['summary'] = summaries
    print("Summarization complete.")
    return df

def process_batch(df_batch, sentiment_tokenizer, sentiment_model, summarizer_tokenizer, summarizer_model):
    print(f"Processing batch with {len(df_batch)} records...")
    df_batch = analyze_sentiment(df_batch, sentiment_tokenizer, sentiment_model)
    df_batch = summarize_content(df_batch, summarizer_tokenizer, summarizer_model)
    print(f"Batch processing complete.")
    return df_batch

def run_sentiment_analysis(input_file, output_file, batch_size=50):
    print("Loading cleaned data...")
    df = pd.read_csv(input_file)
    print("Columns in the dataframe:", df.columns)

    print("Setting up sentiment analyzer...")
    sentiment_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    sentiment_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
    print("Setting up summarizer...")
    summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    print("Processing in batches...")
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            df_batch = df[i:i+batch_size]
            futures.append(executor.submit(process_batch, df_batch, sentiment_tokenizer, sentiment_model, summarizer_tokenizer, summarizer_model))
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing batch: {e}")

    print("Combining results...")
    final_df = pd.concat(results, ignore_index=True)

    print("Saving results...")
    final_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_cleaned.csv'))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_with_sentiment.csv'))
    run_sentiment_analysis(input_file, output_file)
