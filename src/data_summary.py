import pandas as pd
from transformers import pipeline
import torch
from torch.utils.data import DataLoader, Dataset
import gc
from tqdm import tqdm
import os

class ArticleDataset(Dataset):
    def __init__(self, articles):
        self.articles = articles

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.articles[idx]

# Load your data
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_with_sentiment.csv'))
df = pd.read_csv(file_path)

# Load the summarization pipeline with the BART model and GPU support
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Define batch size
batch_size = 15000

# Prepare dataset and dataloader
dataset = ArticleDataset(df['cleaned_content'].tolist())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Function to process data in batches with progress display
def process_in_batches(dataloader):
    summaries = []
    total_batches = len(dataloader)
    for batch in tqdm(dataloader, total=total_batches, desc="Processing batches"):
        batch = list(batch)  # Convert to list for pipeline
        batch_summaries = summarization_pipeline(batch, truncation=True, max_length=50, min_length=10)
        summaries.extend([summ['summary_text'] for summ in batch_summaries])
        gc.collect()  # Clear memory
        torch.cuda.empty_cache()  # Clear GPU cache
    return summaries

# Apply the summarization to the cleaned_content column in batches
df['bart_summary'] = process_in_batches(dataloader)

# Save the updated dataframe
def save_data(df, filename):
    if df.empty:
        print(f"No data to save for {filename}")
        return
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"Data saved to {abs_path}")

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'articles_with_sentiments_and_summaries.csv'))
save_data(df, output_file)

# Display the first few summaries
print(df[['cleaned_content', 'bart_summary']].head())
