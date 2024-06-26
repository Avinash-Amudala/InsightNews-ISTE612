import pandas as pd
import os

def load_data(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"Found {len(files)} files to process.")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def preprocess_data(df):
    if df.empty:
        print("No data to preprocess.")
        return df
    initial_count = len(df)
    df.drop_duplicates(subset='url', keep='first', inplace=True)
    df.dropna(subset=['content'], inplace=True)
    df['content_length'] = df['content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    final_count = len(df)
    print(f"Preprocessed data from {initial_count} to {final_count} records.")
    return df

def save_data(df, filename):
    if df.empty:
        print(f"No data to save for {filename}")  # Debug: no data case
        return
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"Data saved to {abs_path}")

if __name__ == "__main__":
    raw_data_directory = 'data/raw/'
    processed_data_filename = 'data/processed/articles_cleaned.csv'

    print("Loading raw data...")
    df = load_data(raw_data_directory)
    if not df.empty:
        print("Preprocessing data...")
        df = preprocess_data(df)
        print("Saving cleaned data...")
        save_data(df, processed_data_filename)
    else:
        print("No data loaded. Skipping preprocessing and saving steps.")
    print("Data preprocessing complete.")
