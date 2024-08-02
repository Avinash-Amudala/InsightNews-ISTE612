import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from collections import defaultdict
import gc
from nltk.stem import WordNetLemmatizer
import nltk
import joblib  # For saving models
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

def load_data(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"Found {len(files)} files to process.")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def extract_source_name(source):
    if pd.isna(source):
        return None
    try:
        source_dict = eval(source)
        if isinstance(source_dict, dict):
            return source_dict.get('name')
    except:
        return source
    return None

def preprocess_data(df):
    if df.empty:
        print("No data to preprocess.")
        return df

    initial_count = len(df)
    print(f"Initial record count: {initial_count}")

    df.drop_duplicates(subset='url', keep='first', inplace=True)
    pattern = re.compile(r'image \d+ of \d+', re.IGNORECASE)
    df = df[~df['title'].apply(lambda x: bool(pattern.search(x)) if pd.notnull(x) else False)]
    after_dedup_count = len(df)
    print(f"Records after removing duplicates: {after_dedup_count} (Removed {initial_count - after_dedup_count})")

    df = df.dropna(subset=['title', 'content'], how='all')
    after_dropna_count = len(df)
    print(f"Records after removing rows with missing title and content: {after_dropna_count} (Removed {after_dedup_count - after_dropna_count})")

    df['content'] = df['content'].fillna(df['title'])
    df['title'] = df['title'].fillna(df['content'])

    # Extract source names
    df['source'] = df['source'].apply(extract_source_name)
    
    # Save the unique source names to a CSV file
    unique_sources = df['source'].dropna().unique()
    unique_sources_df = pd.DataFrame(unique_sources, columns=['source'])
    os.makedirs('data/processed/', exist_ok=True)
    unique_sources_df.to_csv('data/processed/unique_sources.csv', index=False)
    print(f"Saved unique source names to 'data/processed/unique_sources.csv'")
    
    # Function to combine published_at and publishedAt columns
    def combine_published_at(df):
        df['published_at_combined'] = df['published_at'].combine_first(df['publishedAt'])
        df.drop(columns=['published_at', 'publishedAt'], inplace=True)
        df.rename(columns={'published_at_combined': 'published_at'}, inplace=True)
        return df
    
    # Combine published_at and publishedAt columns
    df = combine_published_at(df)

    # Convert published_at to datetime
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

    # Sort by latest date in descending order
    df = df.sort_values(by='published_at', ascending=False)

    stop_words = set(stopwords.words('english'))

    def clean_content(content):
        tokens = word_tokenize(content.lower())
        tokens = [word for word in tokens if word.isalpha()]  # Keep all words, including stop words
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    df['cleaned_content'] = df['content'].apply(clean_content)
    df['content_length'] = df['cleaned_content'].apply(lambda x: len(x.split()))

    df = df[df['content_length'] > 3]
    after_cleaning_count = len(df)
    print(f"Records after cleaning content: {after_cleaning_count} (Removed {after_dropna_count - after_cleaning_count})")

    # Drop duplicates based on the 'cleaned_content' column
    df = df.drop_duplicates('cleaned_content')

    final_count = len(df)
    print(f"Preprocessed data from {initial_count} to {final_count} records.")

    return df

def create_boolean_incidence_matrix(df):
    vectorizer = CountVectorizer(binary=True)
    incidence_matrix = vectorizer.fit_transform(df['cleaned_content'])
    print(f"Boolean Incidence Matrix shape: {incidence_matrix.shape}")
    return incidence_matrix, vectorizer

def create_inverted_index(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned_content'])
    inverted_index = defaultdict(list)
    for word, idx in vectorizer.vocabulary_.items():
        inverted_index[word] = X[:, idx].nonzero()[0].tolist()
    print(f"Inverted Index created with {len(inverted_index)} terms.")
    return inverted_index

def vector_space_model(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_content'])
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer

def text_classification(df, vectorizer):
    df = df.dropna(subset=['cleaned_content', 'category'])
    X = vectorizer.fit_transform(df['cleaned_content'])
    y = df['category']
    X = X[~y.isna()]
    y = y.dropna()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    return classifier

def text_clustering(df, vectorizer):
    X = vectorizer.fit_transform(df['cleaned_content'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    print(f"Cluster distribution:\n{df['cluster'].value_counts()}")
    return kmeans

def save_data(df, filename):
    if df.empty:
        print(f"No data to save for {filename}")
        return
    abs_path = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"Data saved to {abs_path}")

if __name__ == "__main__":
    raw_data_directory = 'data/raw/'
    processed_data_filename = 'data/processed/articles_cleaned.csv'
    tfidf_vectorizer_filename = 'models/tfidf_vectorizer.pkl'
    tfidf_matrix_filename = 'models/tfidf_matrix.pkl'
    classifier_filename = 'models/classifier.pkl'
    kmeans_filename = 'models/kmeans.pkl'

    print("Loading raw data...")
    df = load_data(raw_data_directory)
    if not df.empty:
        print("Preprocessing data...")
        df = preprocess_data(df)

        print("Creating Boolean Incidence Matrix...")
        incidence_matrix, incidence_vectorizer = create_boolean_incidence_matrix(df)

        print("Creating Inverted Index...")
        inverted_index = create_inverted_index(df)

        print("Applying Vector Space Model...")
        tfidf_matrix, tfidf_vectorizer = vector_space_model(df)

        print("Performing Text Classification...")
        classifier = text_classification(df, tfidf_vectorizer)

        print("Performing Text Clustering...")
        kmeans = text_clustering(df, tfidf_vectorizer)

        print("Saving cleaned data...")
        save_data(df, processed_data_filename)

        print("Saving models...")
        # Ensure the 'models' directory exists
        os.makedirs(os.path.dirname(tfidf_vectorizer_filename), exist_ok=True)
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_filename)
        joblib.dump(tfidf_matrix, tfidf_matrix_filename)
        joblib.dump(classifier, classifier_filename)
        joblib.dump(kmeans, kmeans_filename)

    else:
        print("No data loaded. Skipping preprocessing and saving steps.")
    print("Data preprocessing complete.")
