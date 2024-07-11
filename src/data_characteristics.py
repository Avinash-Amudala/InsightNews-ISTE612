from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/processed/articles_cleaned.csv')

os.makedirs('figures', exist_ok=True)

# Distribution of article lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['content_length'], bins=30, kde=True)
plt.title('Distribution of Article Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.savefig('figures/article_length_distribution.png')
plt.show()

topic_counts = df['topic'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=topic_counts.index, y=topic_counts.values,
            palette='viridis', hue=topic_counts.index, dodge=False)
plt.title('Number of Articles per Topic')
plt.xlabel('Topic')
plt.ylabel('Number of Articles')
plt.legend(title='Topic', loc='upper right')
plt.savefig('figures/articles_per_topic.png')
plt.show()

summary_stats = df.describe()
summary_stats.to_csv('figures/summary_statistics.csv')
print(summary_stats)

plt.figure(figsize=(12, 8))
sns.boxplot(x='topic', y='content_length', data=df)
plt.title('Content Length Distribution by Topic')
plt.xlabel('Topic')
plt.ylabel('Content Length (Number of Words)')
plt.savefig('figures/content_length_distribution_by_topic.png')
plt.show()

top_sources = df.groupby('topic')['source'].value_counts().groupby(
    level=0).head(5).reset_index(name='count')

plt.figure(figsize=(14, 10))
sns.barplot(x='count', y='source', hue='topic', data=top_sources, dodge=False)
plt.title('Top Sources for Each Topic')
plt.xlabel('Number of Articles')
plt.ylabel('Source')
plt.legend(title='Topic', loc='upper right')
plt.tight_layout()
plt.savefig('figures/top_sources_per_topic.png')
plt.show()


def tokenize_content(content):
    return content.split()


df['tokens'] = df['cleaned_content'].apply(tokenize_content)


def plot_term_frequency(df, topic, top_n=20):
    all_words = [word for tokens in df[df['topic'] == topic]['tokens']
                 for word in tokens]
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(top_n)

    words_df = pd.DataFrame(common_words, columns=['word', 'count'])

    print(f"Top {top_n} terms in {topic.capitalize()} articles:")
    print(words_df)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='word', data=words_df, palette='viridis')
    plt.title(f'Top {top_n} Terms in {topic.capitalize()} Articles')
    plt.xlabel('Frequency')
    plt.ylabel('Term')
    plt.savefig(f'figures/top_terms_{topic}.png')
    plt.show()


def generate_word_cloud(df, topic):
    all_words = ' '.join(
        [word for tokens in df[df['topic'] == topic]['tokens'] for word in tokens])
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(all_words)

    print(f"Generated word cloud for {topic.capitalize()} articles.")

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {topic.capitalize()} Articles')
    plt.savefig(f'figures/wordcloud_{topic}.png')
    plt.show()


def plot_tfidf_ngrams(df, topic, ngram_range=(1, 3), top_n=20):
    text_data = df[df['topic'] == topic]['cleaned_content']
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=ngram_range, max_features=top_n)
    X = vectorizer.fit_transform(text_data)
    tfidf_scores = X.toarray().sum(axis=0)
    ngrams = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame({'ngram': ngrams, 'tfidf': tfidf_scores}).sort_values(
        by='tfidf', ascending=False)

    print(
        f"Top {top_n} TF-IDF {ngram_range[1]}-grams in {topic.capitalize()} articles:")
    print(tfidf_df)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='tfidf', y='ngram', data=tfidf_df, palette='viridis')
    plt.title(
        f'Top {top_n} TF-IDF {ngram_range[1]}-grams in {topic.capitalize()} Articles')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('N-gram')
    plt.savefig(f'figures/top_tfidf_{ngram_range[1]}grams_{topic}.png')
    plt.show()


topics = df['topic'].unique()
for topic in topics:
    plot_term_frequency(df, topic)
    generate_word_cloud(df, topic)
    plot_tfidf_ngrams(df, topic, ngram_range=(2, 2),
                      top_n=20)  # TF-IDF Bigrams
    plot_tfidf_ngrams(df, topic, ngram_range=(3, 3),
                      top_n=20)  # TF-IDF Trigrams
