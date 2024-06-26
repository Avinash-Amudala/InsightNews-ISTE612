import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load preprocessed data
df = pd.read_csv('data/processed/articles_cleaned.csv')

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# Distribution of article lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['content_length'], bins=30, kde=True)
plt.title('Distribution of Article Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.savefig('figures/article_length_distribution.png')
plt.show()

# Number of articles per topic
topic_counts = df['topic'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
plt.title('Number of Articles per Topic')
plt.xlabel('Topic')
plt.ylabel('Number of Articles')
plt.savefig('figures/articles_per_topic.png')
plt.show()

# Summary statistics
summary_stats = df.describe()
summary_stats.to_csv('figures/summary_statistics.csv')
print(summary_stats)

# Content Length Distribution by Topic
plt.figure(figsize=(12, 8))
sns.boxplot(x='topic', y='content_length', data=df)
plt.title('Content Length Distribution by Topic')
plt.xlabel('Topic')
plt.ylabel('Content Length (Number of Words)')
plt.savefig('figures/content_length_distribution_by_topic.png')
plt.show()

# Top sources for each topic
top_sources = df.groupby('topic')['source'].value_counts().groupby(level=0).head(5).reset_index(name='count')

plt.figure(figsize=(14, 10))
sns.barplot(x='count', y='source', hue='topic', data=top_sources, dodge=False)
plt.title('Top Sources for Each Topic')
plt.xlabel('Number of Articles')
plt.ylabel('Source')
plt.legend(title='Topic')
plt.tight_layout()
plt.savefig('figures/top_sources_per_topic.png')
plt.show()

