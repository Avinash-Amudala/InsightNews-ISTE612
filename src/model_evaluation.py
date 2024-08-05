import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px

data_path = os.path.abspath(os.path.join(os.path.dirname(
    __file__), '..', 'data', 'processed', 'articles_with_sentiment.csv'))

data = pd.read_csv(data_path)

X = data['topic']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)


classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)


num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_train_tfidf)

# Reduce dimensions using t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_train_tfidf.toarray())

# Create a DataFrame with t-SNE coordinates and cluster labels
df_tsne = pd.DataFrame(X_tsne, columns=['x', 'y'])
df_tsne['cluster'] = kmeans.labels_


fig = px.scatter(df_tsne, x='x', y='y', color='cluster',
                 title='K-means Clusters')
fig.show()


os.makedirs('models_for_eval', exist_ok=True)
joblib.dump(tfidf_vectorizer, 'models_for_eval/tfidf_vectorizer.pkl')
joblib.dump(classifier, 'models_for_eval/classifier.pkl')
joblib.dump(kmeans, 'models_for_eval/kmeans.pkl')


tfidf_vectorizer = joblib.load('models_for_eval/tfidf_vectorizer.pkl')
classifier = joblib.load('models_for_eval/classifier.pkl')
kmeans = joblib.load('models_for_eval/kmeans.pkl')

X_test_tfidf = tfidf_vectorizer.transform(X_test)


y_pred = classifier.predict(X_test_tfidf)


cm = confusion_matrix(y_test, y_pred)

# Get the unique labels from the actual data
cm_labels = sorted(y.unique())


fig = ff.create_annotated_heatmap(
    z=cm,
    x=cm_labels,
    y=cm_labels,
    colorscale='Viridis'
)

fig.update_layout(
    title_text='Confusion Matrix',
    xaxis=dict(title='Predicted'),
    yaxis=dict(title='Actual')
)


pio.write_image(fig, 'confusion_matrix.png')

report = classification_report(y_test, y_pred, target_names=cm_labels)
print(report)

silhouette_avg = silhouette_score(X_train_tfidf, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')
