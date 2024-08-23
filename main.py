



# import pandas as pd
# import pytz
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from tqdm import tqdm
# import seaborn
# from datetime import datetime
# from matplotlib import pyplot as plt
# import ssl, os
# # Disable SSL verification
# ssl._create_default_https_context = ssl._create_unverified_context
#
# import nltk
# # Define the path to the vader_lexicon
# nltk_data_path = os.path.expanduser('~/nltk_data/sentiment/vader_lexicon')
#
# # Check if the vader_lexicon directory exists
# if not os.path.exists(nltk_data_path):
#     nltk.download('vader_lexicon')
# else:
#     print('vader_lexicon already downloaded.')
# # Load data from disk
# # Path to the cached file
# cached_file = './twcs/twcs.parquet'
#
# # Check if the cached file exists
# if os.path.exists(cached_file):
#     tweets = pd.read_parquet(cached_file)
# else:
#     tweets = pd.read_csv('./twcs/twcs.csv')
#     tweets.to_parquet(cached_file)
#
#################
# import pandas as pd
# import re
# import os
#
# # Load the CSV file into a DataFrame
# source_file = './twcs/twcs.csv'
# df = pd.read_csv(source_file)
#
# # Extract the directory from the source file path
# directory = os.path.dirname(source_file)
#
# # Specify the author_id you're interested in (e.g., 'sprintcare')
# # author_id = 'sprintcare'
# author_id = 'AmazonHelp'
#
# # Filter the DataFrame to get only the responses from the specified support author_id
# support_responses = df[(df['author_id'] == author_id) & (df['inbound'] == False)]
#
# # Function to remove anonymized customer IDs and support initials from the text
# def clean_text(text):
#     # Remove anonymized IDs like @12345
#     text = re.sub(r'@\d+', '', text)
#     # Remove support initials such as -AB, --RB, - CE, -LNJ, AL, -Y.F. at the end of the text
#     text = re.sub(r'(\s*-{1,2}\s?[A-Z]{2,4}\.?$)|(\s*[A-Z]{2,4}\.?$)', '', text.strip())
#     return text
# # Apply the function to the 'text' column
# # support_responses['text_cleaned'] = support_responses['text'].apply(remove_anonymized_ids)
#
# # Apply the function to the 'text' column using .loc to avoid the warning
# support_responses.loc[:, 'text_cleaned'] = support_responses['text'].apply(clean_text)
#
# # Define the output file path using the author_id
# output_file = os.path.join(directory, f'{author_id}_responses.csv')
#
# # Save the cleaned responses to a new CSV file
# support_responses[['tweet_id', 'created_at', 'text_cleaned']].to_csv(output_file, index=False)
#
#
# print(f'Support responses saved to {output_file}')

##########################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# Load the cleaned text data
support_responses = pd.read_csv('./twcs/AmazonHelp_responses.csv')
print("Here 1")
# Remove rows with NaN values in the 'text_cleaned' column
support_responses = support_responses.dropna(subset=['text_cleaned'])
print("Here 2")
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(support_responses['text_cleaned'])
print("Here 3")
# Apply Gaussian Mixture Model for clustering
n_clusters = 20  # Adjust the number of clusters as needed

# gmm = GaussianMixture(n_components=n_clusters, random_state=42)
# support_responses['cluster'] = gmm.fit_predict(X.toarray())

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
support_responses['cluster'] = kmeans.fit_predict(X.toarray())


print("Here 4")
# Find the centroid for each cluster and the most representative sentence
centroids = []
representative_sentences = []

for i in tqdm(range(n_clusters), desc="Processing Clusters"):
    cluster_indices = support_responses['cluster'] == i
    cluster_vectors = X.toarray()[cluster_indices]

    # Calculate the centroid for the cluster
    centroid = np.mean(cluster_vectors, axis=0)
    centroids.append(centroid)

    # Calculate cosine similarity between the centroid and all sentences in the cluster
    similarities = cosine_similarity([centroid], cluster_vectors)

    # Find the index of the most representative sentence
    most_representative_idx = np.argmax(similarities)
    representative_sentence = support_responses.loc[cluster_indices, 'text_cleaned'].iloc[most_representative_idx]

    representative_sentences.append(representative_sentence)
print("Here 5")
# Print the most representative sentence for each cluster
for idx, sentence in enumerate(representative_sentences):
    print(f'Cluster {idx} Representative Sentence: {sentence}')

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Plot the clusters with PCA
plt.figure(figsize=(10, 7))
for i in range(n_clusters):
    plt.scatter(X_pca[support_responses['cluster'] == i, 0],
                X_pca[support_responses['cluster'] == i, 1],
                label=f'Cluster {i}', alpha=0.5)

# Annotate the centroids with the most representative sentence
for i, centroid in enumerate(centroids):
    centroid_pca = pca.transform([centroid])
    plt.annotate(f'Cluster {i}', (centroid_pca[0, 0], centroid_pca[0, 1]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, weight='bold')

plt.title('GMM Clustering of Support Text (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

##########







# # Get customer requests and company responses
# first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
#
# inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', right_on='in_response_to_tweet_id')
#
# # Filter for company responses only
# inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]
#
# # Enable progress reporting on `df.apply` calls
# tqdm.pandas()
#
# # Instantiate sentiment analyzer from NLTK, make helper function
# sentiment_analyzer = SentimentIntensityAnalyzer()
#
# def sentiment_for(text: str) -> float:
#     return sentiment_analyzer.polarity_scores(text)['compound']
#
# # Analyze sentiment of inbound customer support requests
# inbounds_and_outbounds['inbound_sentiment'] = inbounds_and_outbounds.text_x.progress_apply(sentiment_for)
#
# # Plot top 20 brands by support volume
# author_grouped = inbounds_and_outbounds.groupby('author_id_y')
# top_support_providers = set(author_grouped.agg('count')
#                                 .sort_values(['tweet_id_x'], ascending=[0])
#                                 .index[:20]
#                                 .values)
#
# inbounds_and_outbounds \
#     .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)] \
#     .groupby('author_id_y') \
#     .tweet_id_x.count() \
#     .sort_values() \
#     .plot(kind='barh', title='Top 20 Brands by Volume')
#
# plt.show()  # Ensure that the plot is displayed
#
# # Plot customer sentiment by brand
# inbounds_and_outbounds \
#     .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)] \
#     .groupby('author_id_y') \
#     .inbound_sentiment.mean() \
#     .sort_values() \
#     .plot(kind='barh', title='Customer Sentiment by Brand (top 20 by volume)')
#
# plt.show()  # Ensure that the plot is displayed

