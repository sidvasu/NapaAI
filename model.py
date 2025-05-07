# train_and_save_model.py
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import os

def save_model(tfidf_vectorizer, svd_model, tfidf_reduced, train_df, results_dict, folder='wine_model'):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(tfidf_vectorizer, f'{folder}/tfidf_vectorizer.joblib')
    joblib.dump(svd_model, f'{folder}/svd_model.joblib')
    np.save(f'{folder}/tfidf_reduced.npy', tfidf_reduced)
    joblib.dump(train_df, f'{folder}/train_df.joblib')
    joblib.dump(results_dict, f'{folder}/results_dict.joblib')
    print(f"Model saved to '{folder}'.")

def main():
    # Load and prepare data
    wine_df = pd.read_csv('data/winemag-data-130k-v2.csv')
    wine_df.dropna(subset=['price', 'description', 'variety'], inplace=True)
    wine_df = wine_df.drop_duplicates(subset=['title'])
    wine_df.reset_index(drop=True, inplace=True)
    wine_df['wineId'] = wine_df['title'].astype('category').cat.codes
    wine_df['enriched_description'] = wine_df['variety'] + ' ' + wine_df['description']

    train_df, _ = train_test_split(wine_df, train_size=0.2, random_state=42)
    train_df.reset_index(drop=True, inplace=True)

    print(f"Training on {len(train_df)} samples...")

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.8, stop_words='english', sublinear_tf=True)
    tfidf_matrix = tf.fit_transform(train_df['enriched_description'])

    svd = TruncatedSVD(n_components=100, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_reduced)

    distances, indices = knn_model.kneighbors(tfidf_reduced, n_neighbors=6)

    results = {}
    for idx, (dist_list, neighbor_indices) in enumerate(zip(distances, indices)):
        wine_id = train_df.loc[idx, 'wineId']
        similar_items = [(1 - dist_list[i], train_df.loc[neighbor_indices[i], 'wineId']) 
                        for i in range(1, len(neighbor_indices))]  # skip the first one (self)
        results[wine_id] = similar_items


    save_model(tf, svd, tfidf_reduced, train_df, results)

if __name__ == '__main__':
    main()