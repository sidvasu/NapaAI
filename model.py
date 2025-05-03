import numpy as np
import pandas as pd
import os
import joblib
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.model_selection import train_test_split

wine_df = pd.read_csv('data/winemag-data-130k-v2.csv')
wine_df.dropna(subset=['price', 'description', 'variety'], inplace=True)
wine_df = wine_df.reset_index(drop=True)

def top50():
    return df.sort_values('points', ascending=False).head(50)

# Create wineId as a categorical code
wine_df['wineId'] = wine_df['title'].astype('category').cat.codes

# Enrich the description with variety (optional but often helpful)
wine_df['enriched_description'] = wine_df['variety'] + ' ' + wine_df['description']

# Use a sample for faster prototyping
train_wine, _ = train_test_split(wine_df, train_size=0.05, random_state=42)
train_wine.reset_index(drop=True, inplace=True)

print(f"Training on {len(train_wine)} samples.")

# TF-IDF with optimized settings
tf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    stop_words='english',
    sublinear_tf=True
)

tfidf_matrix = tf.fit_transform(train_wine['enriched_description'])

# Apply Truncated SVD to reduce dimensionality (LSA)
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Compute cosine similarity matrix
cosine_similarities = linear_kernel(tfidf_reduced, tfidf_reduced)

# Create results dictionary
results = {}
for idx, row in train_wine.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:100:-1]
    similar_items = [(cosine_similarities[idx][i], train_wine['wineId'][i]) for i in similar_indices]
    results[row['wineId']] = similar_items[1:]

# Utility functions
def item(id):
    return train_wine.loc[train_wine['wineId'] == id]['title'].tolist()[0].split(' - ')[0]

def recommend(item_id, num):
    print('Recommending ' + str(num) + ' products similar to ' + item(item_id) + ' ...')
    print('-----')
    recs = results[item_id][:num]
    for rec in recs:
        print('Recommended: ' + item(rec[1]) + ' (score: ' + f"{rec[0]:.2f}" + ')')

'''
def save_model(tfidf_vectorizer, svd_model, tfidf_reduced, train_wine_df, results_dict, folder='wine_model'):
    os.makedirs(folder, exist_ok=True)
    
    joblib.dump(tfidf_vectorizer, f'{folder}/tfidf_vectorizer.joblib')
    joblib.dump(svd_model, f'{folder}/svd_model.joblib')
    np.save(f'{folder}/tfidf_reduced.npy', tfidf_reduced)
    joblib.dump(train_wine_df, f'{folder}/train_wine_df.joblib')
    joblib.dump(results_dict, f'{folder}/results_dict.joblib')
    
    print(f"Model saved to folder: {folder}")

def load_model(folder):
    tfidf_vectorizer = joblib.load(f'{folder}/tfidf_vectorizer.joblib')
    svd_model = joblib.load(f'{folder}/svd_model.joblib')
    tfidf_reduced = np.load(f'{folder}/tfidf_reduced.npy')
    train_wine_df = joblib.load(f'{folder}/train_wine_df.joblib')
    results_dict = joblib.load(f'{folder}/results_dict.joblib')
    
    print(f"Model loaded from folder: {folder}")
    return tfidf_vectorizer, svd_model, tfidf_reduced, train_wine_df, results_dict

# Saving
save_model(tf, svd, tfidf_reduced, train_wine, results)
tf, svd, tfidf_reduced, train_wine, results = load_model('wine_model')
'''

# Run a test
itemId_ = train_wine['wineId'].values[0]
itemName_ = train_wine.loc[train_wine['wineId'] == itemId_, 'title'].values[0]
print(f"\nUsing itemId {itemId_} which is '{itemName_}'\n")

recommend(item_id=itemId_, num=5)

# Show comparison of descriptions
description_original = train_wine.loc[train_wine['title'] == itemName_, 'description'].values[0]
description_matched = train_wine.loc[train_wine['wineId'] == results[itemId_][0][1], 'description'].values[0]

print(f"\nOriginal Wine Description:\n{description_original}\n\nMatched Wine Description:\n{description_matched}")