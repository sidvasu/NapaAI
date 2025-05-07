import joblib
import numpy as np
import pandas as pd
import difflib

def load_model(folder='wine_model'):
    tfidf_vectorizer = joblib.load(f'{folder}/tfidf_vectorizer.joblib')
    svd_model = joblib.load(f'{folder}/svd_model.joblib')
    tfidf_reduced = np.load(f'{folder}/tfidf_reduced.npy')
    train_df = joblib.load(f'{folder}/train_df.joblib')
    results_dict = joblib.load(f'{folder}/results_dict.joblib')
    print(f"Model loaded from '{folder}'")
    return tfidf_vectorizer, svd_model, tfidf_reduced, train_df, results_dict

def item(train_df, wine_id):
    return train_df.loc[train_df['wineId'] == wine_id]['title'].values[0].split(' - ')[0]

def recommend(train_df, results, title_query, num=5):
    titles = train_df['title'].tolist()
    close_matches = difflib.get_close_matches(title_query, titles, n=1, cutoff=0.6)

    if not close_matches:
        print(f"No close match found for: '{title_query}'")
        return -1

    matched_title = close_matches[0]
    match_row = train_df[train_df['title'] == matched_title].iloc[0]
    wine_id = match_row['wineId']

    print(f"\nRecommending {num} wines similar to: '{title_query}'")
    print("------")

    recs = results.get(wine_id, [])[:num]

    return recs

def extractValues(train_df, recs):
    values = []
    for score, rec_id in recs:
        rec_title = train_df.loc[train_df['wineId'] == rec_id, 'title'].values[0]
        rec_variety = train_df.loc[train_df['wineId'] == rec_id, 'variety'].values[0]
        rec_winery = train_df.loc[train_df['wineId'] == rec_id, 'winery'].values[0]
        rec_price = train_df.loc[train_df['wineId'] == rec_id, 'price'].values[0]

        values.append([rec_title, rec_variety, rec_winery, rec_price])

    return values

def top50():
    wine_df = pd.read_csv('data/winemag-data-130k-v2.csv')
    return wine_df.sort_values('points', ascending=False).head(50)

# Run example
if __name__ == '__main__':
    tf, svd, tfidf_reduced, train_df, results = load_model()

    title_query = "Montevina 2011 Sauvignon Blanc (California)"
    recommend(train_df, results, title_query)
