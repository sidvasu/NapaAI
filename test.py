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

'''
def recommend(train_df, results, item_id, num=5):
    print(f"\nRecommending {num} wines similar to: {item(train_df, item_id)}")
    print("------")
    recs = results[item_id][:num]
    for score, rec_id in recs:
        print(f"Recommended: {item(train_df, rec_id)} (score: {score:.2f})")
'''
def recommend(train_df, results, title_query, num=5):
    #print(f"\nRecommending {num} wines similar to: {item(train_df, item_id)}")
    #print("------")
    '''
    matches = train_df[train_df['title'].str.contains(title_query, case=False, na=False, regex=False)]

    if matches.empty:
        print("Not right")
        return

    match_row = matches.iloc[0]
    item_id = match_row['wineId']

    recs = results[item_id][:num]
    for score, rec_id in recs:
        print(f"Recommended: {item(train_df, rec_id)} (score: {score:.2f})")
    '''
    titles = train_df['title'].tolist()
    close_matches = difflib.get_close_matches(title_query, titles, n=1, cutoff=0.6)

    if not close_matches:
        print(f"No close match found for: '{title_query}'")
        return

    matched_title = close_matches[0]
    match_row = train_df[train_df['title'] == matched_title].iloc[0]
    wine_id = match_row['wineId']

    print(f"\nRecommending {num} wines similar to: '{title_query}'")
    print("------")

    recs = results.get(wine_id, [])[:num]
    for score, rec_id in recs:
        rec_title = train_df.loc[train_df['wineId'] == rec_id, 'title'].values[0]
        print(f"Recommended: {rec_title} (score: {score:.2f})")

def top50():
    wine_df = pd.read_csv('data/winemag-data-130k-v2.csv')
    return wine_df.sort_values('points', ascending=False).head(50)

# Run example
if __name__ == '__main__':
    tf, svd, tfidf_reduced, train_df, results = load_model()

    #item_id = train_df['wineId'].iloc[0]
    title_query = "Montevina 2011 Sauvignon Blanc (California)"
    recommend(train_df, results, title_query)

    #desc_orig = train_df.loc[train_df['wineId'] == item_id, 'description'].values[0]
    #desc_match = train_df.loc[train_df['wineId'] == results[item_id][0][1], 'description'].values[0]
    #print(f"\nOriginal Wine Description:\n{desc_orig}\n\nMatched Wine Description:\n{desc_match}")
