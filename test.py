# load_and_test_model.py
import joblib
import numpy as np

def load_model(folder='wine_model'):
    tfidf_vectorizer = joblib.load(f'{folder}/tfidf_vectorizer.joblib')
    svd_model = joblib.load(f'{folder}/svd_model.joblib')
    tfidf_reduced = np.load(f'{folder}/tfidf_reduced.npy')
    train_df = joblib.load(f'{folder}/train_df.joblib')
    results_dict = joblib.load(f'{folder}/results_dict.joblib')
    print(f"Model loaded from '{folder}'")
    return tfidf_vectorizer, svd_model, tfidf_reduced, train_df, results_dict

# Utility functions
def item(train_df, wine_id):
    return train_df.loc[train_df['wineId'] == wine_id]['title'].values[0].split(' - ')[0]

def recommend(train_df, results, item_id, num=5):
    print(f"\nRecommending {num} wines similar to: {item(train_df, item_id)}")
    print("------")
    recs = results[item_id][:num]
    for score, rec_id in recs:
        print(f"Recommended: {item(train_df, rec_id)} (score: {score:.2f})")

# Run example
if __name__ == '__main__':
    tf, svd, tfidf_reduced, train_df, results = load_model()

    item_id = train_df['wineId'].iloc[0]
    recommend(train_df, results, item_id)

    desc_orig = train_df.loc[train_df['wineId'] == item_id, 'description'].values[0]
    desc_match = train_df.loc[train_df['wineId'] == results[item_id][0][1], 'description'].values[0]
    print(f"\nOriginal Wine Description:\n{desc_orig}\n\nMatched Wine Description:\n{desc_match}")
