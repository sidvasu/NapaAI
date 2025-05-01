import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

wine_df = pd.read_csv('data/winemag-data-130k-v2.csv')
wine_df.dropna(subset=['price'], inplace=True)
wine_df = wine_df.reset_index(drop=True)

def top50():
    return df.sort_values('points', ascending=False).head(50)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.model_selection import train_test_split

wine_df.loc[:, 'wineId'] = wine_df.loc[:, 'title'].astype('category').cat.codes
wine_df.loc[:, ['description', 'wineId', 'title']].head()

train_wine, test_wine = train_test_split(wine_df, train_size=0.05)

train_wine.reset_index(drop=True, inplace=True)

print(f"Training on {len(train_wine)} samples.")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df = 1, stop_words='english')

tfidf_matrix = tf.fit_transform(train_wine['description'])

print(f"The term-frequency inverse document frequency matrix is {tfidf_matrix.shape[0]} by {tfidf_matrix.shape[1]}")

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Next, appending the results to a dictionary of the similar items to each wine
results = {}
for idx, row in train_wine.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:100:-1]
    similar_items = [(cosine_similarities[idx][i], train_wine['wineId'][i]) for i in similar_indices]
    results[row['wineId']] = similar_items[1:]

def item(id):
    return train_wine.loc[train_wine['wineId'] == id]['title'].tolist()[0].split(' - ')[0]

def recommend(item_id, num):
    print('Recommending ' + str(num) + ' products similar to ' + item(item_id) + ' ...')
    print('-----')
    recs = results[item_id][:num]
    for rec in recs:
        print('Recommended: ' + item(rec[1]) + '(score: ' + f"{rec[0]:.2f}" + ')')

itemId_ = train_wine.loc[:, 'wineId'].values[0]
itemName_ = train_wine.loc[train_wine['wineId'] == itemId_, 'title'].values[0]

print(f"Using itemId {itemId_} which is {itemName_} \n")

# The recommend function is then run to find and return the top num matches (5 in this case)

recommend(item_id=itemId_, num=5)

results[itemId_][0][1]

description_original = train_wine.loc[train_wine['title'] == itemName_, 'description'].values[0]

description_matched = train_wine.loc[train_wine['wineId'] == results[itemId_][0][1], 'description'].values[0]

print(f"First wine description: \n{description_original} \n\nMatched wine description: \n{description_matched}")

'''
col = ['province','variety','points']
wine1 = wine[col]
wine1 = wine1.dropna(axis=0)
wine1 = wine1.drop_duplicates(['province','variety'])
wine1 = wine1[wine1['points'] >85]
wine_pivot = wine1.pivot(index= 'variety',columns='province',values='points').fillna(0)
wine_pivot_matrix = csr_matrix(wine_pivot)

knn = NearestNeighbors(n_neighbors=10,algorithm= 'brute', metric= 'cosine')
model_knn = knn.fit(wine_pivot_matrix)

query_index = np.random.choice(wine_pivot.shape[0])
distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0, len(distance.flatten())):
    if  i == 0:
        print('Recmmendation for {0}:\n'.format(wine_pivot.index[query_index]))
    else:
        print('{0}: {1} with distance: {2}'.format(i,wine_pivot.index[indice.flatten()[i]],distance.flatten()[i]))
'''