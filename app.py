from flask import Flask, render_template, redirect, url_for, request
import test
import os

app = Flask(__name__)

tf, svd, tfidf_reduced, train_df, results = test.load_model()

@app.route('/')
def home():
    wines = test.top50()
    return render_template("index.html",
        title = list(wines['title'].values),
        variety = list(wines['variety'].values),
        winery = list(wines['winery'].values),
        price = list(wines['price'].values)
        )

@app.route('/recommend', methods=["POST", "GET"])
def recommend():
    title_query = request.form.get('user_input')
    if title_query:
        recs = test.recommend(train_df, results, title_query)

        if (recs != -1):
            values = test.extractValues(train_df, recs)
    else:
        values = -1

    return render_template('recommender.html',
        results = values
        )

if __name__ == "__main__":
    app.run(debug=True)