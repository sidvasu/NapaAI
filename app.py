from flask import Flask, render_template, redirect, url_for, request
import test
import os

app = Flask(__name__)

@app.route('/')
def home():
    wines = test.top50()
    return render_template("index.html",
        title = list(wines['title'].values),
        variety = list(wines['variety'].values),
        winery = list(wines['variety'].values),
        price = list(wines['price'].values)
        )

if __name__ == "__main__":
    app.run(debug=True)