from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
import numpy as np
import nltk
import pickle
import re
import contractions
from nltk.tokenize import RegexpTokenizer
import unicodedata
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

spacy_nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/movie_review_sentiment', methods=['GET'])
def movie_review_sentiment():
    return render_template('movie_review_sentiment.html')


@app.route('/predict', methods=['POST'])
def predict():
    # GETTING REQUEST
    review = None
    if request.method == "POST":
        review = request.get_json('data')

    # LOADING PKL FILE
    model_file = 'log_model.pkl'
    transforms_file = 'tfidf_transformer.pkl'
    model = pickle.load(open(model_file, 'rb'))
    transformer = pickle.load(open(transforms_file, 'rb'))

    data = [review]

    # CLEANING
    df = pd.DataFrame(data, columns=['review'])
    df.review = df.review.apply(lambda x: x.strip())
    df.review = df.review.apply(lambda x: re.sub(' +', ' ', x))
    df.review = df.review.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    df.review = df.review.apply(lambda x: contractions.fix(x))
    onlyWorkdstokenizer = RegexpTokenizer(r'\w+')
    df.review = df.review.apply(lambda x: " ".join(onlyWorkdstokenizer.tokenize(x)))
    df.review = df.review.apply(
        lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    df.review = df.review.apply(lambda x: x.lower())
    df.review = df.review.apply(lambda x: ' '.join([w for w in x.split() if w not in STOP_WORDS]))
    df.review = df.review.apply(lambda x: " ".join([token.lemma_ for token in spacy_nlp(x)]))

    # PREDICTING SENTIMENT
    vector = transformer.transform(df.review).toarray()
    proba = model.predict_proba(vector)
    proba = np.round(proba * 100, 0).tolist()
    prediction = model.predict(vector).tolist()

    # Feature Importance
    feature_importance = zip(transformer.get_feature_names(), model.coef_[0])
    feature_importance = dict(feature_importance)
    cleaned_review = df.review.iloc[0].split()
    important_words = {}
    for t in cleaned_review:
        if t in feature_importance:
            important_words[t] = feature_importance[t]
    sortReverse = False
    if prediction[0] == 1:
        sortReverse = True
    important_words = dict(sorted(important_words.items(), key=lambda kv: kv[1], reverse=sortReverse)[:10])

    # JSON
    review_prediction = {
        'predict': prediction[0],
        'proba': proba[0],
        'review_cleaned': df.review.tolist(),
        'important_words': important_words
    }

    return jsonify(review_prediction)


if __name__ == '__main__':
    app.run()
