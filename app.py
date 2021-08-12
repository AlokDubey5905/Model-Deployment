# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:47:12 2021

@author: ALOK DUBEY
"""

from flask import Flask, render_template, url_for, request
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=['label', 'message'])
    # data cleaning and preprocessing

    ps = PorterStemmer()
    wordnet = WordNetLemmatizer()

    corpus = []
    for i in range(len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
        review = review.lower()
        review = review.split()

        review = [wordnet.lemmatize(word) for word in review if word not in set(
            stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)


    # creating the bag of words
    '''we got the size of X as (5572,6296) and 6296
    are the features so we change them and observe
    the change in accuracy.'''
    tf = TfidfVectorizer(max_features=5000)

    X = tf.fit_transform(corpus).toarray()
    y = pd.get_dummies(messages['label'])
    y = y['spam']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # training model using Naive bayes classifier
    spam_detect_model = MultinomialNB().fit(x_train, y_train)

    y_pred = spam_detect_model.predict(x_test)

    cf = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=tf.transform(data).toarray()
        my_prediction=spam_detect_model.predict(vect)
    return render_template('result.html',prediction=my_prediction)
