# -*- coding: utf-8 -*-
import random
import time

import pandas as pd

from nltk.corpora import stopwords

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from Preprocessing import preprocess, bigrams, trigrams

#       --- IMPORTING DATASET ---
# Using Sentiment140 dataset with 1.6 million tweets
# https://www.kaggle.com/kazanova/sentiment140/data
DATASET_COLUMNS = names = ['target', 'ids', 'date', 'flag', 'user', 'text']
tweets_raw = pd.read_csv(r'.\dataset\tweets.csv', sep = ',', quotechar ='"',encoding='latin-1', names=DATASET_COLUMNS)
tweets_raw['sentiment'] = tweets_raw['sentiment'].replace(4,'pos')
tweets_raw['sentiment'] = tweets_raw['sentiment'].replace(0,'neg')
tweets_raw, label = list(tweets_raw['text']), list(tweets_raw['sentiment'])

neg_tweets_raw = tweets_raw[:799999]
pos_tweets_raw = tweets_raw[800000:]

t = time.time()
pos_tweets, neg_words_ngrams = preprocess(pos_tweets_raw)
neg_tweets, pos_words_ngrams = preprocess(neg_tweets_raw)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')

while len(neg_tweets) > len(pos_tweets):
    del neg_tweets[0]
    del label[-1]
while len(pos_tweets) > len(neg_tweets):
    del pos_tweets[0]
    del label[0]

t = time.time()
bgram = bigrams(neg_words_ngrams,stopwords.words('english'))
tgram = trigrams(neg_words_ngrams,stopwords.words('english'))
print(f'Ngrams processing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')

tweets = neg_tweets + pos_tweets

label = []
for i in range(len(tweets)):
    if i < len(neg_tweets):
        label[i] = 'neg'
    else:
        label[i] = 'pos'

X_train, X_test, y_train, y_test = train_test_split(tweets, label, test_size = 0.05, random_state = 0)
print(f'Data Split done.')

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformed.')