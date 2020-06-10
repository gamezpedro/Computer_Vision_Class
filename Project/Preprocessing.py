#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import movie_reviews, stopwords
from nltk import ngrams
import string

important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 
                   'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 
                   'very', 'just', 'but']

def clean_words(words, stopwords_language):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_language and word not in string.punctuation:
            words_clean.append(word)
    return words_clean

# for unigram
def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary

# for ngram (bigrams)
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words,n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    #print(words_dictionary)
    return words_dictionary

def bag_of_all_words(words, n=2):
    stopwords_english = stopwords.words('english')
    words_clean = clean_words(words, stopwords_english)
    words_clean_brigram = clean_words(words, set(stopwords_english) - set(important_words))

    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_brigram)

    all_features = unigram_features.copy()
    all_features.update(bigram_features)

    return all_features
