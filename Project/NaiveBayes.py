#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk
import random
import pickle
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize


# Bag on NGrams feature
# From: http://blog.chapagain.com.np/python-nltk-sentiment-analysis-on-movie-reviews-natural-language-processing-nlp/

# Comment when not using this dataset, 5532 comments each list
short_pos = open("reviews/positive.txt", encoding='latin-1').read()
short_neg = open("reviews/negative.txt", encoding='latin-1').read()

pos_reviews = []
# When using movie_reviews as dataset
# for fileid in movie_reviews.fileids('pos'):
#     pos_reviews.append(movie_reviews.words(fileid))
for pos_review in short_pos.split('\n'):
    pos_reviews.append(word_tokenize(pos_review)) 

neg_reviews = []
# When using movie_reviews as dataset:
# for fileid in movie_reviews.fileids('neg'):
#     neg_reviews.append(movie_reviews.words(fileid))
for neg_review in short_neg.split('\n'):
    neg_reviews.append(word_tokenize(neg_review)) 

pos_review_set = []
for words in pos_reviews:
    pos_review_set.append((bag_of_all_words(words), 'pos'))

neg_review_set = []
for words in neg_reviews:
    neg_review_set.append((bag_of_all_words(words), 'neg'))

random.shuffle(pos_review_set)
random.shuffle(neg_review_set)

test_set = pos_review_set[:1066] + neg_review_set[:1066]
train_set = pos_review_set[1066:] + neg_review_set[1066:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier,test_set)
print(accuracy)
print(classifier.show_most_informative_features(15))

save_classifier = open("pickled_algos/NaiveBayes_customdataset.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()