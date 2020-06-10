#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
from Preprocessing import bag_of_all_words

classifier_f = open("pickled_algos/NaiveBayes_customdataset.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

youtube_comments = open("Comments/Starlink.txt", "r").read().split('\n')
random.shuffle(youtube_comments)
if youtube_comments[0] is not '':
    custom_review_tokens = word_tokenize(youtube_comments[0])
    custom_review_set = bag_of_all_words(custom_review_tokens)
    print(youtube_comments[0])
    print(classifier.classify(custom_review_set))
    prob_result = classifier.prob_classify(custom_review_set)
    print (prob_result.prob("neg")) 
    print (prob_result.prob("pos")) 

# custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
# custom_review_tokens = word_tokenize(custom_review)
# custom_review_set = bag_of_all_words(custom_review_tokens)
# print (classifier.classify(custom_review_set))
# # probability result
# prob_result = classifier.prob_classify(custom_review_set)
# # print (prob_result) 
# # print (prob_result.max()) 
# print (prob_result.prob("neg")) 
# print (prob_result.prob("pos")) 
# custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
# custom_review_tokens = word_tokenize(custom_review)
# custom_review_set = bag_of_all_words(custom_review_tokens)
# print (classifier.classify(custom_review_set)) 
# prob_result = classifier.prob_classify(custom_review_set)
# # print (prob_result)
# # print (prob_result.max())
# print (prob_result.prob("neg")) 
# print (prob_result.prob("pos"))