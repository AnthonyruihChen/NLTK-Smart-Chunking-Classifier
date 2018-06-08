# Import libraries
import csv
import re
import nltk
import codecs
import numpy as np
import math
import random
from itertools import groupby
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# 3 Supervised Classification Methods using NLTK:
    # 1) Last Word of a Main Bullet Point
    # 2) Keyword Frequency Distribution in Main Bullet Point
    # 3) Phrase Classification in Main Bullet Point



# Reads in CSV File and inserts csv rows into a list of lists
def retrieve_data(csv_name):
    with codecs.open(csv_name, "r",encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=','))
    return data #returns a list of lists


# Splits the dataset into training and test data by a ratio specified by the user
def split_dataset(dataset, split_ratio):
    trainSize = int(len(dataset) * split_ratio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]


# Isolates Main Bullets that have associated sub-bullets and returns only those rows for feature extraction
def break_by_bullets(dataset):
    bullet_w_subs = []
    for key, chapter in groupby (dataset, lambda x: x[0]): # groups by chapter
        for key, bullet in groupby(chapter, lambda x: str((x[1])).split(".", 1)[0]): # groups by bullet
            count = 0
            main_bullet = []
            for line in bullet:
                if int(line[1].split(".", 1)[1]) == 0:
                    main_bullet.append(line)
                count += 1
            if count > 1:
                # print count
                # print main_bullet
                bullet_w_subs.append(main_bullet)
            # print ("-----")
    return bullet_w_subs

# Feature Extractor: get last word in 'Bullet' items where the sub-bullets are 'single requirements'
# NOTE: Keeps any final punctuation (:, ;, etc.)
def extract_features_lastword(dataset):
    keywords = []
    for line in dataset:
        for k in line:
            last_word = k[2].split(' ')[-1]
            ok = {'last word': last_word}
            ik = k[3]
            tup = (ok, ik)
            keywords.append(tup)
    # print keywords
    return keywords # list(tup(dictionary, value))


# Keyword Frequency Naive Bayes Classifier & Accuracy Score
def keyword_frequency(dataset):
    keywords2 = []
    tokenizer = RegexpTokenizer(r'\w+')
    for l in dataset:
        for r in l:
            classy = r[3]
            words = tokenizer.tokenize(r[2])
            filtered_words = [w for w in words if w not in stopwords.words('english')] # Gets rid of 'filler words aka: a, and, it
            for w in filtered_words:
                features = {'keyword': w}
                tup = (features,classy)
                # print tup
                keywords2.append(tup)
    return keywords2

# Applying the Naive Bayes Classifier & Accuracy Score
# input: [dictionary{feature, value}, class]
def apply_classifier(trainset, testset, type):
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    print (type + ' Classifier Accuracy Score:')
    print(nltk.classify.accuracy(classifier, testset))
    classifier.show_most_informative_features(5)
    print ("--------------")



# MAIN PROGRAM
# Getting the data into the correct structure
train = retrieve_data('MIFID NLTK Ready.csv')
train_main_bullets_subs = break_by_bullets(train)

test = retrieve_data('Full Swedish NLTK Ready.csv')
test_main_bullets_subs = break_by_bullets(test)

# Classifier # 1: Last Word
train_keyword_tup = extract_features_lastword(train_main_bullets_subs)
test_keyword_tup = extract_features_lastword(test_main_bullets_subs)
apply_classifier(train_keyword_tup, test_keyword_tup, 'Last Word') #Prints out accuracy score

# Classifier # 2: Keyword Frequency
training_data = keyword_frequency(train_main_bullets_subs)
testing_data = keyword_frequency(test_main_bullets_subs)
apply_classifier(training_data, testing_data, 'Keyword') #Prints out accuracy score

