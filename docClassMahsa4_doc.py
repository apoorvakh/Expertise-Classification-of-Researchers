from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.datasets.base import Bunch
from sklearn.utils import check_random_state

import csv
import random
#import math

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
# Load some categories from the training set

filename = "data/try1_1_6_AbC.csv"

#filename = "ncs-AI.csv"

documents = []
classes = []

docsf = csv.reader(open(filename, "rb"))
dataset = list(docsf)
#print dataset[:5]
for doc in dataset:
    #print "l", doc
    documents.append(doc[0])
    classes.append((doc[1]))
'''
names=['AI','CL','DS','GR','CR','PF']
for n in names:
    filename = "data/ncs-"+n+".csv"
    docsf = csv.reader(open(filename, "rb"))
    dataset = list(docsf)
    #print dataset[:5]
    for doc in dataset:
        #print "l", doc
        documents.append(doc[1])
        classes.append((doc[2]))
'''
print("doc len : ", len(documents), len(classes))

def splitDataset(dataset, classes, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    trainC =[]
    copyC = list(classes)
    #print(" len : ", len(copy), len(copyC))
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        #print (index)
        trainSet.append(copy.pop(index))
        trainC.append(copyC.pop(index))
    print (len(trainSet),len(copy),len(trainC),len(copyC))
    return [trainSet, copy, trainC, copyC]

def get_data():

    ### Divide data into train and test sets
    splitRatio = 0.67
    trainingSet, testSet, trainClasses, testClasses = splitDataset(documents, classes, splitRatio)

    train_data = trainingSet
    train_target = np.array(trainClasses)
    train_target_names = list(set(classes))

    test_data = testSet
    test_target = np.array(testClasses)
    test_target_names = list(set(classes))

    if True : # shuffle
        random_state = check_random_state(42)
        indices = np.arange(train_target.shape[0])
        random_state.shuffle(indices)
        train_target = train_target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(train_data, dtype=object)
        data_lst = data_lst[indices]
        train_data = data_lst.tolist()

        indices = np.arange(test_target.shape[0])
        random_state.shuffle(indices)
        test_target = test_target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(test_data, dtype=object)
        data_lst = data_lst[indices]
        test_data = data_lst.tolist()

    return Bunch(data=train_data, target=np.array(train_target), target_names=train_target_names),\
           Bunch(data=test_data, target=np.array(test_target), target_names=test_target_names)


data_train, data_test = get_data()
print('data loaded')

categories = data_train.target_names    # for case categories == None
print(categories)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

print("%d documents - " % (len(data_train.data)))
print("%d documents - " % (len(data_test.data)))
print("%d categories" % len(categories))

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if False:#opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,ngram_range=tuple([1,3]),
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()

if False:#.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    if "Gaussain" in str(clf):
        print("hello")
        clf.fit(X_train.toarray(),y_train.toarray())
    else:
        clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if False:# or opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()
    if False: #or opts.print_report :
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))
    if False :#opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
'''
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))
# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))
'''

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
#results.append(benchmark(GaussianNB()))
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

# Combined Results

print("*************** Combined Results : ****************")
for x in results:
    print("%21s \t %6.3f "%(x[0],x[1]))