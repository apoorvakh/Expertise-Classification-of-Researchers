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
filename = "F:/ABCD/New folder/Offical/Mitacs/WORK/Data/try1_1_6_AbC.csv"
documents = []
classes = []
'''
# Mahasa ::
docsf = csv.reader(open(filename, "rb"))
dataset = list(docsf)
#print dataset[:5]
for doc in dataset:
    #print "l", doc
    documents.append(doc[0])
    classes.append((doc[1]))
'''
# arXiv ::
#names=['AI','CC','CE','CL','CR','DS','GR','IR','LG','PF','SI']
names = ['IR','AI','CE','GR']
for n in names:
    filename = "data/ncs-"+n+".csv"
    docsf = csv.reader(open(filename, "rb"))
    dataset = list(docsf)
    #print dataset[:5]
    for doc in dataset[:600]:
        #print "l", doc
        documents.append(doc[0].lower()+'. '+doc[1].lower())
        classes.append(n)

print("doc len : ", len(documents), len(classes))

def separateByClass(documents,classes):
    separated = {}
    for c in list(set(classes)):
        separated[c] = []
    l = len(documents)
    #print(separated, l)
    for m in range(l):
        #print(m)
        vector = documents[m]
        if m<5:
            pass
            #print(vector)
        separated[classes[m]].append(vector)
    return separated



# Each author has papers from only one topic!
def getVirtualAuthors(docsInClass):
    vList = []
    vC=[]
    for c,d in docsInClass.iteritems():
        for i in range(0,len(d),5):
            vList.append(' '.join([a for a in d[i:i+5]]))
            vC.append(c)
    return vList, vC



#exit()
def splitDataset(dataset, classes, splitRatio):
    # split whole dataset into 2:1 :: train:test
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
    #print (len(trainSet),len(copy),len(trainC),len(copyC))
    return [trainSet, copy, trainC, copyC]

def get_data(documents, classes):
    ### Divide data into train and test sets

    # both the train and test are vAs
    def shuffle(data, target):
        random_state = check_random_state(42)
        indices = np.arange(target.shape[0])
        random_state.shuffle(indices)
        target =target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data, dtype=object)
        data_lst = data_lst[indices]
        data = data_lst.tolist()
        return data, target

    shuffle(documents, np.array(classes))
    docsInClass = separateByClass(documents,classes)
    # make sure every class is split as 2:1 :: train:test

    splitRatio = 0.67
    trainingSet=[]
    testSet=[]
    trainClasses=[]
    testClasses=[]
    # divide each class into 2:1
    for k in docsInClass.keys():
        trainD, testD, trainC, testC = splitDataset(docsInClass[k], [k]*len(docsInClass[k]), splitRatio)
        trainingSet.extend(trainD)
        testSet.extend(testD)
        trainClasses.extend(trainC)
        testClasses.extend(testC)
        #print("traaa",k,len(trainD),len(testD),len(trainC),len(testC))
    train_data = trainingSet
    train_target = np.array(trainClasses)
    train_target_names = list(set(classes))

    test_data = testSet
    test_target = np.array(testClasses)
    test_target_names = list(set(classes))

    shuffle(train_data, train_target)
    shuffle(test_data, test_target)

    # returning both train and test as docs ( test docs can be converted to vAs later)
    return Bunch(data=train_data, target=np.array(train_target), target_names=train_target_names),\
           Bunch(data=test_data, target=np.array(test_target), target_names=test_target_names)


#data_train, data_test = get_data(vAuthors, vAClasses)  # both are vA
data_train, data_test = get_data(documents, classes)  # train:doc and test:doc ; ...

# converting test data from doc representation to vA representation  ### train:doc and test:vA
print("doc:vA? (True/False)")
#B2 = input("doc:vA? (True/False)")
print("vA:vA? (True/False)")
#B3 = input("vA:vA? (True/False)")
if False and B2:    # train:doc and test:vA
    #data_train, data_test = get_data(documents, classes)
    #print("before",data_test.data[0])
    testDInC = separateByClass(data_test.data, data_test.target.tolist())
    data_test.data, data_test.target = getVirtualAuthors(testDInC)
    data_test.target = np.array(data_test.target)
    #print("after",data_test.data[0])
elif False and B3 :    # train:vA and test:vA
    docsInClass = separateByClass(documents,classes)
    vAuthors, vAClasses = getVirtualAuthors(docsInClass)
    data_train, data_test = get_data(vAuthors, vAClasses)

print('data loaded')

categories = data_train.target_names    # for case categories == None
print("The Categories are : ", categories)

print("Training :: %d documents - " % (len(data_train.data)))
print("Testing :: %d documents - " % (len(data_test.data)))
print("%d categories" % len(categories))

# ## split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("For training : Extracting features from the training data using a sparse vectorizer")
t0 = time()
if False: # or opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(lowercase=True, use_idf=True, sublinear_tf=True, max_df=0.5,ngram_range=tuple([1,3]),
                                 stop_words='english')

    # ## add semantics here !!!
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs" % (duration))#, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("For testing : Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs" % (duration))#, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()

select_chi2=100000
print("chi-2 ? (True/False)")
#Bchi=input("chi-2 ? (True/False)")
if True or Bchi:#opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

###############################################################################
# Benchmark classifiers
# build n-models for each classifier
# a test sample is given to all n-models.. each model classifies it and each model's accuracy is calculated separately and averaged to get the accuracy of the whole model
# ## ! when a test sample is given it does NOT get the list of probabilities from all classifiers ; it does get to know whether it belongs to a class or not; all bulk tests are given to a model and it is tested
# each model takes in the test and gives out accuracy
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
    if True: #or opts.print_report :
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))
    if True :#opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []

'''
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
#results.append(benchmark(BernoulliNB(alpha=.01)))

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



"""
tdNo =0
    for n,t in enumerate(X_test):
        tdNo+=1
        print("For test doc ",tdNo)
        cumPred=[]
        cumPPred = []
        cumPLPred = []
        t0 = time()
        for c in classifiers.values():
            #print(type(c))
            #print("Classifier : ",c)
            pred = c.predict(t)
            ppred=c.predict_proba(t)
            plpred=c.predict_log_proba(t)
            cumPred.append(pred)
            cumPPred.append(ppred[0])
            #print("Rank in this class : ",ppred[0])
            cumPLPred.append(plpred)
        #print("Prediction : ",pred," (Actual : ",new_y_test[n],' :: Correct )' if pred==y_test[n] else 'Wrong )')
        for j,c in enumerate(classifiers):
            print(c," - ",cumPred[j], cumPPred[j])
        #print(plpred)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
        cumTestTime+=test_time
"""