#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Apoorva
#
# Created:     01/06/2016
# Copyright:   (c) Apoorva 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
import os, os.path, codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import numpy as np
from nltk.tokenize import RegexpTokenizer
import networkx as nx
from nltk.corpus import stopwords
import re

stop=[]
#new_stop=[hi]
f_stop = open("F:/ABCD/New folder/Offical/Mitacs/WORK/Data/SmartStoplist.txt","r")
for i in f_stop:
    stop.append(i.strip())
for i in range(2016):
    stop.append(str(i))

filename = "F:/ABCD/New folder/Offical/Mitacs/WORK/Data/try2_3_6_ToAbC.csv"

documents = []
classes = []

docs = csv.reader(open(filename, "rb"))
dataset = list(docs)
#print dataset[:5]
for doc in dataset[:3000]:
    #print "l", doc
    new = []
    for i in doc[0].split()+doc[1].split():
        if len(i)>3 or type(i):
            new.append(i)
    if int(doc[2])!=10:
        documents.append(" ".join(new))
        classes.append(int(doc[2]))

print len(documents)


tfidf = TfidfVectorizer(stop_words=stop ,ngram_range=tuple([1,3]), lowercase=True, strip_accents="unicode", use_idf=True, norm="l1",
                        min_df=2,max_df=10, max_features = 9000)
A = tfidf.fit_transform(documents)

num_terms = len(tfidf.vocabulary_)
terms = [""] * num_terms
for term in tfidf.vocabulary_.keys():
    terms[tfidf.vocabulary_[term]] = term
print "Created document-term matrix of size %d x %d" % (A.shape[0], A.shape[1])

fnames = tfidf.get_feature_names()
#print(fnames)
arr = A.toarray()


#import numpy as np
#X = np.random.randint(5, size=(6, 100))
X = arr
#y = np.array([1, 2, 3, 4, 5, 6])
y = np.array(classes)

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])


clf = GaussianNB()
clf.fit(X, y)
#print(clf.predict([[-0.8, -1]]))
print(clf.predict(X[:3000]),y[:3000])
print(clf.score(X[:3000]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
#print(clf_pf.predict([[-0.8, -1]]))



