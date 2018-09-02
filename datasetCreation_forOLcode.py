#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Apoorva
#
# Created:     07/06/2016
# Copyright:   (c) Apoorva 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------


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


# file with abstracts and topics
stop=[]
#new_stop=[hi]
f_stop = open("F:/ABCD/New folder/Offical/Mitacs/WORK/Data/SmartStoplist.txt","r")
for i in f_stop:
    stop.append(i.strip())

filename = "F:/ABCD/New folder/Offical/Mitacs/WORK/Data/try1_1_6.csv"

documents = []
classes = []

docs = csv.reader(open(filename, "rb"))
dataset = list(docs)
#print dataset[:5]
for doc in dataset[:300]:
    #print "l", doc
    documents.append(doc[0])
    classes.append(float(doc[1]))

print len(documents)


#performing tfidf doc ( doc : terms ) ---[ for terms use all words from corpus(data) ]
tfidf = TfidfVectorizer(stop_words=stop ,ngram_range=tuple([1,3]), lowercase=True, strip_accents="unicode", use_idf=True, norm="l1",
                        min_df=2,max_df=10)
'''
#performing tfidf doc ( doc : terms ) ---[ for terms use vocabulary ]
tfidf = TfidfVectorizer(stop_words=stop ,ngram_range=tuple([1,3]), lowercase=True, strip_accents="unicode", use_idf=True, norm="l1",
                        min_df=2,max_df=10)
'''
A = tfidf.fit_transform(documents)

num_terms = len(tfidf.vocabulary_)
terms = [""] * num_terms
for term in tfidf.vocabulary_.keys():
    terms[tfidf.vocabulary_[term]] = term
print "Created document-term matrix of size %d x %d" % (A.shape[0], A.shape[1])

fnames = tfidf.get_feature_names()
#print(fnames)
arr = A.toarray()


def get_data():
    data = documents
    target = classes
    target_names = [ ('C-'+str(i)) for i in range(1,11)]