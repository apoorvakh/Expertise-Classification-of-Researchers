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
#performing tfidf doc ( doc : terms ) ---[ for terms use vocabulary ]
tfidf = TfidfVectorizer(stop_words=stop ,ngram_range=tuple([1,3]), lowercase=True, strip_accents="unicode", use_idf=True, norm="l1",
                        min_df=2,max_df=10)
A = tfidf.fit_transform(documents)

num_terms = len(tfidf.vocabulary_)
terms = [""] * num_terms
for term in tfidf.vocabulary_.keys():
    terms[tfidf.vocabulary_[term]] = term
print "Created document-term matrix of size %d x %d" % (A.shape[0], A.shape[1])

fnames = tfidf.get_feature_names()
#print(fnames)
arr = A.toarray()




'''
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            print line
            writer.writerow(repr(line))
        #for lineNo in range(len(data)):
            #print type(data[lineNo]), float(classes[lineNo])
            #float(classes[lineNo])
            #np.append(data[lineNo], [float(classes[lineNo])], 0)
            #print data[lineNo]
            #writer.writerow(np.concatenate(data[lineNo], np.array([float(classes[lineNo])])))
            #writer.writerow(da)


csv_writer(arr,"F:/ABCD/New folder/Offical/Mitacs/WORK/Data/tfidf1.csv")
# list of classes of all instances
csv_writer(classes,"F:/ABCD/New folder/Offical/Mitacs/WORK/Data/classes1.csv")

arr2 = np.append(arr, [[float(x)] for x in classes], 1)

#arr = A.toarray()
print len(arr2[0])


datapath = "F:/ABCD/New folder/Offical/Mitacs/WORK/Data/out1_1_6.csv"

csv_writer(arr2,datapath)
'''