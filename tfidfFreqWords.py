import unicodecsv as csv
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import pandas as pd
import os
import random
import numpy as np
from sklearn import decomposition
from nltk.tokenize import RegexpTokenizer

train_documents = []
test_documents = []
train_labels = []
test_labels = []
classes = []

names=['AI','CC','CE','CL','CR','DS','GR','IR','LG','PF','SI']
#names = ['IR','AI','CE']
for n in names:
    filename = "data/ncs-"+n+".csv"
    docsf = csv.reader(open(filename, "rb"))
    dataset = list(docsf)
    #print dataset[:5]
    documents=[]
    for doc in dataset:
        #print "l", doc
        documents.append(doc[0].lower()+'. '+doc[1].lower())
        classes.append(n)
    print "Ok", len(documents)
    random.shuffle(documents)
    #Documents is a list of strings for the given class
    train_size = len(documents[:int(0.7*len(documents))])
    test_size = len(documents[int(0.7*len(documents)):])
    #Training and testing examples of the given class
    train_labels.extend([n]*train_size)
    test_labels.extend([n]*test_size)
    train_documents.extend(documents[:int(0.7*len(documents))])
    test_documents.extend(documents[int(0.7*len(documents)):])
classes=list(set(classes))
# train_set = list(itertools.chain.from_iterable(train_documents))
# test_set = list(itertools.chain.from_iterable(test_documents))

# count_vectorizer = CountVectorizer()
# count_vectorizer.fit_transform(train_set)
# freq_term_matrix = count_vectorizer.transform(test_set)
# tfidf = TfidfTransformer(norm="l2")
# tfidf.fit(freq_term_matrix)



def top_tfidf_feats(row, features, top_n=25):
	''' Get top n tfidf values in row and return them with their corresponding feature names.'''
	topn_ids = np.argsort(row)[::-1][:top_n]
	top_feats = [(features[i], row[i]) for i in topn_ids]
	df = pd.DataFrame(top_feats)
	df.columns = ['feature', 'tfidf']
	return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
	''' Top tfidf features in specific document (matrix row) '''
	row = np.squeeze(Xtr[row_id].toarray())
	return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
	''' Return the top n features that on average are most important amongst documents in rows
		indentified by indices in grp_ids. '''
	if grp_ids:
		D = Xtr[grp_ids].toarray()
	else:
		D = Xtr.toarray()

	D[D < min_tfidf] = 0
	tfidf_means = np.mean(D, axis=0)
	return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=30):
	''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
		calculated across documents with the same class label. '''
	dfs = []
	labels = np.unique(y)
	for label in labels:
		ids = np.where(y==label)
		feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
		feats_df.label = label
		dfs.append(feats_df)
	return dfs

vectorizer = TfidfVectorizer(
		max_df=0.7, min_df=6, max_features=None, strip_accents='unicode', decode_error='ignore',
		analyzer="word", token_pattern=r'\w{2,}', ngram_range=(1, 3),
		use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

# print len(train_set)
# print train_set[0]

Xtr = vectorizer.fit_transform(train_documents)
features = vectorizer.get_feature_names()

num_terms = len(vectorizer.vocabulary_)
terms = [""] * num_terms
for term in vectorizer.vocabulary_.keys():
    terms[vectorizer.vocabulary_[term]] = term
print "Created document-term matrix of size %d x %d" % (Xtr.shape[0], Xtr.shape[1])

numbered = map(lambda x:classes.index(x), train_labels)
y = np.array(list(numbered))
#y=np.array(train_labels)

feats = top_feats_by_class(Xtr, y, features)
#print feats
df=''
for i, label, in enumerate(classes):
    print "-----------", label, "-----------"
    print feats[i]
    '''for j, df in enumerate(feats):
	   print j," babaS  ",j, str(df)'''

'''
print "\n   +++++++++   NMF   +++++++++   \n"
model = decomposition.NMF(init="nndsvd", n_components=30, max_iter=75)
W = model.fit_transform(Xtr)
H = model.components_
for topic_index in range(W.shape[1]):
    top_indices = np.argsort(W[:,topic_index])[::-1][0:10]
    term_ranking = [train_documents[i] for i in top_indices]
    #print "TTTT       " , term_ranking
    print "TT   ", top_indices
    topic="topic"+str(topic_index)
    #G2.add_node(topic)
    #G2.add_nodes_from(term_ranking)
    #for i in term_ranking:
        #G2.add_edge(topic,i)
#nx.write_gml(G2,"test9w.gml")

print "Matrix H"
for topic_index in range(H.shape[0]):
    #'For H'
    top_indices = np.argsort(H[topic_index, :])[::-1][0:15]
    #print top_indices

    top=np.argsort(H[topic_index, :])[::-1][0:1]
    #print top
    #print "length",len(terms)

    term_ranking = [terms[i] for i in top_indices]
    #print terms[top[0]]
    s = ", ".join(term_ranking)
    print topic_index, s
    tokenizer = RegexpTokenizer(r'\w+')


    # #Getting index for topic assgimnet in W
    index_topic_W = np.argsort(W[:,topic_index])[::-1][0:1]
    #print "Topic  " ,index_topic_W
'''