#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Apoorva
#
# Created:     08/06/2016
# Copyright:   (c) Apoorva 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier

newsTrainer = Trainer(tokenizer)

filename = "F:/ABCD/New folder/Offical/Mitacs/WORK/Data/try1_1_6_AbC.csv"

#filename = "ncs-AI.csv"

docSet =[] # data to train

documents = []
classes = []

names=['AI','CL','DS','GR','CR','PF']
for n in names:
    filename = "ncs-"+n+".csv"
    docsf = csv.reader(open(filename, "rb"))
    dataset = list(docsf)
    for i in dataset:
        docSet.append({'text':i[1],'category':i[2]})


# You need to train the system passing each text one by one to the trainer module.

for doc in docSet:
    newsTrainer.train(doc['text'], doc['category'])

# When you have sufficient trained data, you are almost done and can start to use
# a classifier.
newsClassifier = Classifier(newsTrainer.data, tokenizer)

# Now you have a classifier which can give a try to classifiy text of news whose
# category is unknown, yet.
#unknownInstance = "Even if I eat too much, is not it possible to lose some weight"
unknownInstance ="This paper describes a system, called PLP, for compiling ordered logic programs into standard logic programs under the answer set semantics. In an ordered logic program, rules are named by unique terms, and preferences among rules are given by a set of dedicated atoms. An ordered logic program is transformed into a second, regular, extended logic program wherein the preferences are respected, in that the answer sets obtained in the transformed theory correspond with the preferred answer sets of the original theory. Since the result of the translation is an extended logic program, existing logic programming systems can be used as underlying reasoning engine. In particular, PLP is conceived as a front-end to the logic programming systems dlv and smodels."

classification = newsClassifier.classify(unknownInstance)
# the classification variable holds the possible categories sorted by
# their probablity value
print classification

max(c, key=lambda x: x[1])