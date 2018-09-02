# import modules & set up logging
import gensim, logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = ["Hello, how are you?".split(), "HI, I am good!".split()]
# train word2vec on the two sentences

'''
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('data') # a memory-friendly iterator
'''

model = gensim.models.Word2Vec() # an empty model, no training
#model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
#model.train(other_sentences)  # can be a non-repeatable, 1-pass generator


model = gensim.models.Word2Vec(sentences, min_count=1, size = 3, workers = 4)
model.save('mymodel1')

model2 = gensim.models.Word2Vec.load('mymodel1')