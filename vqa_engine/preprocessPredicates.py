from gensim import models
from util import nltk_util
from util.Parameters import *


def getW2VVocabWords(initialWs, w2v_model):
    newWs = []
    for W in initialWs:
        if W == "of":
            newWs.append("Of")
            continue
        if W in w2v_model.vocab:
            newWs.append(W)
    return newWs

WordVectors_File = AUX_DATA_DIR + "GoogleNews-vectors-negative300.bin"
word2vec_model =  models.word2vec.Word2Vec.load_word2vec_format(WordVectors_File, binary=True);
word2vec_model.init_sims(replace=True)
word2Index = {}
for i, k in enumerate(word2vec_model.index2word):
    word2Index[k] = i
print "model loaded"

dir = REP_NET_PARAMS_DIR
processedPredicates = open(dir + "data/processedFAPredicates.txt",'w')
with open(dir+"data/filteredAnnotatedPredicates.txt") as predicates:
    for line in predicates:
        relW = nltk_util.word_tokenize(line.strip())
        rWs = getW2VVocabWords(relW, word2vec_model)
        processedPredicates.write(line.strip()+"\t"+",".join(rWs)+"\n")
processedPredicates.close()

