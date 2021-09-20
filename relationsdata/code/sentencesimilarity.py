import nltk_util

import utils

from charagram_master.main.charagram_model import charagram_model
from charagram_master.main.params import params
import lasagne


def getW2VVocabWords(initialWs):
	global w2v_model
	if w2v_model == None:
		init_Word2Vec()
	newWs = [];
	for W in initialWs:
		if W in w2v_model.vocab:
			newWs.append(W);
	return newWs

def init_Word2Vec():
	global w2v_model
	w2v_model = models.word2vec.Word2Vec.load_word2vec_format('../../../GoogleNews-vectors-negative300.bin',binary=True);
	w2v_model.init_sims(replace=True);


def getCNNSimilarity(question1, question2):
	global cnn_model
	if cnn_model == None:
		intiCNNModel()
	g1 = []
	g2 = []
	g1.append(cnn_model.hash(question1))
	g2.append(cnn_model.hash(question2)
	e1 = cnn_model.feedforward_function(g1)
	e2 = cnn_model.feedforward_function(g2)
	return cnn_model.scoring_function(e1,e2)[0]

def intiCNNModel():
	global cnn_model
	param_cnn = params()
	param_cnn.domain = 'sentence'; 
	param_cnn.nntype='charagram';
	param_cnn.train='charagram_master/data/ppdb-xl-phrasal-preprocessed.txt';
	param_cnn.evaluate='True';
	param_cnn.numlayers=1;
	param_cnn.act=lasagne.nonlinearities.tanh
	param_cnn.loadmodel='charagram_master/data/charagram_phrase.pickle'
	param_cnn.worddim=200;
	param_cnn.featurefile='charagram_master/data/charagram_phrase_features_234.txt';
	param_cnn.cutoff=0
	param_cnn.margin=0.4;
	param_cnn.type='MAX'
	param_cnn.clip=None
	param_cnn.learner=lasagne.updates.adam
	cnn_model = charagram_model(param_cnn)


'''
  Determine question similarity: based on W2V vectors and wordnet similarity
'''
def getSimilarity(question1, answer1, question2, answer2, useWN=False, useCharagram=False):
	question1Ws = nltk_util.word_tokenize(question1);
	q2Ws = nltk_util.word_tokenize(question2);
	a2Ws = nltk_util.word_tokenize(answer2);
	answer1Ws = nltk_util.word_tokenize(answer1);
	# Get Phrase-question similarity and Phrase-answer similarity
	semanticSim_q1_2 =0;
	semanticSim_a1_2 = 0;
	w2vSim_q1_2 = 0;
	w2vSim_a1_2 = 0;
	cnnSim_q1_2 = 0
	cnnSim_a1_2 = 0 
	if useWN:
		semanticSim_q1_2 = nltk_util.similarity(question1,question2,True)
		semanticSim_a1_2 = nltk_util.similarity(answer1,answer2,True)
	if useCharagram:
		cnnSim_q1_2 = getCNNSimilarity(question1, question2)
		cnnSim_a1_2 = getCNNSimilarity(answer1, answer2)

	qWs_w2v = getW2VVocabWords(question1Ws);
	aWs_w2v = getW2VVocabWords(answer1Ws);
	q2Ws_w2v = getW2VVocabWords(q2Ws);
	a2Ws_w2v = getW2VVocabWords(a2Ws);
	if min(len(qWs_w2v), len(q2Ws_w2v)) > 0:
		w2vSim_q1_2 = w2v_model.n_similarity(qWs_w2v,q2Ws_w2v)
	if min(len(aWs_w2v), len(a2Ws_w2v)) > 0:
		w2vSim_a1_2 = w2v_model.n_similarity(aWs_w2v,a2Ws_w2v)

	finalSim = (semanticSim_q1_2+w2vSim_q1_2)+(semanticSim_a1_2+w2vSim_a1_2)/2.0
	#print phrase+";"+str(finalSim)
	return finalSim


w2v_model = None
cnn_model = None
