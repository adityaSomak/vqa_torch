from gensim import models
import argparse
import sys
import re
import nltk_util
import numpy as np
#import lasagne

#from charagram_master import utils

#from charagram_master.charagram_model import charagram_model
#from charagram_master.params import params


def getW2VIndices(listofWords):
	indices =[]
	for W in listofWords:
		if W == "of":
			indices.append(word2Index["Of"])
			continue
		if W in w2v_model.vocab:
			indices.append(word2Index[W]);
	return indices

def toString(indices):
	return ",".join(str(x) for x in indices)

def getVectorString(indices):
	mean = np.mean(w2v_model.syn0[indices,:],0);
	return " ".join(str(x) for x in mean);

def loadAllRelations(allPredicatesfile):
	allPredicates = {}
	with open(allPredicatesfile,'r') as allPredicatesF:
		id=0
		for line in allPredicatesF:
			allPredicates[line.strip()] = id;
			id += 1
	print "All relations loaded"
	return allPredicates

def getCNNVector(question, modelType="charagram"):
	global cnn_model
	if cnn_model == None:
		intiCNNModel()
	g1 = []
	g1.append(cnn_model.hash(question))
	return cnn_model.feedforward_function(g1)[0]

def intiCNNModel():
	global cnn_model
	param_cnn = params()
	param_cnn.domain = 'sentence'; 
	param_cnn.nntype='charagram';
	param_cnn.train='charagram_master/data/ppdb-xl-phrasal-preprocessed.txt';
	param_cnn.evaluate='True';
	param_cnn.numlayers=1;
	param_cnn.act= lasagne.nonlinearities.tanh
	param_cnn.loadmodel='charagram_master/data/charagram_phrase.pickle'
	param_cnn.worddim=200;
	param_cnn.featurefile='charagram_master/data/charagram_phrase_features_234.txt';
	param_cnn.cutoff=0
	param_cnn.margin=0.4;
	param_cnn.type='MAX'
	param_cnn.clip=None
	param_cnn.learner=lasagne.updates.adam
	cnn_model = charagram_model(param_cnn)
	print "CNN Model loaded..."

##############################################

chunkSize = 50000;
MAX_NUM_CHUNKS = 1445322/chunkSize;

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('sntEmbType', help='type of sentence embedding: word/phrase')
parser.add_argument('-numChunks', type=int, help ="number of chunks to process", default=MAX_NUM_CHUNKS)
parser.add_argument('-functionName', help='name of function if phrase selected', default ='False')
parser.add_argument('-isIndexBased', help='True: if sentence is just one index/False', default ='False')
parser.add_argument('-numThread', type=int, help ="number of chunks to process")
parser.add_argument('-outputMatrices', default='False')
argdict =  vars(parser.parse_args(sys.argv[1:]))

print argdict
useWordAggregates = True;
if argdict['sntEmbType'] == "phrase":
	useWordAggregates = False;
	functionName = argdict['functionName']
	isIndexBased = (argdict['isIndexBased'] == 'True');

chunks = []
for i in range(0,MAX_NUM_CHUNKS):
	chunks.append((i*chunkSize,(i+1)*chunkSize));
chunks.append((MAX_NUM_CHUNKS*chunkSize, -1))

cleanedDataFilePrefix = "../trainingData/cleaned_march/nounPairPhraseQuestionTraining"
allPredicatesfile = "../input/filteredAnnotatedPredicates.txt"

predicatesIndex = loadAllRelations(allPredicatesfile)

w2v_model = None
outputdir = "matrices_nn"
threadSuffix = "_"+str(argdict['numThread'])
if useWordAggregates:
	outputdir = 'matrices'
	threadSuffix = ""
	print "Initializing word2vec..."
	w2v_model = models.word2vec.Word2Vec.load_word2vec_format('../../../GoogleNews-vectors-negative300.bin',binary=True);
	w2v_model.init_sims(replace=True);
	word2Index ={}
	for i,k in enumerate(w2v_model.index2word):
		word2Index[k] = i;
	print "Model Loaded..."

cnn_model = None


if argdict['outputMatrices'] == 'True':
	E_QFile = open('../trainingData/cleaned_march/'+outputdir+'/E_Q'+threadSuffix+'.txt','w');
	E_W1File = open('../trainingData/cleaned_march/'+outputdir+'/E_W1'+threadSuffix+'.txt','w');
	E_W2File = open('../trainingData/cleaned_march/'+outputdir+'/E_W2'+threadSuffix+'.txt','w');
	E_PFile = open('../trainingData/cleaned_march/'+outputdir+'/E_P'+threadSuffix+'.txt','w');
	R_predFile = open('../trainingData/cleaned_march/'+outputdir+'/R_pred'+threadSuffix+'.txt','w');
	R_pred_indexFile = open('../trainingData/cleaned_march/'+outputdir+'/R_pred_id'+threadSuffix+'.txt','w')
	allFiles = [E_QFile, E_W1File, E_W2File, E_PFile, R_predFile, R_pred_indexFile] 

#trnDataVecIndicesF =  open("../trnDataVectorIndices.txt",'w'); 
maxChunksToConsider = min(MAX_NUM_CHUNKS, argdict['numChunks']);
minChunkToConsider = 0
if not useWordAggregates:
	minChunkToConsider = 0+ argdict['numThread'] * (maxChunksToConsider/4)
	maxChunksToConsider = minChunkToConsider+(maxChunksToConsider/4)
for i in range(minChunkToConsider,maxChunksToConsider):
	fileName = cleanedDataFilePrefix+str(chunks[i][0])+".txt";
	j=0;
	print "processing chunk: "+ str(i);
	print "processing file: "+ fileName;
	with open(fileName,'r') as f:
		for line in f:
			tokens = line.split('\t');
			words = tokens[0].split(",")
			words[0] = words[0][1:].strip()
			words[0] = re.sub(r"\'","", words[0])
			words[1] = words[1][:-1].strip()
			words[1] = re.sub(r"\'","", words[1])

			if useWordAggregates:
				wordIndices = getW2VIndices(words);
				if len(wordIndices) < 2:
					continue;
				index = predicatesIndex[tokens[1].strip()]
				relation = nltk_util.word_tokenize(tokens[1]);
				question = nltk_util.word_tokenize(tokens[2]);
				phrase = nltk_util.word_tokenize(tokens[3]);

				relationIndices = getW2VIndices(relation);
				questionIndices = getW2VIndices(question);
				phraseIndices = getW2VIndices(phrase);

				#trnDataVecIndicesF.write(str(wordIndices[0])+"\t"+str(wordIndices[1])+"\t"+\
				#	toString(relationIndices)+"\t"+toString(questionIndices)+"\t"+
				#	toString(phraseIndices));

				if argdict['outputMatrices'] == 'True':
					E_W1File.write(getVectorString([wordIndices[0]])+"\n")
					E_W2File.write(getVectorString([wordIndices[1]])+"\n")
					E_PFile.write(getVectorString(phraseIndices)+"\n")
					E_QFile.write(getVectorString(questionIndices)+"\n")
					R_predFile.write(getVectorString(relationIndices)+"\n")
					R_pred_indexFile.write(str(index)+"\n")
					if j%1000==0:
						for file in allFiles:
							file.flush()
				#trnDataVecIndicesF.write("\n");
				#if j%1000==0:
				#	trnDataVecIndicesF.flush();
			else:
				word1vector = getCNNVector(words[0], "charagram")
				word2vector = getCNNVector(words[1], "charagram")
				index = predicatesIndex[tokens[1].strip()]
				relationVector = getCNNVector(tokens[1], "charagram")
				questionVector = getCNNVector(tokens[2], "charagram")
				phraseVector = getCNNVector(tokens[3], "charagram")
				if argdict['outputMatrices'] == 'True':
					E_W1File.write(" ".join(str(x) for x in word1vector)+"\n")
					E_W2File.write(" ".join(str(x) for x in word2vector)+"\n")
					E_PFile.write(" ".join(str(x) for x in phraseVector)+"\n")
					E_QFile.write(" ".join(str(x) for x in questionVector)+"\n")
					R_predFile.write(" ".join(str(x) for x in relationVector)+"\n")
					R_pred_indexFile.write(str(index)+"\n")
					if j%1000==0:
						for file in allFiles:
							file.flush()
				
			j=j+1;

#trnDataVecIndicesF.close()
if argdict['outputMatrices'] == 'True':
	for file in allFiles:
		file.close()