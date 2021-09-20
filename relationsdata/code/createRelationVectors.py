from gensim import models
import argparse
import sys
import re
import nltk_util
import numpy as np

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

##############################################

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('sntEmbType', help='type of sentence embedding: word/phrase')
parser.add_argument('-functionName', help='name of function if phrase selected', default ='False')
parser.add_argument('-isIndexBased', help='True: if sentence is just one index/False', default ='False')
parser.add_argument('-outputMatrices', default='False')
argdict =  vars(parser.parse_args(sys.argv[1:]))

print argdict
useWordAggregates = True;
if argdict['sntEmbType'] == "phrase":
	useWordAggregates = False;
	functionName = argdict['functionName']
	isIndexBased = (argdict['isIndexBased'] == 'True');

allRelationsF = "../input/filteredAnnotatedPredicates.txt";

print "Initializing word2vec..."
w2v_model = models.word2vec.Word2Vec.load_word2vec_format('../../../GoogleNews-vectors-negative300.bin',binary=True);
w2v_model.init_sims(replace=True);
word2Index ={}
for i,k in enumerate(w2v_model.index2word):
	word2Index[k] = i;
print "Model Loaded..."

if argdict['outputMatrices'] == 'True':
	relationsVFile = open('../trainingData/cleaned_march/matrices/relations.txt','w');

allRelationVectorsF =  open("../trainingData/cleaned_march/matrices/filteredAnnotatedPredicateVectors.txt",'w'); 
j=0;

print "processing file: "+ allRelationsF;
with open(allRelationsF,'r') as f:
	for line in f:
		words = nltk_util.word_tokenize(line.strip());
		relationIndices = getW2VIndices(words);

		if useWordAggregates:
			allRelationVectorsF.write(toString(relationIndices));
			allRelationVectorsF.write("\n");
			if argdict['outputMatrices'] == 'True':
				relationsVFile.write(getVectorString(relationIndices)+"\n");
				if j%1000==0:
					relationsVFile.flush();
		if j%1000==0:
			allRelationVectorsF.flush();
		j=j+1;

allRelationVectorsF.close()
if argdict['outputMatrices'] == 'True':
	relationsVFile.close();