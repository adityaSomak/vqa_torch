from gensim import models
#from assoc_space import AssocSpace
import nltk_util
import networkx as nx
from fuzzywuzzy import fuzz


def getW2VVocabWords(initialWs):
	newWs = [];
	for W in initialWs:
		if W in w2v_model.vocab:
			newWs.append(W);
	return newWs

'''
  Based on the question and the answer, get the most similar phrase
'''
def getSimilarity(question, answer, phrase, useWN=False):
	questionWs = nltk_util.word_tokenize(question);
	phraseWs = nltk_util.word_tokenize(phrase);
	answerWs = nltk_util.word_tokenize(answer);
	# Get Phrase-question similarity and Phrase-answer similarity
	semanticSim_q_p =0;
	semanticSim_a_p = 0;
	w2vSim_a_p = 0;
	w2vSim_q_p = 0;
	if useWN:
		semanticSim_q_p = nltk_util.similarity(question,phrase,True)
		semanticSim_a_p = nltk_util.similarity(answer,phrase,True)
	qWs_w2v = getW2VVocabWords(questionWs);
	aWs_w2v = getW2VVocabWords(answerWs);
	pWs_w2v = getW2VVocabWords(phraseWs);
	if len(qWs_w2v) > 0 and len(pWs_w2v) > 0:
		w2vSim_q_p = w2v_model.n_similarity(qWs_w2v,pWs_w2v)
	if len(aWs_w2v) > 0 and len(pWs_w2v) > 0:
		w2vSim_a_p = w2v_model.n_similarity(aWs_w2v,pWs_w2v)

	finalSim = (semanticSim_q_p+w2vSim_q_p)+(semanticSim_a_p+w2vSim_a_p)/2.0
	#print phrase+";"+str(finalSim)
	return finalSim

def fuzzyAndDistributionalWordSimilarity(word1,word2,inw2v):
	sim =0;
	if inw2v:
		sim = w2v_model.similarity(word1,word2)
	sim = sim*2+ fuzz.ratio(word1,word2)/100.0;
	return sim/3.0;

'''
  For a Noun-pair in phrase, get the overall maximum similarity with
  noun-pairs in question.
'''
def getPairSimilarity(npsInQuestion,npPairInPhrase):
	nounWsPhrase = map(lambda x: x[:x.rindex("-")],npPairInPhrase)
	wordsInQuestion = set()
	for npInQ in npsInQuestion:
		nounWs = map(lambda x: x[:x.rindex("-")],npInQ.split(","))
		for nW in nounWs:
			wordsInQuestion.add(nW)
	maxSimW1 = 0;
	maxSimW2 = 0;
	nw1_inw2v = False
	if nounWsPhrase[0] in w2v_model.vocab:
		nw1_inw2v = True;
	nw2_inw2v = False;
	if nounWsPhrase[1] in w2v_model.vocab:
		nw2_inw2v = True;
	for winQ in wordsInQuestion:
		if winQ not in w2v_model.vocab:
			continue;
		sim = fuzzyAndDistributionalWordSimilarity(winQ,nounWsPhrase[0],nw1_inw2v);#
		if sim > maxSimW1:
			maxSimW1 = sim
		sim = fuzzyAndDistributionalWordSimilarity(winQ,nounWsPhrase[1],nw2_inw2v);#
		if sim > maxSimW2:
			maxSimW2 = sim
	return (maxSimW1+maxSimW2)/2.0

'''
  Given noun-pair, phrase and question, get most similar relation
'''
def getRelationUsingHeuristics(pairInPhrase, phrase, question, relations):
	wordPositionsInPhrase = map(lambda x: int(x[x.rindex("-")+1:]),pairInPhrase)
	nounWsPhrase = map(lambda x: x[:x.rindex("-")],pairInPhrase)
	maxSim = 0
	bestRelation = ""
	minWordPos = min(wordPositionsInPhrase)
	maxWordPos = max(wordPositionsInPhrase)
	phraseWs = nltk_util.word_tokenize(phrase)
	pWs = getW2VVocabWords(phraseWs[minWordPos-1:maxWordPos])
	for rel in relations:
		relW = nltk_util.word_tokenize(rel);	
		relW.extend(nounWsPhrase)
		rWs = getW2VVocabWords(relW)
		w2vSim = w2v_model.n_similarity(rWs,pWs)
		if w2vSim > maxSim:
			maxSim = w2vSim;
			bestRelation = rel
	return bestRelation

def getMostSimilarPhrase(imageID, question, answer,regionDescriptF):
	global curPhraseID
	if curPhraseID != imageID:
		for line in regionDescriptF:
			tokens = line.split('\t');
			pID_ = int(tokens[0]);
			if pID_ < imageID:
				# This should not happen!
				try:
					del phrasesStore[pID_];
				except KeyError:
					pass
				continue;
			if pID_ == imageID:
				curPhraseID = imageID
				try:
					phrasesStore[pID_].append((tokens[2],tokens[3]))
				except KeyError:
					phrasesStore[pID_] = [(tokens[2],tokens[3])]			
			else:
				phrasesStore[pID_] = [(tokens[2],tokens[3])]	
				break;
	similarities = [];
	index=0
	for phrase in phrasesStore[imageID]:
		sim = getSimilarity(question, answer,phrase[0].replace("/'",""),False)
		if not isinstance(sim, list):
			similarities.append((sim,index,phrase[1].split(";")));
		index=index+1
		#if sim > maxSim:
		#	maxSim = sim;
		#	maxPhrase = phrase;
	print str(imageID)+" #Phrases: "+str(len(similarities));
	if len(similarities) == 0:
		return None;
	#print similarities
	similarities = sorted(similarities, key=lambda i: (i[0],len(i[2])), reverse=True)
	# TODO: 
	## Get the top 10, then use WN+W2V similarity
	return phrasesStore[imageID][similarities[0][1]]

def pairPassesPosTagCheck(pairInPhrase, postaggedPhrase):
	pairW_pos = pairInPhrase.split(",");
	w1_pos = int(pairW_pos[0].split("-")[1])
	w2_pos = int(pairW_pos[1].split("-")[1])
	if not (postaggedPhrase[w1_pos-1][1].startswith("N") or \
		postaggedPhrase[w1_pos-1][1]=="PRP"):
		return False;
	if not (postaggedPhrase[w2_pos-1][1].startswith("N") or \
		postaggedPhrase[w2_pos-1][1]=="PRP"):
		return False;
	return True

'''
  Get noun/PRP/JJ pairs that are connected in the dependency graph.
'''
def getAllNounPairs(postaggedPhrase,wordPairsInPhrase):
	nprps = []
	
	G=nx.Graph()
	for wordpair in wordPairsInPhrase:
		words_and_pos = wordpair.split(",");
		G.add_edge(words_and_pos[0],words_and_pos[1]);
	index =1;
	for wordpos in postaggedPhrase:
		if wordpos[1].startswith("N") or wordpos[1] == "PRP" or \
		wordpos[1].startswith("J"):
			nprps.append((wordpos[0],wordpos[1],index));
		index= index+1;
	nounpairs = []
	for j in range(1,len(nprps)):
		src_node = nprps[j][0]+"-"+str(nprps[j][2]);
		for i in range(0,j):
			target_node = nprps[i][0]+"-"+str(nprps[i][2]);
			try:
				sh_path = nx.shortest_path(G,source=src_node,target=target_node);
			except nx.NetworkXNoPath, nx.exception.NetworkXError:
				sh_path = []
			if sh_path and len(sh_path)-2 < 3:
				nounpairs.append((src_node,target_node));
	print nounpairs
	return nounpairs

def getWords(pairInPhrase):
	wordsInpair = map(lambda x: x[:x.rindex("-")], pairInPhrase)
	return wordsInpair;

#########################################################
# This method is for running multi-threaded training data
# creation. Each process will take one chunk of the data
# and write to a different file
#########################################################
def processAndWriteCleanedTrainingData(questionsFile, regionDescriptionsFileName, \
	cleanedDataFileName, START=0, END=-1):
	cleanedData = open(cleanedDataFileName,'w')
	regionDescriptF = open(regionDescriptionsFileName, 'r')
	EXISTENCE_THRESHOLD = 0.6
	i=0
	qaLineNum = -1; 
	with open(questionsFile,'r') as questionsF:
		for line in questionsF:
			qaLineNum = qaLineNum+1;
			if qaLineNum < START:
				continue;
			if END!= -1 and qaLineNum >= END:
				break;
			tokens = line.split('\t');
			imageID = int(tokens[0]);
			question = tokens[1][:-1].strip().lower()
			answer = tokens[2][:-1].strip().lower()
			npsInQuestion = (tokens[3].split(";"))[:-1]
			print "Q: "+ question+";\tA: "+answer
			phrase = getMostSimilarPhrase(imageID, question, answer, regionDescriptF);
			if phrase == None:
				print "Phrase Not Found due to OOV"
				continue;
			print "Selected Phrase: "+ phrase[0]
			wordPairsInPhrase = (phrase[1].split(";"))[:-1]; # the last index is empty
			postaggedPhrase = nltk_util.getPosTags(phrase[0])
			nounPairsInPhrase = getAllNounPairs(postaggedPhrase,wordPairsInPhrase);
			for pairInPhrase in nounPairsInPhrase:
				#print pairInPhrase
				#if not pairPassesPosTagCheck(pairInPhrase, postaggedPhrase):
				#	continue			
				sim = getPairSimilarity(npsInQuestion,pairInPhrase);
				if sim > EXISTENCE_THRESHOLD:
					print str(pairInPhrase)+"\t"+phrase[0]+"\t"+question
					relation = getRelationUsingHeuristics(pairInPhrase, phrase[0], question, allPredicates)
					cleanedData.write(str(getWords(pairInPhrase))+"\t"+ relation+"\t" + question+"\t"+phrase[0]+"\n");
					i=i+1
					if i%1000==0:
						cleanedData.flush()
	cleanedData.close()
##########################################################

questionsFile = "../sortedqaDependencies.txt"
regionDescriptionsFileName = "../sortedregionDescriptionsDependencies.txt"

allPredicatesfile = "../vgAndQCPredicates.txt";#"../finalSortedPredicates2.txt"
curPhraseID = 0
phrasesStore = {}

#regionDescriptF = open(regionDescriptionsFileName, 'r')
cleanedData = open("../nounPairPhraseQuestionTraining.txt",'w')

print "Initializing word2vec and conceptnet5..."
w2v_model = models.word2vec.Word2Vec.load_word2vec_format('../../GoogleNews-vectors-negative300.bin',binary=True);
w2v_model.init_sims(replace=True);
print "Model Loaded..."
#assocDir = "/home/ASUAD/saditya1/Desktop/Image_Riddle/conceptnet5/data/assoc/assoc-space-5.4";
#assocSpace = AssocSpace.load_dir(assocDir);
#names = assocSpace.labels
print "Done"

allPredicates = []
with open(allPredicatesfile,'r') as allPredicatesF:
	for line in allPredicatesF:
		allPredicates.append(line.strip());

cleanedDataFileName = "../nounPairPhraseQuestionTraining.txt";
processAndWriteCleanedTrainingData(questionsFile, regionDescriptionsFileName, \
	cleanedDataFileName, START=0, END=-1);
# EXISTENCE_THRESHOLD = 0.6
# i=0
# with open(questionsFile,'r') as questionsF:
# 	for line in questionsF:
# 		tokens = line.split('\t');
# 		imageID = int(tokens[0]);
# 		question = tokens[1][:-1].strip().lower()
# 		answer = tokens[2][:-1].strip().lower()
# 		npsInQuestion = (tokens[3].split(";"))[:-1]
# 		print "Q: "+ question+";\tA: "+answer
# 		phrase = getMostSimilarPhrase(imageID, question, answer, regionDescriptF);
# 		if phrase == None:
# 			print "Phrase Not Found due to OOV"
# 			continue;
# 		print "Selected Phrase: "+ phrase[0]
# 		wordPairsInPhrase = (phrase[1].split(";"))[:-1]; # the last index is empty
# 		postaggedPhrase = nltk_util.getPosTags(phrase[0])
# 		nounPairsInPhrase = getAllNounPairs(postaggedPhrase,wordPairsInPhrase);
# 		for pairInPhrase in nounPairsInPhrase:
# 			#print pairInPhrase
# 			#if not pairPassesPosTagCheck(pairInPhrase, postaggedPhrase):
# 			#	continue			
# 			sim = getPairSimilarity(npsInQuestion,pairInPhrase);
# 			if sim > EXISTENCE_THRESHOLD:
# 				print str(pairInPhrase)+"\t"+phrase[0]+"\t"+question
# 				relation = getRelationUsingHeuristics(pairInPhrase, phrase[0], question, allPredicates)
# 				cleanedData.write(str(getWords(pairInPhrase))+"\t"+ relation+"\t" + question+"\t"+phrase[0]+"\n");
# 				i=i+1
# 				if i%1000==0:
# 					cleanedData.flush()

# cleanedData.close()
