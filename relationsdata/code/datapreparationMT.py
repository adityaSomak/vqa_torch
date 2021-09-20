from gensim import models
#from assoc_space import AssocSpace
import nltk_util
import networkx as nx
import szClrLocDictionary
from fuzzywuzzy import fuzz
from joblib import Parallel, delayed


def getW2VVocabWords(initialWs):
	newWs = [];
	for W in initialWs:
		if W == "of":
			newWs.append("Of")
			continue
		if W in w2v_model.vocab:
			newWs.append(W);
	return newWs

def calculateMaximumSimilarityIn(answerWs, phraseWs):
	maxSim = 0
	for aW in answerWs:
		for pW in phraseWs:
			sim = w2v_model.similarity(aW,pW)
			if sim > maxSim:
				maxSim = sim
	return maxSim

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
		#maxSim_a_p = calculateMaximumSimilarityIn(aWs_w2v,pWs_w2v)

	finalSim = (semanticSim_q_p+w2vSim_q_p)+(semanticSim_a_p+w2vSim_a_p)/2.0
	#print phrase+";"+str(finalSim)
	return finalSim#, maxSim_a_p

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
	pWs = getW2VVocabWords(phraseWs[minWordPos:maxWordPos-1])
	#### Change in 28-Feb ####
	if (" ".join(pWs)) == "t shirt":
		return None
	pWs_without_beVerb = []
	onlyStopWords = True
	stopWords = set(["is","are","a","an","the"])
	for i,pW in enumerate(pWs):
		if pW not in stopWords:
			onlyStopWords = False
		if i==0 and (pW == "is" or pW == "are"): 
			continue
		elif szClrLocDictionary.matchesSzColorLocation(pW) != "color":
			pWs_without_beVerb.append(pW)
	if not onlyStopWords:
		pWs = pWs_without_beVerb
	print "\t"+str(pWs)+"\t"+str(pWs_without_beVerb)+","+ str(nounWsPhrase)
	#### Change in 28-Feb ####
	if len(pWs) == 0:
		return None
	for rel in relations:
		relW = nltk_util.word_tokenize(rel);	
		#relW.extend(nounWsPhrase)
		rWs = getW2VVocabWords(relW)
		w2vSim = w2v_model.n_similarity(rWs,pWs)
		if w2vSim > maxSim:
			maxSim = w2vSim;
			bestRelation = rel
	return bestRelation

def getMostSimilarPhrase(imageID, question, answer,regionDescriptF, curPhraseID, phrasesStore):
	if curPhraseID[0] != imageID:
		for line in regionDescriptF:
			tokens = line.split('\t');
			pID_ = int(tokens[0]);
			if pID_ < imageID:
				# This should not happen!
				try:
					if pID_ > imageID-5:
						del phrasesStore[pID_];
				except KeyError:
					pass
				continue;
			if pID_ == imageID:
				curPhraseID[0] = imageID
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
  Remove junks such as "t" for "t shirt"
'''
def isJunkWord(word):
	return (len(word) == 1)


def areConsecutiveWords(nprps, i, j):
	return  (abs(nprps[j][2]-nprps[i][2]) == 1)

'''
  Get noun/PRP/JJ pairs that are connected in the dependency graph.
'''
def getAllNounPairs(postaggedPhrase,wordPairsInPhrase):
	nprps = []
	
	G=nx.Graph()
	nodes = set()
	#print postaggedPhrase
	#print wordPairsInPhrase
	for wordpair in wordPairsInPhrase:
		words_and_pos = wordpair.split(",");
		G.add_edge(words_and_pos[0],words_and_pos[1]);
		nodes.add(words_and_pos[0]);
		nodes.add(words_and_pos[1]);
	index =1;
	for wordpos in postaggedPhrase:
		if wordpos[1].startswith("N") or wordpos[1] == "PRP" or \
		wordpos[1].startswith("J"):
			nprps.append((wordpos[0],wordpos[1],index));
		index= index+1;
	nounpairs = []
	pairsSeen = set()

	for j in range(0,len(nprps)):
		src_node = nprps[j][0]+"-"+str(nprps[j][2]);
		if (not nprps[j][1].startswith("N")) or isJunkWord(nprps[j][0]):
			continue
		#print pairsSeen
		for i in range(0,len(nprps)):
			target_node = nprps[i][0]+"-"+str(nprps[i][2]);
			reversePair = str(i)+","+str(j)
			if i == j or (reversePair in pairsSeen) or \
			areConsecutiveWords(nprps, i, j) or \
			isJunkWord(nprps[i][0]):
				continue
			#print "Considering:"+ src_node+" "+ target_node
			sh_path=[]
			try:
				if (src_node in nodes) and (target_node in nodes):
					sh_path = nx.shortest_path(G,source=src_node,target=target_node);
			except nx.NetworkXNoPath, nx.exception.NetworkXError:
				sh_path = []
			if sh_path and len(sh_path)-2 < 3:
				nounpairs.append((src_node,target_node));
				pairsSeen.add(str(j)+","+str(i))
	#print nounpairs
	return nounpairs

def getWords(pairInPhrase):
	wordsInpair = map(lambda x: x[:x.rindex("-")], pairInPhrase)
	return wordsInpair;

#########################################################
# This method is for running multi-threaded training data
# creation. Each process will take one chunk of the data
# and write to a different file
#########################################################
def processAndWriteCleanedTrainingData(questionsFileName, regionDescriptionsFileName, \
	cleanedDataFilePrefix, START=0, END=-1):
	curPhraseID = [0]
	phrasesStore = {}

	questionsStore = set()
	curImageID = -1
	cleanedData = open(cleanedDataFilePrefix+str(START)+".txt",'w')
	regionDescriptF = open(regionDescriptionsFileName, 'r')
	EXISTENCE_THRESHOLD = 0.6
	i=0
	qaLineNum = -1; 
	print "Thread Processing: ("+str(START)+","+str(END)+")"
	with open(questionsFileName,'r') as questionsF:
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
			## manage questions-store #####
			if imageID != curImageID:
				curImageID = imageID
				questionsStore = set()
			if question in questionsStore:
				continue
			questionsStore.add(question)
			## remove-repeats #############

			npsInQuestion = (tokens[3].split(";"))[:-1]
			print "ID:" + str(imageID)+"; Q: "+ question+";\tA: "+answer
			phrase = getMostSimilarPhrase(imageID, question, answer, regionDescriptF,\
				curPhraseID, phrasesStore);
			print "\t\t; Q: "+ question+";\tPhrase: "+phrase[0]
			if phrase == None:
				print "Phrase Not Found due to OOV"
				continue;
			#print "Selected Phrase: "+ phrase[0]
			wordPairsInPhrase = (phrase[1].split(";"))[:-1]; # the last index is empty
			postaggedPhrase = nltk_util.getPosTags(phrase[0])
			nounPairsInPhrase = getAllNounPairs(postaggedPhrase,wordPairsInPhrase);
			for pairInPhrase in nounPairsInPhrase:
				#print pairInPhrase
				#if not pairPassesPosTagCheck(pairInPhrase, postaggedPhrase):
				#	continue
				nounWsPhrase = map(lambda x: x[:x.rindex("-")],pairInPhrase)		
				if "it" in nounWsPhrase or nounWsPhrase[0] == nounWsPhrase[1]:
					continue
				wordPositionsInPhrase = map(lambda x: int(x[x.rindex("-")+1:]),pairInPhrase)
				if wordPositionsInPhrase[1] - wordPositionsInPhrase[0] == 2 and \
					postaggedPhrase[wordPositionsInPhrase[0]+1 - 1][0] == "'s":
					continue
				sim = getPairSimilarity(npsInQuestion,pairInPhrase);
				if sim > EXISTENCE_THRESHOLD:
					print "\t"+str(pairInPhrase)+"\t"+phrase[0]+"\t"+question
					relation = getRelationUsingHeuristics(pairInPhrase, phrase[0], question, allPredicates)
					revisedRelation = szClrLocDictionary.simplifyRelation(relation, pairInPhrase, postaggedPhrase, w2v_model)
					if revisedRelation != None:
						cleanedData.write(str(getWords(pairInPhrase))+"\t"+ revisedRelation+"\t" + question+"\t"+phrase[0]+"\n");
						print "\t\t"+str(relation)+"\t"+revisedRelation
						i=i+1
					if i%1000==0:
						cleanedData.flush()
	cleanedData.close()
##########################################################
import sys

mt = sys.argv[1];
print mt
questionsFileName = "../input/sortedqaDependencies.txt"
regionDescriptionsFileName = "../input/sortedregionDescriptionsDependencies.txt"
allPredicatesfile = "../input/filteredAnnotatedPredicates.txt";#"../finalSortedPredicates2.txt"

print "Initializing word2vec and conceptnet5..."
w2v_model = models.word2vec.Word2Vec.load_word2vec_format('../../../GoogleNews-vectors-negative300.bin',binary=True);
w2v_model.init_sims(replace=True);
print "Model Loaded..."
print "Done"

allPredicates = []
with open(allPredicatesfile,'r') as allPredicatesF:
	for line in allPredicatesF:
		allPredicates.append(line.strip());

chunkSize = 50000;
chunks = []
numChunks = 1445322/chunkSize;
for i in range(0,numChunks):
	chunks.append((i*chunkSize,(i+1)*chunkSize));
chunks.append((numChunks*chunkSize, -1))

cleanedDataFilePrefix = "../trainingData/nounPairPhraseQuestionTraining";

if mt == "True":
	Parallel(n_jobs=8)(delayed(processAndWriteCleanedTrainingData)(questionsFileName, \
	regionDescriptionsFileName, cleanedDataFilePrefix, chunk[0], chunk[1] ) \
			for chunk in chunks);
else:
	chunkNum = int(sys.argv[2])
	processAndWriteCleanedTrainingData(questionsFileName, regionDescriptionsFileName, \
		cleanedDataFilePrefix, chunks[chunkNum][0], chunks[chunkNum][1])
