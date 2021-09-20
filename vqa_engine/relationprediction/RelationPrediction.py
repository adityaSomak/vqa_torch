import nltk
import scipy.io
import numpy as np
from scipy.spatial.distance import cdist

import szClrLocDictionary
from util.Parameters import *
from gensim import  matutils
import dot


def getW2VVocabWords(initialWs, w2v_model):
    newWs = []
    for W in initialWs:
        if W == "of":
            newWs.append("Of")
            continue
        if W in w2v_model.vocab:
            newWs.append(W)
    return newWs

'''
  Given noun-pair, phrase and question, get most similar relation
'''
def predictRelationUsingHeuristics(pairInPhrase, phraseWs, indicatorVocab, w2v_model, dependency_path=None):
    global predicateNames, predicateWordsList
    if predicateNames is None:
        [predicateNames, predicateWordsList] = loadPredicateNames()
    wordPositionsInPhrase = map(lambda x: int(x[x.rindex("-") + 1:]), pairInPhrase)
    nounWsPhrase = map(lambda x: x[:x.rindex("-")], pairInPhrase)
    maxSim = 0
    bestRelation = ""
    minWordPos = min(wordPositionsInPhrase)
    maxWordPos = max(wordPositionsInPhrase)

    if abs(maxWordPos-minWordPos) == 1 and wordPositionsInPhrase[0] > wordPositionsInPhrase[1]:
        rel = szClrLocDictionary.matchesSzColorLocation(nounWsPhrase[1])
        if rel is None:
            return ["is",0.5]
        return [rel,1.0]
    pWs = []
    stopWords = {"is", "are", "a", "an", "the", "this", "that"}
    onlyStopWords = True
    if dependency_path is not None:
        for depWordPos in dependency_path:
            depWord = depWordPos.split("-")[:-1]
            depWord = "-".join(depWord)
            position = int(depWordPos.split("-")[-1])-1
            if depWord not in stopWords:
                onlyStopWords = False
            if indicatorVocab[position] == 1:
                pWs.append(depWord)
            elif indicatorVocab[position] == 2:
                pWs.append("Of")
    if onlyStopWords:
        pWs = []
        for i,w in enumerate(phraseWs):
            if minWordPos <= i < maxWordPos-1:
                if indicatorVocab[i] == 1:
                    pWs.append(w)
                elif indicatorVocab[i] == 2:
                    pWs.append("Of")
    #pWs = getW2VVocabWords(phraseWs[minWordPos:maxWordPos - 1], w2v_model)
    #### Change in 28-Feb ####
    compoundPhrase = " ".join(pWs)
    if compoundPhrase in {"is this", "is the", "is a", "is an"}:
        return ["is", 1.0]
    if compoundPhrase in {"of the", "of an", "of", "of a"}:
        return ["Of", 1.0]
    if (" ".join(pWs)) == "t shirt":
        return [None, 0.0]
    pWs_without_beVerb = []
    onlyStopWords = True

    for i, pW in enumerate(pWs):
        if pW not in stopWords:
            onlyStopWords = False
        if i == 0 and (pW == "is" or pW == "are"):
            continue
        elif szClrLocDictionary.matchesSzColorLocation(pW) != "color":
            pWs_without_beVerb.append(pW)
    if not onlyStopWords:
        pWs = pWs_without_beVerb
    #print "\t" + str(pWs) + "\t" + str(pWs_without_beVerb) + "," + str(nounWsPhrase)
    #### Change in 28-Feb ####
    if len(pWs) == 0:
        return [None,0]
    for i,rWs in enumerate(predicateWordsList):
        #relW = nltk_util.word_tokenize(rel)
        # relW.extend(nounWsPhrase)
        #rWs = getW2VVocabWords(relW, w2v_model)
        if not onlyStopWords:
            rWs = set(rWs).difference(stopWords)
        if len(rWs) < 1:
            continue
        v1 = [w2v_model[word] for word in rWs]
        v2 = [w2v_model[word] for word in pWs]
        w2vSim = dot.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))
        #w2vSim = w2v_model.n_similarity(rWs, pWs)
        if w2vSim > maxSim:
            maxSim = w2vSim
            bestRelation = predicateNames[i]
    return bestRelation, maxSim

def loadPredicateNames():
    global predicateNames, predicateWordsList, predicatesFile
    predicateNames = []
    predicateWordsList = []
    with open(predicatesFile, "r") as file:
        for line in file:
            tokens = line.split("\t")
            predicateWordsList.append(tokens[1].replace("\n","").split(","))
            predicateNames.append(tokens[0])
            # predicateNames = np.loadtxt(predicates)
    return predicateNames, predicateWordsList

def loadMatrices():
    global Wp, Wq, W1, W2, Wfc, relations, predicateNames
    Wp = np.loadtxt(questionParams)
    Wq = np.loadtxt(phraseParams)
    W1 = np.loadtxt(word1Params)
    W2 = np.loadtxt(word2Params)
    Wfc = np.loadtxt(fcParams)
    relations = np.loadtxt(allrelations)
    loadPredicateNames()

def loadMatricesFromMatfiles():
    global Wp, Wq, W1, W2, Wfc, b_l1, b_fc, relations, predicateNames
    Wq = scipy.io.loadmat(questionMat)['Wq']
    Wp = scipy.io.loadmat(phraseMat)['Wp']
    W1 = scipy.io.loadmat(word1Mat)['W1']
    W2 = scipy.io.loadmat(word2Mat)['W2']
    Wfc = scipy.io.loadmat(fcMat)['W_fc']
    b_l1 = scipy.io.loadmat(biasl1Mat)['b_l1']
    b_fc = scipy.io.loadmat(biasFCMat)['b_fc']
    relations = np.loadtxt(allrelations)
    loadPredicateNames()
    # ffnet = REPNet(Wp, Wq, W1, W2, Wfc, b_l1, b_fc)

def getWordEmbedding(word, w2v):
    return w2v.word2vec_model.syn0[w2v.word2Index[word], :]


def getPhraseEmbedding(caption, w2v):
    words = nltk.word_tokenize(caption)
    indices = []
    for W in words:
        if W in w2v.word2vec_model.vocab:
            indices.append(w2v.word2Index[W])
    return np.mean(w2v.word2vec_model.syn0[indices, :], 0)


def assignClosestRelation(R_pred, relations):
    id = np.argmin(cdist(R_pred, relations, 'euclidean'))
    return predicateNames[id]


def softmax(x):
    e_x = np.exp(x)
    out = e_x / np.sum(e_x)
    return out


def predictRelation(nounPair, caption, w2v, isQuestion):
    global matricesLoaded
    if not matricesLoaded:
        loadMatricesFromMatfiles()
    W1emb = getPhraseEmbedding(nounPair[0], w2v)
    W2emb = getPhraseEmbedding(nounPair[1], w2v)
    captionEmb = getPhraseEmbedding(caption, w2v)

    # Feed-Forward
    # r_id = np.argmax(ffnet.feed_forward_q(captionEmb, W1emb, W2emb))
    # r_id = np.argmax(ffnet.feed_forward_p(captionEmb, W1emb, W2emb))
    if isQuestion:
        R_pred = np.dot(captionEmb, Wq) + np.dot(W1emb, W1) + np.dot(W2emb, W2) + b_l1.T
    else:
        R_pred = np.dot(captionEmb, Wp) + np.dot(W1emb, W1) + np.dot(W2emb, W2) + b_l1.T
    o_r = softmax(np.dot(R_pred,Wfc) + b_fc.T)
    r_id = np.argmax(o_r)
    return [predicateNames[r_id], o_r[0,r_id]]


Wp = None
Wq = None
W1 = None
W2 = None
Wfc = None
b_l1 = None
b_fc = None
relations = None
predicateNames = None
predicateWordsList = None
matricesLoaded = False
dir = REP_NET_PARAMS_DIR #"/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/relationsdata/"
prefix = "models/params_REP1_b/"
postfix = "_50"
questionMat = dir + prefix+ "Wq"+postfix+".mat"
phraseMat = dir + prefix+ "Wp"+postfix+".mat"
word1Mat = dir + prefix+ "W1"+postfix+".mat"
word2Mat = dir + prefix+ "W2"+postfix+".mat"
fcMat = dir + prefix+ "W_fc"+postfix+".mat"
biasl1Mat = dir + prefix+ "b_l1"+postfix+".mat"
biasFCMat = dir + prefix+ "b_fc"+postfix+".mat"
allrelations = dir + "models/"+ "relations.txt"

ffnet = None

questionParams = "parameters_Wq.txt"
phraseParams = "parameters_Wp.txt"
word1Params = "parameters_W1.txt"
word2Params = "parameters_W2.txt"
fcParams = "parameters_Wfc.txt"

predicatesFile = dir + "data/"+ "processedFAPredicates.txt"
