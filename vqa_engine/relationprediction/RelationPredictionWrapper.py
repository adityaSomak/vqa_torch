import math

import networkx as nx
import numpy as np
from pslplus.core.similarity import W2VPredicateSimilarity as w2v
from scipy.spatial import distance

import RelationPrediction
import stanfordcorenlpwrapper
import szClrLocDictionary
from util import CompoundNounVocab
from util import nltk_util


def getUniquesObjectsInTheBox(allObjects, i_box):
    objects = []
    for obj in allObjects:
        if allObjects[3] == i_box:
            objects.append(obj)
    return objects


def getUniqueObject(uniqueObjectsInBox, nounPairs):
    IDs = []
    for n in nounPairs:
        flag = False
        for uObj in uniqueObjectsInBox:
            if w2v.word2vec_model.similarity(uObj[0],n) > 0.8:
                IDs.append(uObj[4])
                flag = True
                break
        if not flag:
            IDs.append(-1)
    return IDs

def getWhWords(questionposDict, numWords):
    words = {"how much", "how many", "which", "why", "what",
             "how", "where","who", "are", "was", "can",
             "could", "should", "do", "does", "has", "is"}
    firstWord = questionposDict[0][1].lower()
    first2Words = firstWord+" "+questionposDict[1][1].lower()
    if first2Words in words:
        return [first2Words,2]
    if firstWord in words:
        return [firstWord,1]
    for i in range(0,numWords):
        if questionposDict[i][0].startswith("W"):
            return [questionposDict[i][1].lower(),i+1]
    return [firstWord,1]

def getRelationsFromQuestion(question):
    relationTriplets = []

    [nounPairs, posDict, numWords] = getNounPairsUsingParsing(question, True)
    [whWords, lastPos] = getWhWords(posDict, numWords)
    questionWs = nltk_util.word_tokenize(question)
    indicatorVocab = getW2VVocabIndicator(questionWs, w2v.word2vec_model)
    for nounPair in nounPairs:
        #print nounPair
        dependency_path = None
        positions = map(lambda x: int(x[x.rindex("-") + 1:]), nounPair[:2])
        #if positions[0] == lastPos:
        if len(nounPair) > 2 and len(nounPair[2]) > 2:
            dependency_path = nounPair[2][1:-1]
            print ",".join(nounPair[:2])
            print ",".join(dependency_path)
        [relation, confidence] = RelationPrediction.predictRelationUsingHeuristics(
            nounPair[:2], questionWs, indicatorVocab, w2v.word2vec_model, dependency_path)
        #else:
        #    [relation, confidence] = predictRelationInSentence(question, nounPair, posDict, True)
        if relation is None:
            continue
        nounPair = replaceNounWithCompoundNouns(nounPair[:2], posDict)
        wordPair = map(lambda x: x[:x.rindex("-")], nounPair[:2])
        if positions[0] == lastPos:
            relationTriplets.append(Relation(relation, "?x", wordPair[1],
                                             -1, -1, confidence))
        elif positions[1] == lastPos:
            relationTriplets.append(Relation(relation, wordPair[0], "?x",
                                             -1, -1, confidence))
        relationTriplets.append(Relation(relation, wordPair[0], wordPair[1],
                                         -1, -1, confidence))
    return [relationTriplets, nounPairs]


def getAllCaptionParses(captions,limit=20):
    captionParses = []
    for i,caption in enumerate(captions):
        #print caption
        #[dependentPairs, posDict, numWords] = stanfordcorenlpwrapper.getDependentPairs(str(caption))
        [dependentPairs, posDict, numWords] = getNounPairsUsingParsing(str(caption), False)
        if dependentPairs is not None:
            captionParses.append((dependentPairs, posDict, numWords))
        if i > 20:
            break
    return captionParses

def getRelationsFromCaptions(captions, captionParses, boxes, allObjects):
    if w2v.word2vec_model is None:
        w2v.loadW2VModel()
        print "word2vec model loaded...."
    relationTriplets = []
    for i, caption in enumerate(captions):
        #uniqueObjectsInBox = getUniquesObjectsInTheBox(allObjects, i)
        [nounPairs, posDict, numWords] = captionParses[i]
        captionWs = nltk_util.word_tokenize(caption)
        indicatorVocab = getW2VVocabIndicator(captionWs,w2v.word2vec_model)
        for nounPair in nounPairs:
            positions = map(lambda x: int(x[x.rindex("-") + 1:]), nounPair[:2])
            postags = (posDict[positions[0]-1][0], posDict[positions[1]-1][0])
            if not validPosTagsCheck(postags):
                continue
            relation = None
            confidence = 0
            [relation, confidence] = RelationPrediction.predictRelationUsingHeuristics(
                nounPair[:2], captionWs, indicatorVocab, w2v.word2vec_model)
            #[relation, confidence] = predictRelationInSentence(caption, nounPair, posDict, False)
            if relation is None:
                continue
            #nounPair = replaceNounWithCompoundNouns(nounPair, posDict)
            #IDs = getUniqueObject(uniqueObjectsInBox, nounPairs)
            relationTriplets.append(Relation(relation,nounPair[0],nounPair[1],\
                                             -1, -1, confidence))
    return relationTriplets


def replaceNounWithCompoundNouns(nounPair, posDict):
    # replace noun with compound nouns if found
    positionsZ = map(lambda x: int(x[x.rindex("-") + 1:])-1, nounPair[:2])
    n1_firstWord = posDict[positionsZ[0]][1].lower()

    if abs(positionsZ[1]- positionsZ[0]) == 1:
        return nounPair
    if positionsZ[0] >= 1:
        n1_phrase = posDict[positionsZ[0]-1][1].lower()+" "+n1_firstWord
        if CompoundNounVocab.isCompoundNoun(n1_phrase):
            nounPair = (n1_phrase+"-"+str(positionsZ[0]+1), nounPair[1])
    if positionsZ[1] >= 1:
        n2_firstWord = posDict[positionsZ[1]][1].lower()
        n2_phrase = posDict[positionsZ[1]-1][1].lower() + " " + n2_firstWord
        if CompoundNounVocab.isCompoundNoun(n2_phrase):
            nounPair = (nounPair[0], n2_phrase + "-" + str(positionsZ[1]+1))
    return nounPair

def validPosTagsCheck(postags):
    for postag in postags:
        if postag.startswith("N") or postag.startswith("J"):
            continue
        else:
            return False
    return True

def getW2VVocabIndicator(initialWs, w2v_model):
    indicator = np.zeros((len(initialWs),1))
    for i,W in enumerate(initialWs):
        if W == "of":
            indicator[i] = 2
            continue
        if W in w2v_model.vocab:
            indicator[i] = 1
    return indicator


def getNounPairsUsingParsing(caption, question=True):
    #print caption
    caption = caption.replace("<UNK>","")
    if caption.strip() == "":
        return [None, None, None]
    [wordPairsInPhrase, posDict, numWords] = stanfordcorenlpwrapper.getDependentPairs(caption)
    # TODO: do shortest path, get all connected pairs from the dependencies
    G = nx.Graph()
    nodes = set()
    for wordpair in wordPairsInPhrase:
        G.add_edge(wordpair[0], wordpair[1])
        nodes.add(wordpair[0])
        nodes.add(wordpair[1])
    nprps = []
    for index in range(0,numWords):
        posTag = posDict[index][0]
        if posTag.startswith("N") or posTag == "PRP" or \
                posTag.startswith("J"):
            nprps.append((posDict[index][1], posTag, index+1))
        if question and posTag.startswith("W"):
            nprps.append((posDict[index][1], posTag, index + 1))
    nounPairs = []
    pairsSeen = set()

    for j in range(0, len(nprps)):
        src_node = nprps[j][0] + "-" + str(nprps[j][2])
        if not nprps[j][1].startswith("N"):
            continue
        # print pairsSeen
        for i in range(0, len(nprps)):
            target_node = nprps[i][0] + "-" + str(nprps[i][2])
            reversePair = str(i) + "," + str(j)
            # dont need relations between
            # what-kind, what-color etc.
            if question and abs(nprps[i][2]-nprps[j][2]) == 1 \
                    and nprps[i][1].startswith("W"):
                continue
            if i == j or nprps[i][0] == nprps[j][0] or (reversePair in pairsSeen):
                continue
            # print "Considering:"+ src_node+" "+ target_node
            sh_path = []
            try:
                if (src_node in nodes) and (target_node in nodes):
                    sh_path = nx.shortest_path(G, source=src_node, target=target_node)
            except nx.NetworkXNoPath, nx.exception.NetworkXError:
                sh_path = []
            if sh_path and len(sh_path) - 2 < 3:
                if (not nprps[i][1].startswith("W")) and \
                                len(sh_path) > 2:
                    for name in sh_path:
                        try:
                            pos = int(name.split("-")[-1])-1
                            if posDict[pos][0].startswith("V"):
                                nounPairs.append((src_node, target_node))
                                pairsSeen.add(str(j) + "," + str(i))
                                break
                        except ValueError:
                            print ",".join(sh_path)
                            raise Exception
                else:
                    nounPairs.append((src_node, target_node, sh_path))
                    pairsSeen.add(str(j) + "," + str(i))
    #print nounPairs
    if len(nounPairs) == 0 and question:
        return [wordPairsInPhrase, posDict, numWords]
    return [nounPairs, posDict, numWords]

def onlyadjectivesBetween(positions, postags, caption):
    minPos = min(positions)
    maxPos = max(positions)
    for i in range(minPos,maxPos):
        if postags[i][0].startswith("J") or postags[i][0] =="CC":
            continue
        else:
            return False
    return True

def predictRelationInSentence(caption, nounPair, postags, isQuestion):
    # predict relation using the matlab model
    # For consecutive words (nouns or ADJ+NN,
    #  pairs hardcode as has_property)
    positions = map(lambda x: int(x[x.rindex("-")+1:]), nounPair)
    wordPair = map(lambda x: x[:x.rindex("-")], nounPair)
    if positions[1] == positions[0]-1 or \
            onlyadjectivesBetween(positions, postags, caption):
        rel = szClrLocDictionary.matchesSzColorLocation(wordPair[1])
        if rel is not None:
            return [rel, 1.0]
        if postags[positions[0]-1][0].startswith("N") and \
                postags[positions[1]-1][0].startswith("J"):
            return ["is", 1.0]
    return RelationPrediction.predictRelation(wordPair, caption, w2v, isQuestion)

def getAllNouns(captionParse):
    nouns = []
    for i in range(0,captionParse[2]):
        [pos,word] = captionParse[1][i]
        if pos.startswith("N"):
            nouns.append(word)
    return nouns


'''
Get Spatial Relations between all object mentions based on
captions+boxes information.
   - if same object, merge the relations.
   - if different object, give them unique IDs.
'''
def getSpatialRelations(captions, captionParses, boxes, scores):
    # get spatial relations from objects in captions
    if w2v.word2vec_model is None:
        w2v.loadW2VModel()
        print "word2vec model loaded...."
    allObjects = []
    ID = 0
    for i, caption in enumerate(captions):
        nouns = getAllNouns(captionParses[i])
        for noun in nouns:
            allObjects.append((noun, boxes[i], scores[i], i, ID))
            ID += 1
    perObjectRelations = []
    for i in range(0, len(allObjects)):
        perObjectRelations.append([])
    for i, obj_i in enumerate(allObjects):
        for j, obj_j in enumerate(allObjects):
            if i != j:
                confidence = (obj_i[2]+obj_j[2])/2.0
                spatial = getSpatialRelation(obj_i[1], obj_j[1])
                if spatial is not None:
                    perObjectRelations[i].append(Relation(spatial, obj_i[0], obj_j[0],
                                                          i, j, confidence))
                    #perObjectRelations[j].append(Relation(spatial, obj_i[0], obj_j[0],
                    #                                      i, j, confidence))

    # for i, obj_i in enumerate(allObjects):
    #     for j, obj_j in enumerate(allObjects):
    #         if i != j:
    #             if isSameObject(obj_i, obj_j, perObjectRelations[i], perObjectRelations[j]):
    #                 perObjectRelations[j] = []
    #                 allObjects[j] = None

    allSpatialRelations = []
    for i, obj_i in enumerate(perObjectRelations):
        for relation in perObjectRelations[i]:
            relation.addUniqueIDS(allObjects[relation.ID1][4],allObjects[relation.ID2][4])
            allSpatialRelations.append(relation)

    return [allSpatialRelations, allObjects]


def calculateIntersection(box1, box2):
    # boxes: xywh
    xmin = max(box1[0], box2[0])
    xmax = min(box1[0]+box1[2], box2[0]+box2[2])
    ymin = max(box1[1], box2[1])
    ymax = min(box1[1] + box1[3], box2[1] + box2[3])
    intersect = (xmax-xmin)* (ymax-ymin)
    min_area = min(box1[2]*box1[3], box2[2]*box2[3])
    return intersect/float(min_area)


def isSameObject(obj_i, obj_j, i_relations, j_relations):
    # finds out if object is same base dn
    # overlap of the BBox and meaning of the words.
    if obj_i is None or obj_j is None:
        return False
    IOA = calculateIntersection(obj_i[1],obj_j[1])
    if IOA > 0.7:
        similarityBasedOnWords = w2v.word2vec_model.similarity(obj_i[0],obj_j[0])
        return similarityBasedOnWords > 0.8
    return False

def angle_between(p1, p2, ref):
    distance1r = distance.euclidean(p1,ref)
    distance2r = distance.euclidean(p2, ref)
    distance12 = distance.euclidean(p1, p2)
    return np.arccos((math.pow(distance12,2)+math.pow(distance1r,2) - math.pow(distance2r,2)) /
                  (2*distance1r*distance12)) * 180/math.pi

def getSpatialRelation(box1, box2):
    # Get spatial relation acc. to D. Elliott Thesis
    # NOTE: works if
    xmin = max(box1[0], box2[0])
    xmax = min(box1[0] + box1[2], box2[0] + box2[2])
    ymin = max(box1[1], box2[1])
    ymax = min(box1[1] + box1[3], box2[1] + box2[3])
    intersect = (xmax - xmin) * (ymax - ymin)
    Yarea = box2[2]*box2[3]
    Xarea = box1[2]*box1[3]
    if intersect/ float(Yarea) > 0.9 and Xarea > Yarea:
        return "surrounds"
    if intersect/ float(Yarea) > 0.5:
        return "on"

    ref = [box2[0]+box2[1]/2.0,box1[1]+box1[3]/2.0]
    angle = angle_between([box1[0]+box1[1]/2.0,box1[1]+box1[3]/2.0],
                          [box2[0]+box2[1]/2.0,box2[1]+box2[3]/2.0], ref)
    if (135 < angle < 225) or (15 < angle < 45):
        return "beside"
    elif 225 < angle < 315:
        return "above"
    elif 45 < angle < 135:
        return "below"
    return None

def combineRelations(semanticQRelations,spatialRelations):
    combinedRelations = []
    combinedRelations.extend(semanticQRelations)
    combinedRelations.extend(spatialRelations)
    return combinedRelations

'''
    TODO: Will be same as #getRelationsFromCaptions().
    Relations with wh-keyword will be relaced by answer.
'''
def getRelationsFromTrainingQuestion(question, answer):
    raise NotImplementedError

class Relation:
    def __init__(self, relation, word1, word2, ID1, ID2, confidence):
        self.relation = relation
        self.word1 = word1
        self.word2 = word2
        self.ID1 = ID1
        self.ID2 = ID2
        self.confidence = confidence

    def addUniqueIDS(self, ID1, ID2):
        self.ID1 = ID1
        self.ID2 = ID2

    def __str__(self):
        return "<"+",".join([self.word1,self.relation,self.word2])+">"