from util import nltk_util
import RelationPredictionWrapper, RelationPrediction
from pslplus.core.similarity import W2VPredicateSimilarity as w2v

def addPosition(sentence, nounPair):
    words = nltk_util.word_tokenize(sentence)
    positions = []
    for word in nounPair:
        for i,wordInS in enumerate(words):
            if wordInS == word:
                positions.append(i+1)
                break
    return words,[nounPair[0]+"-"+str(positions[0]),nounPair[1]+"-"+str(positions[1])]


def testAccuracy():
    devFile = "/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/" \
              "relationsdata/data/Relations_Annotation\ -\ First5000Samples.tsv"
    #devFile = "/data/somak/VQA/VQA_torch/" \
    #          "relationsdata/data/Relations_Annotation\ -\ First5000Samples.tsv"

    w2v.loadW2VModel()

    accuracy = 0
    wnAccuracy = 0
    number = 0
    with open(devFile,'r') as devF:
        i = 0
        for line in devF:
            tokens = line.split("\t")
            if tokens[2].strip() == "X":
                continue
            sentence = tokens[0]
            i+= 1
            print str(i)+"::"+ sentence
            nounPair = map(lambda x: x.replace("\'",""), tokens[1].strip().split(",")[1:-1])
            words, nounPair = addPosition(sentence, nounPair)
            indicatorVocab = RelationPredictionWrapper.getW2VVocabIndicator(words, w2v.word2vec_model)
            [relation, confidence] = RelationPrediction.predictRelationUsingHeuristics(
                nounPair[:2], words, indicatorVocab, w2v.word2vec_model)

            if relation.lower() == tokens[2].strip():
                accuracy += 1
            wnAccuracy += nltk_util.semantic_similarity(nltk_util.word_tokenize(relation),
                                                        nltk_util.word_tokenize(tokens[2]),True)
            number +=1

    accuracy = accuracy/float(number)
    wnAccuracy = wnAccuracy/float(number)
    print "Stats: accuracy"+ str(accuracy) + ", Wordnet acc:"+ str(wnAccuracy)
