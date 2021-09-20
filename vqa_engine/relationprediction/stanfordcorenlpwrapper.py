from pycorenlp import StanfordCoreNLP
from subprocess import call
from multiprocessing import Pool
import sys

def getDependentPairs(sentence):
    global nlp
    text = (sentence)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    enhanced = output['sentences'][0]['enhancedDependencies']
    posDict = {}
    i=0
    for tok in output['sentences'][0]['tokens']:
        posDict[str(tok['word'])] = str(tok['pos'])
        posDict[i] = (str(tok['pos']),str(tok['word']))
        i +=1

    pairs =[]
    for dep in enhanced:
        g = str(dep['governorGloss'])
        gPos = str(dep['governor'])
        d = str(dep['dependentGloss'])
        dPos = str(dep['dependent'])
        # TODO: Take only nouns, adjectives, PRPs
        if g != 'ROOT' and d != 'ROOT':
            pairs.append((g+"-"+gPos,d+"-"+dPos))
    return [pairs,posDict, i]


#def startServer(argv):
#    call(['./corenlp_start.sh'], shell=True)

#p = Pool()
#p.map(startServer,sys.argv)
nlp = StanfordCoreNLP('http://localhost:9000')
getDependentPairs('The fox jumped over the rabbit')
#p.close()