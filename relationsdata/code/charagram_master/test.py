import utils

from charagram_model import charagram_model
from params import params
import lasagne
import evaluate

def str2act(v):
    if v is None:
        return v
    if v.lower() == "tanh":
        return lasagne.nonlinearities.tanh
    if v.lower() == "linear":
        return lasagne.nonlinearities.linear
    if v.lower() == "relu":
        return lasagne.nonlinearities.rectify
    raise ValueError('A type that was supposed to be a learner is not.')


examples = utils.get_ppdb_data('../data/ppdb-xl-phrasal-preprocessed.txt')
params= params()
params.domain = 'sentence'; 
params.nntype='charagram';
params.train='./data/ppdb-xl-phrasal-preprocessed.txt';
params.evaluate='True';
params.numlayers=1;
params.act=str2act('tanh');
params.loadmodel='../data/charagram_phrase.pickle'
params.worddim=200;
params.featurefile='../data/charagram_phrase_features_234.txt';
params.cutoff=0
params.margin=0.4;
params.type='MAX'
params.clip=None
params.learner=lasagne.updates.adam

model = charagram_model(params)
examples = evaluate.read_data_sentences('../data/annotated-ppdb-dev',False)

g1 = []
g2 = []
g1.append(model.hash(examples[0][0]))
g2.append(model.hash(examples[0][1]))
e1 = model.feedforward_function(g1)
e2 = model.feedforward_function(g2)

model.scoring_function(e1,e2)
