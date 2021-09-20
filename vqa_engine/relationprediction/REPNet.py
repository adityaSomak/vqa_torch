import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

class REPNet:
    def __init__(self,Wp, Wq, W1, W2, Wfc, b_l1, b_fc):
        q = T.dvector()
        p = T.dvector()
        w1 = T.dvector()
        w2 = T.dvector()
        r_softmax_q = layer_q(q, w1, w2, Wq, W1, W2, Wfc, b_l1, b_fc)
        r_softmax_p = layer_p(p, w1, w2, Wp, W1, W2, Wfc, b_l1, b_fc)
        self.feed_forward_q = theano.function(inputs=[q,w1,w2], outputs=r_softmax_q)
        self.feed_forward_p = theano.function(inputs=[p,w1,w2], outputs=r_softmax_p)

def layer_q(qEmb, W1Emb, W2Emb, Wq, W1, W2, Wfc, b_l1, b_fc):
    m = T.dot(Wq.T, qEmb) + T.dot(W1.T, W1Emb) + T.dot(W2.T,W2Emb) +b_l1.T
    y = T.dot(Wfc.T,m) +b_fc.T
    h = nnet.softmax(y)
    return h

def layer_p(pEmb, W1Emb, W2Emb, Wp, W1, W2, Wfc, b_l1, b_fc):
    m = T.dot(Wp.T, pEmb) + T.dot(W1.T, W1Emb) + T.dot(W2.T,W2Emb) +b_l1.T
    y = T.dot(Wfc.T,m) + b_fc.T
    h = nnet.softmax(y)
    return h