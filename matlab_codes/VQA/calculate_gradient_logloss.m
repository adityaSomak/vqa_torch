function [ Wp, Wq, W1, W2, W_fc ] = calculate_gradient_logloss( Wp, Wq, W1, W2, W_fc, ...
    delWp, delWq, delW1, delW2, delFC_q, delFC_p, start, end1, lr)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

delFC_q = delFC_q/(end1-start);
delFC_p = delFC_p/(end1-start);
%%% normlaize/average update_W_.
W_fc = W_fc - lr * (delFC_q + delFC_p);

delWq = delWq/(end1-start);
delWp = delWp/(end1-start);
Wp = Wp - lr * delWp;
Wq = Wq - lr * delWq;

delW1 = delW1/(end1-start);
%%% normlaize/average update_W_.
W1 = W1 - lr * delW1;

delW2 = delW2/(end1-start);
%%% normlaize/average update_W_.
W2 = W2 - lr * delW2;
end