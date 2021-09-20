function [ Wp, Wq, W_ ] = calculate_gradient( E_Q, E_W1, E_W2, E_P,  ...
    outq_pmeta, outq_rmeta, outp_rmeta, Wp, Wq, W_, start, end1, lr)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

update_Wp = zeros(300,300);
update_Wq = zeros(300,300);
for iter=1:(end1-start)
    for j=1:size(Wp,2)
        update_Wq(:,j) = update_Wq(:,j) + (outq_rmeta(iter,j) * E_Q(start+iter-1,:) + ...
            outq_pmeta(iter,j) * E_Q(start+iter-1,:))';
        
         update_Wp(:,j) = update_Wp(:,j) + (outp_rmeta(iter,j) * E_P(start+iter-1,:) + ...
            outq_pmeta(iter,j) * (-E_P(start+iter-1,:)))';
    end
end
update_Wq = update_Wq/(end1-start);
update_Wp = update_Wp/(end1-start);
%%% normlaize/average update_W.
Wp = Wp - lr * update_Wp;
Wq = Wq - lr * update_Wq;


update_W_ = zeros(300,300);
for iter=1:(end1-start)
    for j=1:size(W_,2)
        update_W_(:,j) = update_W_(:,j) + ...
            (outq_rmeta(iter,j) + outp_rmeta(iter,j)) * ...
            (E_W1(start+iter-1,:) + E_W2(start+iter-1,:))';
    end
end
update_W_ = update_W_/(end1-start);
%%% normlaize/average update_W_.
W_ = W_ - lr * update_W_;
end

