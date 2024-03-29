function [ outq_p, outq_r, outp_r, outq_pmeta, outq_rmeta, outp_rmeta ] = calculate_cost(E_Q, E_W1, ...
    E_W2, E_P, R_pred, Wp, Wq,  W_, start, end1)
% Keep R_pred fixed here
% Calculate C = ||Wq(E_Q)-Wp(E_P)||
% w11*Eq1+w12*Eq2+...+w1n*Eqn
% ...
% wm1*Eq1+wm2*Eq2+...+wmn*Eqn
%             + ||Wq(E_Q)+W_(E_W1+E_W2)-R_pred||
%             + ||Wp(E_P)+W_(E_W1+E_W2)-R_pred||
d1 = E_Q(start:end1,:)* Wq +  (E_W1(start:end1,:) + E_W2(start:end1,:)) * W_; % batchSize * vecsize 
%%%% d1 = logsig(d1);
d2 = E_P(start:end1,:)* Wp +  (E_W1(start:end1,:) + E_W2(start:end1,:)) * W_; % batchSize * vecsize
%%%% d2 = logsig(d2);

outq_p = zeros(end1-start+1,1);
outq_r = zeros(end1-start+1,1);
outp_r = zeros(end1-start+1,1);

outq_pmeta = zeros(size(d1,1),size(d1,2));
outq_rmeta = zeros(size(d1,1),size(d1,2));
outp_rmeta = zeros(size(d1,1),size(d1,2));
for iter=1:(end1-start)
    outq_p(iter) = 0.5 * norm(d1(iter,:)-d2(iter,:),2);
    outq_pmeta(iter,:) = d1(iter,:)-d2(iter,:); % dWp, dWq, dW_
    %%%% outq_pmeta(iter,:) = (d1(iter,:)-d2(iter,:)).* ...
    %%%%     ( d1(iter,:).* (1-d1(iter,:)) - d2(iter,:).*(1-d2(iter,:)) );
    
    outq_r(iter) = 0.5 * norm(d1(iter,:)-R_pred(iter,:),2);
    outq_rmeta(iter,:) = d1(iter,:)-R_pred(iter,:); % dWq, dW_
    %%%% outq_pmeta(iter,:) = (d1(iter,:)-R_pred(iter,:)).* ...
    %%%%     ( d1(iter,:).*(1-d1(iter,:)) );
    
    outp_r(iter) = 0.5 * norm(d2(iter,:)-R_pred(iter,:),2);
    outp_rmeta(iter,:) = d2(iter,:)-R_pred(iter,:); % dWp, dW_
    %%%% outq_pmeta(iter,:) = (d2(iter,:)-R_pred(iter,:)).* ...
    %%%%     ( d2(iter,:).*(1-d2(iter,:)) );
end

end

