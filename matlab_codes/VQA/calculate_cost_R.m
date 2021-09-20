function [ out_r, outq_r, outp_r, outr_meta, outq_rmeta, outp_rmeta ] = calculate_cost_R(E_Q, E_W1, E_W2, ...
    E_P, R_pred, Wp, Wq, W_, relations, start, end1)
% Keep W, W_ fixed here
% Calculate C = avg_i||relation_i - R_pred|| 
%             + ||Wq(E_Q)+W_(E_W1+E_W2)-R_pred||
%             + ||Wp(E_P)+W_(E_W1+E_W2)-R_pred||
% TODO: check other components, negative sampling
d1 = E_Q(start:end1,:)* Wq +  (E_W1(start:end1,:) + E_W2(start:end1,:)) * W_; 
%%%% d1 = logsig(d1);
% batchSize * vecsize 
d2 = E_P(start:end1,:)* Wp +  (E_W1(start:end1,:) + E_W2(start:end1,:)) * W_; 
%%%% d2 = logsig(d2);
% batchSize * vecsize

out_r = zeros(end1-start+1,1);
outq_r = zeros(end1-start+1,1);
outp_r = zeros(end1-start+1,1);

outr_meta = zeros(end1-start,size(relations,2));
outq_rmeta = zeros(end1-start,size(relations,2));
outp_rmeta = zeros(end1-start,size(relations,2));
%fprintf('Size of R_pred: %d,%d\n', size(R_pred,1), size(R_pred,2));
for iter=1:(end1-start)
    %sum = 0;
    %fprintf('indices: start: %d, iter: %d\n', start, iter);
    %sum1 = sum(pdist2(relations, R_pred(start+iter-1,:)).^2 * 0.5);
    %outr_meta(iter,:) = sum(-bsxfun(@minus, relations, R_pred(start+iter-1,:)),1);
    %sum1 = sum1/size(relations,1);
    %out_r(iter) = out_r(iter) + sum1;
    %outr_meta(iter,:) = outr_meta(iter,:)/size(relations,1);
    
    outq_r(iter) = outq_r(iter) + 0.5 * norm(d1(iter,:)-R_pred(iter,:),2);
    outq_rmeta(iter,:) = d1(iter,:)-R_pred(iter,:);
    %%%% outq_rmeta(iter,:) = (d1(iter,:)-R_pred(iter,:)).* ...
    %%%%     ( d1(iter,:) .* (1-d1(iter,:)) );
    
    outp_r(iter) = outp_r(iter) + 0.5 * norm(d2(iter,:)-R_pred(iter,:),2);
    outp_rmeta(iter,:) = d2(iter,:)-R_pred(iter,:);
    %%%% outp_rmeta(iter,:) = (d2(iter,:)-R_pred(iter,:)).* ...
    %%%% ( d2(iter,:) .* (1-d2(iter,:)) );
end
end

