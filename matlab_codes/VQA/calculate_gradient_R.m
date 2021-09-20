function [ R_pred ] = calculate_gradient_R(outr_meta, outq_rmeta, ...
    outp_rmeta, R_pred, R_pred_ids, relations, start, end1, lr)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%update_R = zeros(end1-start+1,size(R_pred,2));
for iter=1:(end1-start)
 %% normlaize/average update_R
 update_R = (outq_rmeta(iter,:) + outp_rmeta(iter,:) + outr_meta(iter,:));
 R_pred(iter,:) = R_pred(iter,:) - lr * update_R;
 % UNUSED: This is a K-Means like step, which assigns the closest
 % relation for this sample
 %R_pred(start+iter-1,:) = assignClosestRelation(R_pred(start+iter-1,:), relations);
 % Updating Relation Embedding
 relations(R_pred_ids(start+iter-1),:) = R_pred(iter,:);
end
end
