function [ lossq_r, lossp_r, delFC_q, delFC_p, delWq, delWp, delW1, delW2 ] = calculate_logloss(E_Q, E_W1, ...
    E_W2, E_P, R_pred_id, Wp, Wq,  W1, W2, W_fc, start, end1)
% Keep R_pred fixed here
y1 = E_Q(start:end1,:)* Wq +  E_W1(start:end1,:) * W1 + E_W2(start:end1,:) * W2; 
% batchSize * vecsize 
y2 = E_P(start:end1,:)* Wp +  E_W1(start:end1,:) * W1 + E_W2(start:end1,:) * W2; 
% batchSize * vecsize

z1 = y1* W_fc;
z2 = y2* W_fc;

%delFC_q = zeros(size(W_fc,1),size(W_fc,2));
%delFC_p = zeros(size(W_fc,1),size(W_fc,2));

[lossq_r, dFC] = softmaxlogloss(z1',R_pred_id,single(1.0));
[lossp_r, dFC] = softmaxlogloss(z2',R_pred_id,single(1.0));

o_q = softmax(z1',R_pred_id,[]);
o_p = softmax(z2',R_pred_id,[]);

delFC_q = y1.'*(o_q- R_pred_id);
delFC_p = y2.'*(o_p- R_pred_id);
%for i=1:batchSize
%	delFC_q = delFC_q + reshape(y1(i,:),[size(W_fc,1),1]) * reshape((o_q- R_pred_id)(i,:),[1,size(W_fc,2)]);
	% vecsize * 1   DOT 1 * relnumber
%	delFC_p = delFC_p + reshape(y2(i,:),[size(W_fc,1),1]) * reshape((o_p- R_pred_id)(i,:),[1,size(W_fc,2)]);
%end

delWq = (W_fc * (o_q- R_pred_id)') *  E_Q(start:end1,:);
delWp = (W_fc * (o_q- R_pred_id)') *  E_P(start:end1,:);

delW1 = (W_fc * (o_q- R_pred_id)' + W_fc * (o_p- R_pred_id)') *  E_W1(start:end1,:);
delW2 = (W_fc * (o_q- R_pred_id)' + W_fc * (o_p- R_pred_id)') *  E_W2(start:end1,:);

%for i=1:batchSize
	%delWq = delWq + (W_fc * (o_q- R_pred_id)(i,:)')' *  E_Q(start+i-1,:);
	%delWp = delWp + (W_fc * (o_q- R_pred_id)(i,:)')' *  E_P(start+i-1,:);

	%delW1 = delW1 + (W_fc * (o_q- R_pred_id)(i,:)' + W_fc * (o_p- R_pred_id)(i,:)')' *  E_W1(start+i-1,:);
	%delW2 = delW2 + (W_fc * (o_q- R_pred_id)(i,:)' + W_fc * (o_p- R_pred_id)(i,:)')' *  E_W2(start+i-1,:);
%end

end