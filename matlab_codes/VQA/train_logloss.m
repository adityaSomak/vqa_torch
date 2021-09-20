% Training File

dir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/trainingData/cleaned_march/matrices/';
E_Q = loadMatrix([dir 'E_Q.txt'],-1);
E_W1 = loadMatrix([dir 'E_W1.txt'],-1);
E_W2 = loadMatrix([dir 'E_W2.txt'],-1);
E_P = loadMatrix([dir 'E_P.txt'],-1);
R_pred = loadMatrix([dir 'R_pred.txt'],-1);
R_pred_ids = dlmread([dir 'R_pred_id.txt']);
R_pred_ids = R_pred_ids+1;
relations = loadMatrix([dir 'relations.txt'],-1);

devdir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/trainingData/dev/matrices/';
E_Q_dev = loadMatrix([dir 'E_Q.txt'],-1);
E_W1_dev = loadMatrix([dir 'E_W1.txt'],-1);
E_W2_dev = loadMatrix([dir 'E_W2.txt'],-1);
E_P_dev = loadMatrix([dir 'E_P.txt'],-1);
R_pred_dev = loadMatrix([dir 'R_pred.txt'],-1);
R_pred_ids_dev = dlmread([dir 'R_pred_id.txt']);
R_pred_ids_dev = R_pred_ids_dev+1;
fprintf('Input files loaded\n');

vectordim = 300;

% Initialize parameter vectors
W1 = -0.25+(0.25+0.25)*rand(vectordim,vectordim);
W2 = -0.25+(0.25+0.25)*rand(vectordim,vectordim);
Wp = -0.25+(0.25+0.25)*rand(vectordim,vectordim);
Wq = -0.25+(0.25+0.25)*rand(vectordim,vectordim);
W_fc = -0.25+(0.25+0.25)*rand(vectordim,size(relations,1));

% Initialize/load hyper-parameters
samples = size(E_Q,1);
batchSize = 25;
epochs = 50;
lr = 0.0001;
start = 0;

% start training
end1 = start+batchSize;
turnoff = 1;
prevCosts = zeros(1,epochs);
minCost = 300000;
for epoch=1:epochs
    for batch=1:samples/batchSize
        start = batchSize *(batch-1)+1;
        end1 = batchSize * batch+1;
        fprintf('start= %d; end1 =%d\n', start, end1);
        R_pred = zeros(batchSize,size(relations,2));
        for iter=1:(end1-start)
            R_pred(iter,:) = relations(R_pred_ids(start+iter-1),:);
        end
        if turnoff == 0 && mod(epoch,2) == 0
            % For odd epochs optimize R_pred
            [out_r, outq_r, outp_r, outr_meta, outq_rmeta, outp_rmeta ] = calculate_cost_R(E_Q, E_W1, E_W2, E_P, R_pred, ...
                Wp, Wq, W_, relations, start, end1);
            R_pred = calculate_gradient_R( outr_meta, outq_rmeta, outp_rmeta, ...
                R_pred, R_pred_ids, relations, start, end1, lr);
            cost = mean(out_r)+mean(outq_r)+mean(outp_r);
            fprintf('Mean cost of batch %d/%d = %f\n',epoch, batch, cost);
        else
            % For even epochs optimize W, W_
            [lossq_r, lossp_r, delFC_q, delFC_p, delWq, delWp, delW1, delW2] = calculate_logloss(E_Q, E_W1, ...
                E_W2, E_P, R_pred_ids, Wp, Wq,  W1, W2, W_fc, start, end1);
            [ Wp, Wq, W1, W2, W_fc ] = calculate_gradient_logloss( Wp, Wq, W1, W2, W_fc, ...
                delWp, delWq, delW1, delW2, delFC_q, delFC_p, start, end1, lr);
            cost = mean(lossq_r)+mean(lossp_r);
            fprintf('Mean cost of batch %d/%d = %f\n',epoch, batch, cost);
        end
    end
    [lossq_r, lossp_r, delFC_q, delFC_p, delWq, delWp, delW1, delW2] = calculate_logloss(E_Q_dev, E_W1_dev, ...
                E_W2_dev, E_P_dev, R_pred_ids_dev, Wp, Wq,  W1, W2, W_fc, 0, size(E_Q_dev,1));
    cost = mean(lossq_r)+mean(lossp_r);
    if epoch > 15 && cost < minCost
        break
    end
    if cost < minCost, minCost = cost; end
    prevCosts(epoch) = cost;
    fprintf('Development cost after epoch %d = %f\n',epoch, cost);
end
% save paramters
disp(prevCosts);
dlmwrite([dir 'parameters_Wp.txt'],Wp);
dlmwrite([dir 'parameters_Wq.txt'],Wq);
dlmwrite([dir 'parameters_W1.txt'],W1);
dlmwrite([dir 'parameters_Wq.txt'],W2);
dlmwrite([dir 'parameters_W1.txt'],W_fc);