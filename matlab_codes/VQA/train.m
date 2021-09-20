% Training File
%phTrnDataFileName = '/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/relationsdata/data/trnDataVectorIndices.txt';
%wordVecTextFName = '/windows/drive2/For PhD/KR Lab/DATASETS/GoogleNews-vectors-negative300.txt';
%relationsFileName = '/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/relationsdata/data/vgAndQCPredicates.txt';
%[E_Q, E_W1, E_W2, E_P, R_pred, relations] = loadInputFiles(phTrnDataFileName, relationsFileName, wordVecTextFName);

%dir = '/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/relationsdata/data/matrices/';
dir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/trainingData/matrices/';
E_Q = loadMatrix([dir 'E_Q.txt'],-1);
E_W1 = loadMatrix([dir 'E_W1.txt'],-1);
E_W2 = loadMatrix([dir 'E_W2.txt'],-1);
E_P = loadMatrix([dir 'E_P.txt'],-1);
R_pred = loadMatrix([dir 'R_pred.txt'],-1);
R_pred_ids = dlmread([dir 'R_pred_id.txt']);
%dir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/trainingData/matrices_nn/';
% E_Q = [];
% E_W1 = [];
% E_W2 =[];
% E_P= [];
% R_pred = [];
% R_pred_ids = [];
% for i=0:3
%     E_Q = [E_Q; loadMatrix([dir 'E_Q_' int2str(i) '.txt'],-1)];
%     E_W1 = [E_W1; loadMatrix([dir 'E_W1_' int2str(i) '.txt'],-1)];
%     E_W2 = [E_W2; loadMatrix([dir 'E_W2_' int2str(i) '.txt'],-1)];
%     E_P = [E_P; loadMatrix([dir 'E_P_' int2str(i) '.txt'],-1)];
%     R_pred = [R_pred; loadMatrix([dir 'R_pred_' int2str(i) '.txt'],-1)];
%     R_pred_ids = [R_pred_ids; dlmread([dir 'R_pred_id_' int2str(i) '.txt'])];
% end
R_pred_ids = R_pred_ids+1;
relations = loadMatrix([dir 'relations.txt'],-1);

%relations = loadAllRelations();
fprintf('Input files loaded\n');

vectordim = 300;

% Initialize parameter vectors
W_ = -0.25+(0.25+0.25)*rand(vectordim,vectordim);
Wp = -0.25+(0.25+0.25)*rand(vectordim,vectordim);
Wq = -0.25+(0.25+0.25)*rand(vectordim,vectordim);

% Initialize/load hyper-parameters
samples = size(E_Q,1);
batchSize = 25;
epochs = 30;%50;
lr = 0.0001;
start = 0;

% start training
end1 = start+batchSize;
turnoff = 1;
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
            [outq_p, outq_r, outp_r, outq_pmeta, outq_rmeta, outp_rmeta] = calculate_cost(E_Q, E_W1, E_W2, E_P, R_pred, ...
                Wp, Wq, W_, start, end1);
            [ Wp, Wq, W_ ] = calculate_gradient( E_Q, E_W1, E_W2, E_P,  ...
                outq_pmeta, outq_rmeta, outp_rmeta, Wp, Wq, W_, start, end1, lr);
            cost = mean(outq_p)+mean(outq_r)+mean(outp_r);
            fprintf('Mean cost of batch %d/%d = %f\n',epoch, batch, cost);
        end
    end
end
% save paramters
dlmwrite([dir 'parameters_Wp.txt'],Wp);
dlmwrite([dir 'parameters_Wq.txt'],Wq);
dlmwrite([dir 'parameters_W_.txt'],W_);