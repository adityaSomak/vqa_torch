%dir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/trainingData/matrices_nn/';
%E_Q = loadMatrix([dir 'E_Q_0.txt'],1);
%E_W1 = loadMatrix([dir 'E_W1_0.txt'],1);
%E_W2 = loadMatrix([dir 'E_W2_0.txt'],1);
dir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/trainingData/matrices/';
E_Q = loadMatrix([dir 'E_Q.txt'],1);
E_W1 = loadMatrix([dir 'E_W1.txt'],1);
E_W2 = loadMatrix([dir 'E_W2.txt'],1);
Wp = dlmread([dir 'parameters_Wp.txt']);
%Wq = dlmread([dir 'parameters_Wq.txt']);
W_ = dlmread([dir 'parameters_W_.txt']);
relations = loadMatrix([dir 'relations.txt'],-1);

Rpred = E_Q(1:500,:)* Wq +  (E_W1(1:500,:) + E_W2(1:500,:)) * W_;
fileID = fopen('op_tr1.txt','w');

dir = '/home/ASUAD/saditya1/Desktop/DATASETS/Visual_Genome/relationPrediction/';
preds= textread([dir 'vgAndQCPredicates.txt'], '%s', 'delimiter', '\n');

for i=1:500
	[rel,id] = assignClosestRelation(Rpred(i,:), relations);
	fprintf(fileID,'%d\t%s\n', id, preds{id});
end
fclose(fileID);