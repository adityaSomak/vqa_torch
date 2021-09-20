function [ E_Q, E_W1, E_W2, E_P, R_pred, relations ] = loadInputFiles(phTrnDataFileName, allRelationsFName, wordVecTextFName )
% UNTITLED2 Summary of this function goes here
% Detailed explanation goes here
[stat, num_lines] = system(['wc -l < ' phTrnDataFileName]);
fprintf('Loading word vectors ...\n');

fid = fopen(wordVecTextFName, 'rt');
formatString = ['%s ' repmat('%f ', 1, 300)];
formatString (end) = [];
w2vMat = zeros(0, 300);
textscan(fid, '%d %d',1);
blocknum = 0;
while ~feof(fid)
  data = textscan(fid, formatString, 100000);
  w2vMat = [w2vMat; cell2mat(data(2:301))];
  fprintf('%d blocks loaded\n',blocknum);
  blocknum = blocknum+1;
end
fclose(fid);

%w2vMat = dlmread(wordVecTextFName,' ',1,1);
fprintf('Done.\n');
E_Q = zeros(num_lines,size(w2vMat,2));
E_W1 = zeros(num_lines,size(w2vMat,2));
E_W2 = zeros(num_lines,size(w2vMat,2));
E_P = zeros(num_lines,size(w2vMat,2));
R_pred = zeros(num_lines,size(w2vMat,2));

fid = fopen(phTrnDataFileName);

tline = 'c';
sampleNum=0;
while ischar(tline)
    tline = fgetl(fid);
    tokens = strsplit(tline); 
    % assumption is 0:w1-index; 1:w2-index; 2:
    % predicted-relation indices; 3: question word-indices, 4:phrase-indices 
    for i=0:4,
        wi = str2num(tokens(i));
        selectedVectors = w2vMat(wi,:);
        vector = mean(selectedVectors,1);        
        if i==0,
            E_W1(sampleNum,:) = vector;
        elseif i==1,
            E_W2(sampleNum,:) = vector;
        elseif i==2,
            R_pred(sampleNum,:) = vector;
        elseif i==3,
            E_Q(sampleNum,:) = vector;    
        elseif i==4,
            E_P(sampleNum,:) = vector;
        end
    end
    if mod(sampleNum,1000) == 0,
        fprintf('%d samples loaded\n',sampleNum);
    end
    sampleNum = sampleNum+1;
end
fclose(fid);

[stat, num_lines_rel] = system(['wc -l < ' allRelationsFName]);
fprintf('Loading relations-file\n');
relations = zeros(num_lines_rel,size(w2vMat,2));
fid = fopen(allRelationsFName);
tline = 'c';
lineNum=0;
while ischar(tline),
    tline = fgetl(fid);
    wi = str2num(tline);
    selectedVectors = w2vMat(wi,:);
    vector = mean(selectedVectors,1); 
    relations(lineNum,:) = vector;
end
clearvars w2vMat
end

