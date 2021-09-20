function [ R_closest,id ] = assignClosestRelation(R_predicted, relations)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%k = dsearchn(relations, R_predicted);
[m,id] = min(pdist2(relations, R_predicted, 'euclidean'));
R_closest = relations(id,:);
end

