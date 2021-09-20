function [Y, dY] = softmaxlogloss(X,c,dzdy)

if(length(size(X))>2),idx_c=3;end %cnn
if(length(size(X))<=2),idx_c=1;end%mlp

n_class=size(X,idx_c);
%disp(n_class);

Xmax = max(X,[],idx_c) ;
%disp(size(Xmax));
E = exp(bsxfun(@minus, X, Xmax)) ;
S=sum(E,idx_c);
GT_idx=c(:)'+n_class*[0:size(X,idx_c+1)-1];%ground truth idx   
%disp(GT_idx);
%disp(reshape(X(GT_idx),size(Xmax)));
if ~exist('dzdy','var')||isempty(dzdy)
    %forward
    logS=log(S) +Xmax;
    Y = logS -reshape(X(GT_idx),size(Xmax));%-log(p) --- the log loss
    %Y = sum(Y,idx_c+1);% sum of batch loss
else
    %bp
    dY = bsxfun(@rdivide, E, S) ;
    dY(GT_idx)=dY(GT_idx)-1;
    if dzdy~=1.0, dY = bsxfun(@times, dY, dzdy);end
end