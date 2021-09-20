function y =softmax(x,dzdy)

if(length(size(x))>2),idx_c=3;end%cnn
if(length(size(x))<=2),idx_c=1;end%mlp

E = exp(bsxfun(@minus, x, max(x,[],idx_c))) ;
S = sum(E,idx_c) ;

y = bsxfun(@rdivide, E, S) ;

if isempty(dzdy), return ; end

% backward
y = y .* bsxfun(@minus, dzdy, sum(dzdy .* y, idx_c)) ;
