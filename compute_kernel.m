% TODO: profile
% TODO: test with perceptron, vectorparser, beamparser,
% with/without cache, test, train, dump etc.
% TODO: integrate with perceptron, tparser and retest.

function scores = compute_kernel(m, x)

% compute_kernel.m  (c) Deniz Yuret, 2014
% Based on the DOGMA library by Francesco Orabona.
%
% Given a model m and an instance matrix x(nd,nx) where
% m.svtr(ns,nd): transpose of the support vector matrix
% m.beta(nc,ns): support vector coefficients
% 
% compute_kernel will return:
% scores(nc,nx) = m.beta * K(m.svtr, x)
%
% where K is the kernel function.  m.kerparam specifies the kernel 
% parameters.  Given hp = m.kerparam:
% If hp.type = 'poly': 
%   K = (hp.gamma * (m.svtr * x) + hp.coef0) .^ hp.degree
% If hp.type = 'rbf':  
%   x2 = sum(x .^ 2, 1);
%   s2 = sum(m.svtr .^ 2, 2);
%   K = exp(-hp.gamma * bsxfun(@plus, x2, bsxfun(@plus, s2, -2 * (m.svtr * x))))
%
% if m.average = 1, m.beta2 (averaged parameters) will be used
% instead of m.beta.
%
% if m.cache is not empty it will be used for lookup.
%
% Implementation notes:
% - Handle multiple sv blocks outside.  Do it using two models and adding results.
% - Handle multiple x blocks and memory checking in here.

  nc = size(m.beta, 1);   % number of classes
  nx = size(x, 2);        % number of instances
  todo = [];

  if m.usecache
    scores = zeros(nc, nx);
    todo = true(1, nx);
    for i=1:nx
      s = m.cache.get(x(:,i));
      if ~isempty(s)
        scores(:,i) = s;
        todo(i) = false;
      end
    end
    x = x(:,todo);
    nx = sum(todo);
  end

  if nx == 0
    % either nothing to do or all was cache hit
    return
  end

  hp = m.kerparam;
  if m.average
    beta = m.beta2;
  else
    beta = m.beta;
  end

  if nx == 1
    batchsize = 1;
  else
    batchsize = maxbatchsize(m);
  end

  fx = zeros(nc, nx);
  for i=1:batchsize:nx
    j = min(nx, i+batchsize-1);
    xij = x(:,i:j);
    if strcmp(hp.type, 'rbf')
      x2 = sum(xij .^ 2, 1);
      s2 = sum(m.svtr .^ 2, 2);
      fx(:,i:j) = gather(beta * exp(-hp.gamma * bsxfun(@plus, x2, bsxfun(@plus, s2, -2 * (m.svtr * xij)))));
      clear x2 s2;
    elseif strcmp(hp.type, 'poly')
      if size(xij, 2) == 1
        % This hack roughly doubles speed on gpu if x is a single vector, sorry:
        fx(:,i:j) = gather(sum(bsxfun(@times, beta', (hp.gamma * (m.svtr * xij) + hp.coef0).^hp.degree),1))';
      else
        fx(:,i:j) = gather(beta * (hp.gamma * (m.svtr * xij) + hp.coef0) .^ hp.degree);
      end
    else
      error('Unknown kernel type %s', hp.type);
    end  % if strcmp(hp.type, xxx)
  end  % for i=1:batchsize:nx

  if isempty(todo)
    scores = fx;
  else
    scores(:,todo) = fx;
  end

end  % compute_kernel



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function batchsize = maxbatchsize(m)
  if gpuDeviceCount() > 0
    gdev = gpuDevice();
    maxnumel = gdev.FreeMemory / 8;	% use all gpu memory
  else
    maxnumel = 32e9 / 8; 		% use 32GB memory
  end
  ns = size(m.svtr, 1);   % number of support vectors
  nd = size(m.svtr, 2);   % number of dimensions
  batchsize = floor(maxnumel / (2*ns + nd) - 20);
  if (batchsize < 1) error('compute_kernel: Out of memory'); end;
end