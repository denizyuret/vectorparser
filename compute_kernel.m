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
% if m.cache is defined and not empty it will be used for lookup.
%
% Implementation notes:
% - Handle multiple sv blocks outside.  Do it using two models and adding results.
% - Handle multiple x blocks and memory checking in here.

  maxram = 32e9;          % max ram to use (32GB) if no gpu
  nd = size(x, 1);        % dimensionality of instances / support vectors
  nx = size(x, 2);        % number of instances
  nc = size(m.beta, 1);   % number of classes
  ns = size(m.beta, 2);   % number of support vectors
  scores = zeros(nc, nx);
  todo = true(1, nx);

  if isfield(m, 'cache') && ~isempty(m.cache)
    for i=1:size(x, 2)
      s = m.cache.get(x);
      if ~isempty(s)
        scores(:,i) = s;
        todo(i) = false;
      end
    end
    xtodo = x(:,todo);
    nxtodo = sum(todo);
  else
    xtodo = x;
    nxtodo = nx;
  end

  if nxtodo == 0
    % either nothing to do or all was cache hit
    return
  end

  hp = m.kerparam;
  if isfield(m, 'average') && m.average
    beta = m.beta2;
  else
    beta = m.beta;
  end

  % See how much memory we can use
  if isfield(m, 'gpu') && m.gpu
    gdev = gpuDevice();
    maxnumel = gdev.FreeMemory / 8;	% use all gpu memory
  else
    gdev = [];
    maxnumel = maxram / 8; 		% use 32GB memory
  end

  % Figure out the maximum batch size nk
  nk = floor(maxnumel / (2*ns + nd) - 20);
  if (nk < 1) error('compute_kernel: Out of memory'); end;

  fxtodo = zeros(nc, nxtodo);
  for i=1:nk:nxtodo
    j = min(nxtodo, i+nk-1);
    xij = xtodo(:,i:j);
    if ~isempty(gdev)
      xij = gpuArray(xij);
    end
    if strcmp(hp.type, 'rbf')
      x2 = sum(xij .^ 2, 1);
      s2 = sum(m.svtr .^ 2, 2);
      fxtodo(:,i:j) = gather(beta * exp(-hp.gamma * bsxfun(@plus, x2, bsxfun(@plus, s2, -2 * (m.svtr * xij)))));
      clear x2 s2;
    elseif strcmp(hp.type, 'poly')
      if size(xij, 2) == 1
        % This hack roughly doubles speed on gpu if x is a single vector, sorry:
        fxtodo(:,i:j) = gather(sum(bsxfun(@times, beta', (hp.gamma * (m.svtr * xij) + hp.coef0).^hp.degree),1))';
      else
        fxtodo(:,i:j) = gather(beta * (hp.gamma * (m.svtr * xij) + hp.coef0) .^ hp.degree);
      end
    else
      error('Unknown kernel type %s', hp.type);
    end  % if strcmp(hp.type, xxx)
    clear xij;
  end  % for i=1:nk:nx
  scores(:,todo) = fxtodo;

end  % compute_kernel

