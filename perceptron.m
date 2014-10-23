function [model, scores, aer] = perceptron(X,Y,model,varargin)

% perceptron: written by Deniz Yuret, August 2, 2014.
% Multi-class, mini-batch, gpu enabled, kernel perceptron.
%
% Based on the k_perceptron_multi_train.m in the DOGMA library by
% Francesco Orabona which references:
%
%     - Crammer, K., & Singer Y. (2003).
%       Ultraconservative Online Algorithms for Multiclass Problems.
%       Journal of Machine Learning Research 3, (pp. 951-991).
%
%
% Typical usage:
%
% m0 = struct('step', 1e5, 'batchsize', 1e3, 'kerparam', 
%             struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3));
%
% m1 = perceptron(x_trn, y_trn, m0);  % first epoch
%
% m1 = perceptron(x_trn, y_trn, m1);  % for 2nd, 3rd etc. epoch.
%
% [~, scores, aer] = perceptron(x_tst, y_tst, m1, 'update', 0);  % testing
% 
%
% Input arguments:
%
% X(nd,nx) has an instance in each column.
%
% Y(1,nx) is a vector of correct answers.  Could be empty [] for testing.
%
% model can be a blank model or the result of a previous epoch, in
% which case it will have non-empty support vectors: model.SV(nd,ns),
% SV coefficients: model.beta(nc,ns) and averaged coefficients:
% model.beta2(nc,ns).
% 
% model.kerparam specifies the polynomial kernel parameters:
% hp = model.kerparam; Defaults: hp.gamma=1, hp.coef0=1, hp.degree=3, hp.type='poly'.
% scores = model.beta * (hp.gamma * (model.SV' * X) + hp.coef0) .^ hp.degree;
%
% model.batchsize gives the mini-batch size (default=1000).
%
% model.step determines how often results are printed (default=10000).
%
% The rest of the input arguments give optional parameters:
% 'gpu': 1 to use gpu, 0 not. (default is to use it if there is one).
% 'update': 1 for training (default), 0 for testing
% 'average': 1 uses averaged model (default for testing), 
%            0 uses latest coefficients (default for training).
%
% The return values:
%
% model: possibly updated model (same as input model if update==0).
% scores(nc,nx) gives the score for each class for each instance.
% aer: average error rate.
%

t0=tic;
default_batchsize = 1000;
default_step = 10000;
perceptron_init(varargin, nargout);
fprintf('nd=%d nx=%d nc=%d ns=%d\n', nd, nx, nc, ns);
fprintf('inst\terr\tnsv\tbatch\ttime\tmem\n');

i = 0; j = 0; 
j_step = m.step;

while j < nx

  i = j + 1;
  nk = real_batchsize();                % may be smaller or larger than model.batchsize
  j = min(nx, i + nk - 1);              % will process minibatch X(:,i:j)
  nk = j - i + 1;                       % in case j==nx

  score = compute_scores(m, X, i, j, opts); % score(nc,nk): scores for X(:,i:j)

  if ~isempty(Y)
    [~, z] = max(score);
    m.nerr = m.nerr + numel(find(z ~= Y(i:j)));
  end

  if nargout >= 2
    scores = [scores score];
  end

  if opts.update                        % in training mode
    yindex = Y(i:j);                    % correct answers for the batch
    ypos = sub2ind(size(score), yindex, 1:nk);
    yscore = score(ypos);               % scores for correct answers
    score(ypos) = -inf;
    [zscore, zindex] = max(score);      % max scores for other answers
    updates = find(yscore <= zscore);
    m.beta2 = m.beta2 + m.beta;

    if ~isempty(updates)
      nu = numel(updates);
      ns = ns + nu;

      check_sv_blocks(nu);
      updates_i = updates+i-1;          % updates uses (1,nk) indexing, updates_i uses (i,j) indexing
      m.svtr2 = [ m.svtr2; X(:,updates_i)' ];

      newbeta = zeros(nc, nu);
      newbeta(sub2ind(size(newbeta), yindex(updates), 1:nu)) = 1;
      newbeta(sub2ind(size(newbeta), zindex(updates), 1:nu)) = -1;

      m.beta2 = [m.beta2 newbeta];
      m.beta = [m.beta newbeta];

    end % if ~isempty(updates)

  end % if opts.update

  if j >= j_step
    fprintf('%d\t%.4f\t%d\t%d\t%.2f\t%.2e\n', j, 100*m.nerr/j, ns, nk, toc(t0), gmem);
    j_step = j_step + m.step;
  end

end % while j < nx

fprintf('%d\t%.4f\t%d\t%d\t%.2f\t%.2e\n', j, 100*m.nerr/j, ns, nk, toc(t0), gmem);

if opts.update
  model.x = X;
  model.y = Y;
  model.beta = gather(m.beta);
  model.beta2 = gather(m.beta2);
  model.SV = [ gather(m.svtr1)', gather(m.svtr2)' ];
  model.batchsize = m.batchsize;
  model = perceptron_compactify(model);
end

if nargout >= 3
  aer = m.nerr/nx;
end

clear m;


%%%%%%%%%%%%%%%%%%%%%%%%%%
function perceptron_init(varargin_save, nargout_save)

opts = struct();      % opts is a struct of options.

for vi = 1:2:numel(varargin_save)
  v = varargin_save{vi};
  v1 = varargin_save{vi+1};
  switch v
   case 'update'  
    opts.update  = v1;
   case 'average' 
    opts.average = v1;
   case 'gpu'     
    opts.gpu     = v1;
   otherwise 
    error('Usage: [model, scores] = perceptron(X, Y, model, opts)');
  end
end

if ~isfield(opts, 'update')
  opts.update = true;   % update: train the model (default), otherwise model is not updated
end
if opts.update
  fprintf('Training mode.\n');
else
  fprintf('Testing mode.\n');
end

if ~isfield(opts,'average')
  opts.average = ~opts.update;  % averaging: by default use during testing, not training
end
if opts.average
  fprintf('Using averaged model to predict.\n');
else
  fprintf('Using last model to predict.\n');
end

if ~isfield(opts, 'gpu')
  opts.gpu = gpuDeviceCount(); % Use the gpu if there is one by default
end

% Get the size of the problem
nd = size(X, 1);
nx = size(X, 2);
if (opts.update && (~isfield(model,'SV') || isempty(model.SV)))
  assert(~isempty(Y), 'Please provide Y for training.');
  fprintf('Initializing empty model.\n');
  nc = max(Y);
  ns = 0;
  model.SV = zeros(nd,0,'like',X);
  % beta and beta2 need to be double otherwise averaging fails.
  model.beta = zeros(nc,0);
  model.beta2 = zeros(nc,0);
else
  nc = size(model.beta, 1);
  ns = size(model.beta, 2);
end

assert(size(model.SV, 1) == nd);
assert(size(model.SV, 2) == ns);
assert(size(model.beta, 1) == nc);
assert(size(model.beta, 2) == ns);
assert(size(model.beta2, 1) == nc);
assert(size(model.beta2, 2) == ns);

% Get the kernel parameters
if ~isfield(model,'kerparam') && opts.update
  fprintf('Using default kernel: gamma=1, coef0=1, degree=3, type=poly.\n');
  model.kerparam = struct('type','poly','degree',3,'gamma',1,'coef0',1);
end

% Initialize a copy of the model m:
% We are going to possibly copy things to gpu.
% We are going to transpose and split SV.
% Best to work on a copy.
m = struct();
m.kerparam = model.kerparam;
% Stupid matlab copies on write, so we need to keep sv in two blocks.
% Accumulate on svtr2 and when svtr2 > sv_block_size merge it with svtr1
m.sv_block_size = 1e8/nd;
m.svtr1 = model.SV';
m.svtr2 = zeros(0, nd, 'like', model.SV);
m.beta = model.beta;
m.beta2 = model.beta2;

if opts.gpu
  assert(gpuDeviceCount()>0, 'No GPU detected.');
  fprintf('Initializing gpu. m=%.2e .. ', gmem);
  gdev = gpuDevice;
  reset(gdev);
  m.beta = gpuArray(m.beta);
  m.beta2 = gpuArray(m.beta2);
  m.svtr1 = gpuArray(m.svtr1);
  m.svtr2 = gpuArray(m.svtr2);
  wait(gdev);
  fprintf('%.2e done.\n', gmem);
end

% Default batchsize for training = 1000
if ~opts.update
  m.batchsize = real_batchsize();
elseif isfield(model,'batchsize')
  m.batchsize = model.batchsize;
else
  m.batchsize=default_batchsize;
end
fprintf('Using X batchsize=%d\n', m.batchsize);

if isfield(model,'step')
  m.step = model.step;
else
  m.step = default_step;
end

m.nerr = 0;
if nargout_save >= 2
  scores = [];
end
if nargout_save >= 3
  aer = 0;
end

end % perceptron_init


%%%%%%%%%%%%%%%%%
function g=gmem()
if gpuDeviceCount > 0
  gdev = gpuDevice;
  g = gdev.FreeMemory/8;
else
  g = 0;
end
end % gmem


%%%%%%%%%%%%%%%%%%%%%%%
function new_sv_block()

fprintf('g:%.2g merging sv blocks %dx%d %dx%d\n', gmem, size(m.svtr1), size(m.svtr2));
m.svtr1 = [ gather(m.svtr1); gather(m.svtr2) ];
m.svtr2 = zeros(0, nd, 'like', m.svtr1);
m.beta = gather(m.beta);
m.beta2 = gather(m.beta2);
if opts.gpu
  gdev = gpuDevice;
  reset(gdev);
  m.svtr1 = gpuArray(m.svtr1);
  m.svtr2 = gpuArray(m.svtr2);
  m.beta = gpuArray(m.beta);
  m.beta2 = gpuArray(m.beta2);
end
fprintf('g:%.2g done with merge\n', gmem);

end % new_sv_block


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_sv_blocks(nu)
assert(nu < m.sv_block_size);
if nu + size(m.svtr2, 1) > m.sv_block_size
  new_sv_block();
end
end % check_sv_blocks


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nk = real_batchsize()
persistent batchsize_warning;
if opts.gpu
  nk = floor(0.9 * gmem / (2*ns+2*nd+5*nc+10));
  if nk == 0
    % make a last ditch effort
    new_sv_block();
    nk = floor(0.9 * gmem / (2*ns+2*nd+5*nc+10));
  end
  if opts.update
    % Listen to user during training, use all available memory during testing
    nk = min(nk, m.batchsize);
    if (isempty(batchsize_warning) && (nk < m.batchsize))
      fprintf('g=%.2g Going to batchsize <= %d due to memory limit.\n', gmem, nk);
      batchsize_warning=true;
    end
  end
  assert(nk >= 1, 'g=%.2g nk=%d sv=%.2g beta=%.2g beta2=%.2g, no space left.\n', gmem, ...
         nk, numel(m.svtr1), numel(m.beta), numel(m.beta2));
else % if opts.gpu
  if isfield(m, 'batchsize')
    nk = m.batchsize;
  else
    nk = default_batchsize;
  end
end % if opts.gpu
end % real_batchsize


end % perceptron


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val_f = compute_scores(m, X, i, j, opts)
    ns1 = size(m.svtr1, 1);
    ns2 = size(m.svtr2, 1);
    ns = ns1 + ns2;
    if ns>0
        hp = m.kerparam;
        if opts.gpu
            xij = gpuArray(X(:,i:j));
        else
            xij = X(:,i:j);
        end
        if ~opts.average
            beta1 = m.beta(:,1:ns1);
            beta2 = m.beta(:,ns1+1:end);
        else
            beta1 = m.beta2(:,1:ns1);
            beta2 = m.beta2(:,ns1+1:end);
        end

        switch m.kerparam.type
          case 'poly'
            if ns2 == 0
                val_f = beta1 * (hp.gamma * (m.svtr1 * xij) + hp.coef0) .^ hp.degree;
            elseif ns1 == 0
                val_f = beta2 * (hp.gamma * (m.svtr2 * xij) + hp.coef0) .^ hp.degree;
            else
                val_f = beta1 * (hp.gamma * (m.svtr1 * xij) + hp.coef0) .^ hp.degree + ...
                        beta2 * (hp.gamma * (m.svtr2 * xij) + hp.coef0) .^ hp.degree;
            end

          case 'rbf'
            x_sq = sum(xij.^2, 1);
            if ns2 == 0
                s1sq = sum(m.svtr1.^2, 2);
                val_f = beta1 * exp(-hp.gamma * bsxfun(@plus, x_sq, bsxfun(@plus, s1sq, -2 * (m.svtr1 * xij))));
                clear s1sq;
            elseif ns1 == 0
                s2sq = sum(m.svtr2.^2, 2);
                val_f = beta2 * exp(-hp.gamma * bsxfun(@plus, x_sq, bsxfun(@plus, s2sq, -2 * (m.svtr2 * xij))));
                clear s2sq;
            else
                s1sq = sum(m.svtr1.^2, 2);
                s2sq = sum(m.svtr2.^2, 2);
                val_f = beta1 * exp(-hp.gamma * bsxfun(@plus, x_sq, bsxfun(@plus, s1sq, -2 * (m.svtr1 * xij)))) + ...
                        beta2 * exp(-hp.gamma * bsxfun(@plus, x_sq, bsxfun(@plus, s2sq, -2 * (m.svtr2 * xij))));
                clear s1sq s2sq;
            end
            clear x_sq;
          otherwise
            error('Unknown kernel');

        end

        val_f = gather(val_f);
        clear xij beta1 beta2;
    else
        val_f = zeros(size(m.beta, 1), j-i+1, 'like', X);
    end % if ns>0
end % compute_scores


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model = perceptron_compactify(model)

% [C, ia, ic] = unique(A,'rows')
% Find the unique rows C(u,d) of A(n,d) and the index vectors ia(u,1) and ic(n,1), such that ...
% C = A(ia,:) and A = C(ic,:).

fprintf('Finding unique SV in %d...\n', size(model.SV, 2));
[~, ia, ic] = unique(model.SV', 'rows');

fprintf('Saving %d unique SV.\n', numel(ia));
b2 = isfield(model, 'beta2');
nc = size(model.beta, 1);
newbeta  = zeros(nc, numel(ia), 'like', model.beta);
if b2 newbeta2 = zeros(nc, numel(ia), 'like', model.beta2); end
assert(numel(ic) == size(model.beta, 2));

for oldi=1:numel(ic)
  newi = ic(oldi);
  newbeta(:,newi) = newbeta(:,newi) + model.beta(:,oldi);
  if b2 newbeta2(:,newi) = newbeta2(:,newi) + model.beta2(:,oldi); end
end

model.SV = model.SV(:,ia);
model.beta = newbeta;
if b2 model.beta2 = newbeta2; end

end % perceptron_compactify
