function model = perceptron(X,Y,model)

% perceptron: written by Deniz Yuret, August 2, 2014.
% Multi-class, mini-batch, cost based, gpu enabled perceptron.
% Based on the DOGMA library by Francesco Orabona.
%
% X(nd,nx) has an instance in each column.
%
% Y(nc,nx) has the cost of each class for each instance.
% - all mincost classes are considered correct.
% - all cost=inf classes are considered invalid regardless of their score.
%
% If Y is a vector, it is taken to be the vector of correct answers
% and automatically converted to a 0-1 cost matrix.
%
% model can be a blank model or the result of a previous epoch, in
% which case it will have non-empty SV(nd,ns), beta(nc,ns) and
% beta2(nc,ns).
% 
% model specifies the polynomial kernel parameters:
% Default gamma=1, coef0=1, degree=3, type='poly'.
% hp = model.kerparam;
% scores = model.beta * (hp.gamma * full(model.SV' * X) + hp.coef0) .^ hp.degree;
%
% model.beta2 are the averaged (rather summed) parameters.
%
% model.batchsize gives the mini-batch size (default=1000).
%
% model.step determines how often results are printed (default=10000).
%


[nd,nx,nc,ns,gpu,gdev] = perceptron_init();
fprintf('inst\tnsv\tbatch\ttime\tmem\n');


% Stupid matlab copies on write, so we need to keep sv in two blocks
sv_block_size = 1e8/nd;
svtr = model.SV';
svtr2 = zeros(0, nd);
if gpu gpu_load_model(); end

i = 0; j = 0; 
j_step = model.step;

while j < nx                          % 26986us/iter for batchsize=500

  i = j + 1;
  nk = real_batchsize();
  j = min(nx, i + nk - 1);            % will process minibatch X(:,i:j)
  nk = j - i + 1;                     % in case j==nx

  score = compute_scores();           % score(nc,nk): scores for X(:,i:j)
  costij = Y(:,i:j);                  % costij(nc,nk): costs for X(:,i:j)
% score(isinf(costij)) = -inf;        % do not punish for impossible answers, turns out bad idea!
  [maxscore, maxscore_i] = max(score); % compare the cost of maxscore answers
  [mincost, mincost_i] = min(costij); % to the mincost answers
  mycost = costij(sub2ind(size(costij), maxscore_i, 1:nk)); % cost of maxscore answers
  updates = find(mycost > mincost);

  if ~isempty(updates)                % 33587us
    nu = numel(updates);
    ns = ns + nu;

    check_sv_blocks(nu);
    updates_i = updates+i-1;          % updates uses (1,nk) indexing, updates_i uses (i,j) indexing
    svtr2 = [ svtr2; X(:,updates_i)' ];

    newbeta = zeros(nc, nu);
    newbeta(sub2ind(size(newbeta), mincost_i(updates), 1:nu)) = 1;
    newbeta(sub2ind(size(newbeta), maxscore_i(updates), 1:nu)) = -1;

    model.beta2 = model.beta2 + model.beta;
    model.beta2 = [model.beta2 newbeta];
    model.beta = [model.beta newbeta];    % 972us

  end % if ~isempty(updates)

  if j >= j_step
    fprintf('%d\t%d\t%d\t%.2f\t%.2e\n', j, ns, nk, toc, gmem);
    j_step = j_step + model.step;
  end

end % while j < nx

fprintf('%d\t%d\t%d\t%.2f\t%.2e\n', j, ns, nk, toc, gmem);
model.beta = gather(model.beta);
model.beta2 = gather(model.beta2);
model.SV = [ gather(svtr)', gather(svtr2)' ];
clear svtr svtr2
model = compactify(model);



%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nd,nx,nc,ns,gpu,gdev] = perceptron_init()

% Get the kernel parameters
if ~isfield(model,'kerparam')
  fprintf('Using default kernel: gamma=1, coef0=1, degree=3, type=poly.\n');
  model.kerparam = struct('type','poly','degree',3,'gamma',1,'coef0',1);
end
assert(strcmp(model.kerparam.type,'poly'), 'Only poly kernel models supported.\n');

% Get the size of the problem
nd = size(X, 1);
nx = size(X, 2);
assert(nx == size(Y, 2));
nc = size(Y, 1);
if nc == 1
  fprintf('Cost matrix is 1D, assuming these are correct answers.\n');
  assert(all(Y>=1));
  nc = max(Y);
  cost = ones(nc, nx);
  cost(sub2ind(size(cost), Y, 1:nx)) = 0;
  Y = cost;
end
if ~isfield(model,'SV') || isempty(model.SV)
  fprintf('Initializing empty model.\n');
  ns = 0;
  model.SV = zeros(nd,0);
  model.beta = zeros(nc,0);
  model.beta2 = zeros(nc,0);
else
  ns = size(model.SV, 2);
end

assert(size(model.SV, 1) == nd);
assert(size(model.SV, 2) == ns);
assert(size(model.beta, 1) == nc);
assert(size(model.beta, 2) == ns);
assert(size(model.beta2, 1) == nc);
assert(size(model.beta2, 2) == ns);

fprintf('nd=%d nx=%d nc=%d ns=%d\n', nd, nx, nc, ns);

% Default batchsize = 1000
if ~isfield(model,'batchsize')
  model.batchsize=1000;
end
model.batchsize_warning = 0;
fprintf('Using X batchsize=%d\n', model.batchsize);

% See if we have a gpu
gpu = gpuDeviceCount(); 
if gpu
  gdev = gpuDevice;
else
  gdev = [];
end

if ~isfield(model,'step')
  model.step = 10000;
end

end % perceptron_init


%%%%%%%%%%%%%%%%%%%%%%%
function new_sv_block()

fprintf('g:%.2g merging sv blocks %dx%d %dx%d\n', gmem, size(svtr), size(svtr2));
svtr = [ gather(svtr); gather(svtr2) ];
svtr2 = zeros(0, nd);
model.beta = gather(model.beta);
model.beta2 = gather(model.beta2);
if gpu
  reset(gdev);
  svtr = gpuArray(svtr);
  svtr2 = gpuArray(svtr2);
  model.beta = gpuArray(model.beta);
  model.beta2 = gpuArray(model.beta2);
end
fprintf('g:%.2g done with merge\n', gmem);

end % new_sv_block


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nk = real_batchsize()
persistent batchsize_warning;
if gpu
  nk = floor(0.9 * gmem / (2*ns+2*nd+5*nc+10));
  nk = min(nk, model.batchsize);
  if nk == 0
    % make a last ditch effort
    new_sv_block();
    nk = floor(0.9 * gmem / (2*ns+2*nd+5*nc+10));
  end
  assert(nk >= 1, 'g=%.2g nk=%d sv=%.2g beta=%.2g beta2=%.2g, no space left.\n', gmem, ...
         nk, numel(svtr), numel(model.beta), numel(model.beta2));
  if (nk < model.batchsize && ~batchsize_warning)
    fprintf('g=%.2g Going to batchsize <= %d due to memory limit.\n', gmem, nk);
    batchsize_warning=true;
  end
else
  nk = model.batchsize;
end % if gpu
end % real_batchsize


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function check_sv_blocks(nu)
assert(nu < sv_block_size);
if nu + size(svtr2, 1) > sv_block_size
  new_sv_block();
end
end % check_sv_blocks


%%%%%%%%%%%%%%%%%
function g=gmem()
if gpu
  g = gdev.FreeMemory/8;
else
  g = 0;
end
end % gmem


%%%%%%%%%%%%%%%%%%%%%%%%%
function gpu_load_model()
fprintf('Initializing gpu. m=%.2e .. ', gmem);
reset(gdev);
model.beta = gpuArray(model.beta);
model.beta2 = gpuArray(model.beta2);
svtr = gpuArray(svtr);
svtr2 = gpuArray(svtr2);
wait(gdev);
fprintf('%.2e done.\n', gmem);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val_f = compute_scores()
assert(nk == j - i + 1, 'nk=%d i=%d j=%d', nk, i, j);

if ns>0                             % 484802us
  if gpu
    xij = gpuArray(X(:,i:j));             % 27027us for batchsize=1250
  else
    xij = X(:,i:j);
  end
  ns1 = size(svtr, 1);
  assert(size(svtr2, 1) == ns - ns1);
  assert(~issparse(svtr2) && ~issparse(xij));
  hp = model.kerparam;

  val_f = model.beta(:,ns1+1:ns) * (hp.gamma * (svtr2 * xij) + hp.coef0) .^ hp.degree; % 166061us
  if ns1 > 0
    val_f = val_f + model.beta(:,1:ns1) * (hp.gamma * (svtr * xij) + hp.coef0) .^ hp.degree; % 166061us
  end
  val_f = gather(val_f);
  clear xij;
else
  val_f = zeros(nc, nk);
end % if ns>0
end % compute_beta_k


end % perceptron
