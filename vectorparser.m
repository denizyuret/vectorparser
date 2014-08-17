% Usage: vectorparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = vectorparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = vectorparser(model, corpus, 'update', 0, 'predict', 0); %% dump features
% [model,dump] = vectorparser(model, corpus, 'update', 0)); %% testing
% stats = eval_conll(corpus, dump);

function [model, dump] = vectorparser(model, corpus, varargin)

msg('Initializing...');
m = vectorparser_init(model, corpus, varargin, nargout);
msg('Processing sentences...');
t0 = tic;

for snum=1:numel(corpus)
  s = corpus{snum};
  h = s.head;
  n = numel(h);
  p = feval(m.parser, n);

  while 1                               % parse one sentence
    valid = p.valid_moves();
    if ~any(valid) break; end

    if m.compute_costs
      cost = p.oracle_cost(h); 		% 1019us
      [mincost, mincostmove] = min(cost);
    end

    if m.compute_features
      f = features(p, s, m.feats);  % 1153us
      ftr = f';                         % f is a row vector, ftr column vector
    end

    if m.compute_scores
      score = compute_scores(m, ftr);
      [maxscore, maxscoremove] = max(score); % 925us
    end

    if m.update
      m.bavg1 = m.bavg1 + m.bfin1;
      m.bavg2 = m.bavg2 + m.bfin2;
      if cost(maxscoremove) > mincost
        m.svtr2(end+1,:) = f;
        newbeta = zeros(1, p.NMOVE);
        newbeta(mincostmove) = 1;
        newbeta(maxscoremove) = -1;
        m.bfin2(end+1,:) = newbeta;
        m.bavg2(end+1,:) = newbeta;
      end
    end % if m.update

    if ~m.predict
      execmove = mincostmove;
    elseif valid(maxscoremove)
      execmove = maxscoremove;
    else
      zscore = score;
      zscore(~valid) = -inf;
      [~,execmove] = max(zscore);
    end

    p.transition(execmove);

    if m.dump 
      if m.compute_scores
        m = update_dump(m, ftr, mincostmove, cost, execmove, score);
      else
        m = update_dump(m, ftr, mincostmove, cost, execmove);
      end
    end

  end % while 1

  if m.dump && m.predict
    m.pred{end+1} = p.head;
  end

  dot(snum, numel(corpus), t0);
end % for s1=corpus


if m.update
  model.x = m.x;
  model.SV = [gather(m.svtr1); gather(m.svtr2)]';
  model.beta = [gather(m.bfin1); gather(m.bfin2)]';
  model.beta2 = [gather(m.bavg1); gather(m.bavg2)]';
  model = compactify(model);
end

if m.dump 
  if m.compute_features
    [~,m.fidx] = features(p, s, m.feats);
  end
  dump = m;
end

end % vectorparser


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score = compute_scores(m, ftr)

% Same matrix operation has different costs on gpu:
% score = gather(sum(bsxfun(@times, btr, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1)); % 7531us
% Matrix multiplication is less efficient than array mult:
% score = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
% Even the transpose makes a difference:
% score = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree),1)); % 11364us

hp = m.kerparam;
if m.average
  b1tr = m.bavg1;
  b2tr = m.bavg2;
else
  b1tr = m.bfin1;
  b2tr = m.bfin2;
end
score1 = [];
if isfield(m, 'cache')
  score1 = m.cache.get(ftr);
end
if isempty(score1)
  % score1 = gather(b1 * (hp.gamma * (m.svtr1 * ftr) + hp.coef0).^hp.degree);
  score1 = gather(sum(bsxfun(@times, b1tr, (hp.gamma * (m.svtr1 * ftr) + hp.coef0).^hp.degree),1))';
end
% score2 = gather(b2 * (hp.gamma * (m.svtr2 * ftr) + hp.coef0).^hp.degree);
score2 = gather(sum(bsxfun(@times, b2tr, (hp.gamma * (m.svtr2 * ftr) + hp.coef0).^hp.degree),1))';
score = score1 + score2;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = vectorparser_init(model, corpus, varargin_save, nargout_save)

m = model;      % m is a copy of model with extra information

for vi = 1:2:numel(varargin_save)
  v = varargin_save{vi};
  v1 = varargin_save{vi+1};
  switch v
   case 'predict' 
    m.predict = v1;
   case 'update'  
    m.update  = v1;
   case 'average' 
    m.average = v1;
   case 'gpu'     
    m.gpu     = v1;
   otherwise 
    error('Usage: [model, dump] = vectorparser(model, corpus, m)');
  end
end

if ~isfield(m, 'predict')
  m.predict = true;  % predict: use the model for prediction (default), otherwise follow gold moves
end
 
if ~isfield(m, 'update')
  m.update = true;   % update: train the model (default), otherwise model is not updated
end

m.dump = (nargout_save >= 2);
m.compute_costs = m.update || ~m.predict;
m.compute_features = m.update || m.dump || m.predict;
m.compute_scores  = m.update || m.predict;

assert(isfield(m,'parser'), 'Please specify model.parser.');
tmp_s = corpus{1};
tmp_p = feval(m.parser, numel(tmp_s.head));
nc = tmp_p.NMOVE;

if m.compute_features
  assert(isfield(m,'feats'), 'Please specify model.feats.');
  assert(size(m.feats, 2) == 3, 'The feats matrix needs 3 columns.');
  tmp_f = features(tmp_p, tmp_s, m.feats);
  nd = numel(tmp_f);
end

if m.compute_scores
  assert(isfield(m,'kerparam') && ...
         strcmp(m.kerparam.type,'poly'), ...
         'Please specify poly kernel in model.kerparam.\n');
  hp = m.kerparam;

  if ~isfield(m,'average')
    m.average = (isfield(m,'beta2') && ~isempty(m.beta2) && ~m.update);
  elseif m.average
    assert(isfield(m,'beta2') && ~isempty(m.beta2),...
           'Please set model.beta2 for averaged model.');
  end

  % We are using 1 and 2 for the old and the new SV blocks
  % For the final vs average model we'll use beta and zeta

  if ~isfield(m,'SV') || isempty(m.SV)
    if m.update
      m.svtr1 = zeros(0, nd);
      m.svtr2 = zeros(0, nd);
      m.bfin1 = zeros(0, nc);
      m.bfin2 = zeros(0, nc);
      m.bavg1 = zeros(0, nc);
      m.bavg2 = zeros(0, nc);
    else
      error('Please specify model.SV');
    end
  else
    assert(size(m.SV, 1) == nd);
    assert(size(m.beta, 1) == nc);
    assert(size(m.SV, 2) == size(m.beta, 2));
    assert(all(size(m.beta) == size(m.beta2)));
    m.svtr1 = m.SV';
    m.svtr2 = zeros(0, nd);
    m.bfin1 = m.beta';
    m.bfin2 = zeros(0, nc);
    m.bavg1 = m.beta2';
    m.bavg2 = zeros(0, nc);
  end

  if m.update && isfield(m, 'x')
    msg('Computing cache scores.');tmp=tic;
    [~,scores] = perceptron(m.x, [], m, 'update', 0, 'average', m.average);
    msg('Initializing kernel cache.');toc(tmp);tmp=tic;
    m.cache = kernelcache(m.x, scores);
    msg('done');toc(tmp);
  end

  if ~isfield(m, 'gpu')
    m.gpu = gpuDeviceCount(); % Use the gpu if there is one
  end

  if m.gpu
    assert(gpuDeviceCount()>0, 'No GPU detected.');
    fprintf('Loading model on GPU.\n');
    gpuDevice(1);
    m.svtr1 = gpuArray(m.svtr1);
    m.bfin1 = gpuArray(m.bfin1);
    m.bavg1 = gpuArray(m.bavg1);
    m.svtr2 = gpuArray(m.svtr2);
    m.bfin2 = gpuArray(m.bfin2);
    m.bavg2 = gpuArray(m.bavg2);
  end

end % if m.compute_scores

if m.predict
  fprintf('Using predicted moves.\n');
else
  fprintf('Using gold moves.\n');
end % if m.predict

if m.dump
  fprintf('Dumping results.\n');
  if m.compute_features
    m.x = [];
  end
  if m.compute_costs
    m.y = [];
    m.cost = [];
  end
  if m.compute_scores
    m.z = [];
    m.score = [];
  end
  if m.predict
    m.pred = {};
  end
end % if m.dump

end % vectorparser_init


%%%%%%%%%%%%%%%%%%%%%%
function m = update_dump(m, ftr, mincostmove, cost, execmove, score)
if m.compute_features
  m.x(:,end+1) = ftr;
end
if m.compute_costs
  m.y(end+1) = mincostmove;
  m.cost(:,end+1) = cost;
end
if m.compute_scores
  m.z(end+1) = execmove;
  m.score(:,end+1) = score;
end
end % update_dump


%%%%%%%%%%%%%%%%%%%%%%%%%%
function dot(cur, tot, t0)
t1 = toc(t0);
if cur == tot
  fprintf('. %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
elseif mod(cur,10) == 0
  fprintf('.');
  if mod(cur, 100) == 0
    fprintf(' %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
  end
end
end % dot

