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

vectorparser_init(varargin, nargout);
fprintf('Processing sentences...\n');
t0 = tic;

for snum=1:numel(corpus)
  s = corpus{snum};
  h = s.head;
  n = numel(h);
  p = feval(model.parser, n);

  while 1                               % parse one sentence
    valid = p.valid_moves();
    if ~any(valid) break; end

    if opts.compute_costs
      cost = p.oracle_cost(h); 		% 1019us
      [mincost, mincostmove] = min(cost);
    end

    if opts.compute_features
      f = features(p, s, model.feats);  % 1153us
      ftr = f';                         % f is a row vector, ftr column vector
    end

    if opts.compute_scores
      if isempty(svtr)
        score = zeros(1, p.NMOVE);
      elseif opts.average
        score = gather(sum(bsxfun(@times, betatr2, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1));
      else
        score = gather(sum(bsxfun(@times, betatr, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1));
      end
      [maxscore, maxscoremove] = max(score); % 925us

      % Same matrix operation has different costs on gpu:
      % score = gather(sum(bsxfun(@times, betatr, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1)); % 7531us
      % Matrix multiplication is less efficient than array mult:
      % score = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
      % Even the transpose makes a difference:
      % score = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree),1)); % 11364us

    end

    if opts.update
      betatr2 = betatr2 + betatr;
      if cost(maxscoremove) > mincost
        svtr(end+1,:) = f;
        newbeta = zeros(1, p.NMOVE);
        newbeta(mincostmove) = 1;
        newbeta(maxscoremove) = -1;
        betatr(end+1,:) = newbeta;
        betatr2(end+1,:) = newbeta;
      end
    end % if opts.update

    if ~opts.predict
      execmove = mincostmove;
    elseif valid(maxscoremove)
      execmove = maxscoremove;
    else
      zscore = score;
      zscore(~valid) = -inf;
      [~,execmove] = max(zscore);
    end

    p.transition(execmove);

    if opts.dump update_dump(); end

  end % while 1

  if opts.dump && opts.predict
    dump.pred{end+1} = p.head;
  end

  dot(snum, numel(corpus), t0);
end % for s1=corpus


if opts.update
  model.SV = gather(svtr)';
  model.beta = gather(betatr)';
  model.beta2 = gather(betatr2)';
  model = compactify(model);
end

if opts.dump && opts.compute_features
  dump.feats = model.feats;
  [~,dump.fidx] = features(p, s, model.feats);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vectorparser_init(varargin_save, nargout_save)

opts = struct();      % opts is a struct of options.

for vi = 1:2:numel(varargin_save)
  v = varargin_save{vi};
  v1 = varargin_save{vi+1};
  switch v
   case 'predict' 
    opts.predict = v1;
   case 'update'  
    opts.update  = v1;
   case 'average' 
    opts.average = v1;
   case 'gpu'     
    opts.gpu     = v1;
   otherwise 
    error('Usage: [model, dump] = vectorparser(model, corpus, opts)');
  end
end

if ~isfield(opts, 'predict')
  opts.predict = true;  % predict: use the model for prediction (default), otherwise follow gold moves
end
 
if ~isfield(opts, 'update')
  opts.update = true;   % update: train the model (default), otherwise model is not updated
end

opts.dump = (nargout_save >= 2);
opts.compute_costs = opts.update || opts.dump || ~opts.predict;
opts.compute_features = opts.update || opts.dump || opts.predict;
opts.compute_scores  = opts.update || opts.predict;

assert(isfield(model,'parser'), 'Please specify model.parser.');
tmp_s = corpus{1};
tmp_p = feval(model.parser, numel(tmp_s.head));
nc = tmp_p.NMOVE;

if opts.compute_features
  assert(isfield(model,'feats'), 'Please specify model.feats.');
  assert(size(model.feats, 2) == 3, 'The feats matrix needs 3 columns.');
  tmp_f = features(tmp_p, tmp_s, model.feats);
  nd = numel(tmp_f);
end

if opts.compute_scores
  assert(isfield(model,'kerparam') && ...
         strcmp(model.kerparam.type,'poly'), ...
         'Please specify poly kernel in model.kerparam.\n');
  hp = model.kerparam;

  if ~isfield(opts,'average')
    opts.average = (isfield(model,'beta2') && ~isempty(model.beta2) && ~opts.update);
  elseif opts.average
    assert(isfield(model,'beta2') && ~isempty(model.beta2),...
           'Please set model.beta2 for averaged model.');
  end

  if ~isfield(model,'SV') || isempty(model.SV)
    if opts.update
      svtr = zeros(0, nd);
      betatr = zeros(0, nc);
      betatr2 = zeros(0, nc);
    else
      error('Please specify model.SV');
    end
  else
    assert(size(model.SV, 1) == nd);
    assert(size(model.beta, 1) == nc);
    assert(size(model.SV, 2) == size(model.beta, 2));
    assert(all(size(model.beta) == size(model.beta2)));
    svtr = model.SV';
    betatr = model.beta';
    betatr2 = model.beta2';
  end

  if ~isfield(opts, 'gpu')
    opts.gpu = gpuDeviceCount(); % Use the gpu if there is one
  end

  if opts.gpu
    assert(gpuDeviceCount()>0, 'No GPU detected.');
    fprintf('Loading model on GPU.\n');
    gpuDevice(1);
    svtr = gpuArray(svtr);
    betatr = gpuArray(betatr);
    betatr2 = gpuArray(betatr2);
  end

end % if opts.compute_scores

if opts.predict
  fprintf('Using predicted moves.\n');
else
  fprintf('Using gold moves.\n');
end % if opts.predict

if opts.dump
  fprintf('Dumping results.\n');
  if opts.compute_features
    dump.x = [];
  end
  if opts.compute_costs
    dump.y = [];
    dump.cost = [];
  end
  if opts.compute_scores
    dump.z = [];
    dump.score = [];
  end
  if opts.predict
    dump.pred = {};
  end
end % if opts.dump

end % vectorparser_init


%%%%%%%%%%%%%%%%%%%%%%
function update_dump()
if opts.compute_features
  dump.x(:,end+1) = ftr;
end
if opts.compute_costs
  dump.y(end+1) = mincostmove;
  dump.cost(:,end+1) = cost;
end
if opts.compute_scores
  dump.z(end+1) = execmove;
  dump.score(:,end+1) = score;
end
end % update_dump

%%%%%%%%%%%%%%%%%%%%%%
function dot(cur, tot, t0)
t1 = toc(t0);
if cur == tot
  fprintf(' %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
elseif mod(cur,10) == 0
  fprintf('.');
  if mod(cur, 100) == 0
    fprintf(' %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
  end
end
end % dot

end % vectorparser
