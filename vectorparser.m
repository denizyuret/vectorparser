% Usage: vectorparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = vectorparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = vectorparser(model, corpus, 'update', 0)); %% testing
% [model,dump] = vectorparser(model, corpus, 'update', 0, 'predict', 0); %% dump features

function [model, dump] = vectorparser(model, corpus, varargin)

vectorparser_init(nargout);
fprintf('Processing sentences...\n');

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
      [mincost, bestmove] = min(cost);
    end

    if opts.compute_features
      f = features(p, s, model.feats);  % 1153us
      ftr = f';                         % f is a row vector, ftr column vector
    end

    if opts.compute_scores
      % Same matrix operation has different costs on gpu:
      % score = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
      % score = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree))); % 11364us
      if isempty(svtr)
        score = zeros(1, model.n_cla);
      elseif opts.average
        score = model.b2 + gather(sum(bsxfun(@times, betatr2, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1)); % 7531us
      else
        score = model.b + gather(sum(bsxfun(@times, betatr, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1)); % 7531us
      end
      % score(cost==inf) = -inf;        % very very bad idea!
      [maxscore, maxmove] = max(score); % 925us
    end

    if opts.update
      if cost(maxmove) > mincost
        if isempty(svtr)
          svtr = f;
        else
          svtr(end+1,:) = f;
        end
        betatr(end+1,:) = zeros(1, model.n_cla);
        betatr2(end+1,:) = zeros(1, model.n_cla);
        betatr(end, bestmove) = 1;
        betatr(end, maxmove) = -1;
      end
      betatr2 = betatr2 + betatr;
    end % if opts.update

    if opts.predict
      vscore = score;
      vscore(~valid) = -inf;
      [~,vmove] = max(vscore);
      p.transition(vmove);                 % 1019us
    else
      p.transition(bestmove);
    end

    if opts.dump update_dump(); end

  end % while 1

  if opts.dump && opts.predict
    dump.pred{end+1} = p.head;
  end

  fprintf('.');
end % for s1=corpus
fprintf('\n');

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


function vectorparser_init(nargout_save)

opts = struct();      % opts is an (optional) struct of options.

for vi = 1:2:numel(varargin)
  v = varargin{vi};
  v1 = varargin{vi+1};
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

if opts.update % Initialize model by defaults if necessary
  fprintf('Updating model.\n');
  if ~isfield(model,'parser')
    fprintf('Using default parser archybrid.\n');
    model.parser = @archybrid;
  end
  if ~isfield(model,'n_cla')
    p = feval(model.parser, 1);
    model.n_cla=p.NMOVE;
  end
  if ~isfield(model,'beta')
    model.beta=zeros(model.n_cla, 0);
    model.beta2=zeros(model.n_cla, 0);
  end
  if ~isfield(model,'feats')
    fprintf('Using default feature set fv808.\n');
    model.feats = ...
        [ % use fv808 as the default feature set
          %n0 s0 s1 n1 n0l1 s0r1 s1r1l s0l1 s0r1l s2
          0 -1 -2  1  0   -1   -2    -1   -1    -3;
          0  0  0  0 -1    1    1    -1    1     0;
          0  0  0  0  0    0   -2     0   -2     0;
        ]';
  end
end

opts.dump = (nargout_save >= 2);
opts.compute_costs = opts.update || opts.dump || ~opts.predict;
opts.compute_features = opts.update || opts.dump || opts.predict;
opts.compute_scores  = opts.update || opts.predict;

assert(isfield(model,'parser'), 'Please specify model.parser.');

if opts.compute_features
  assert(isfield(model,'feats'), 'Please specify model.feats.');
  assert(size(model.feats, 2) == 3, 'The feats matrix needs 3 columns.');
end

if opts.compute_scores
  assert(isfield(model,'n_cla'), 'Please specify model.n_cla.');
  assert(isfield(model,'beta'), 'Please specify model.beta.');
  assert(isfield(model,'beta2'), 'Please specify model.beta2.');
  assert(isfield(model,'b'), 'Please specify model.b.');
  assert(isfield(model,'b2'), 'Please specify model.b2.');
  assert(isfield(model,'kerparam') && ...
         strcmp(model.kerparam.type,'poly'), ...
         'Please specify poly kernel in model.kerparam.\n');
  hp = model.kerparam;

  if ~isfield(opts,'average')
    opts.average = (isfield(model,'beta2') && ~isempty(model.beta2) && ~opts.update);
  elseif opts.average
    assert(isfield(model,'beta2') && ~isempty(model.beta2),...
           'Please set model.beta2 for averaged model.');
    assert(~opts.update, 'Cannot use averaged model during update.');
  end

  if ~isempty(model.SV)
    assert(~isempty(model.beta) && all(size(model.beta) == size(model.beta2)));
    svtr = model.SV';
    betatr = model.beta';
    betatr2 = model.beta2';
  else
    svtr = [];
    betatr = zeros(0, model.n_cla);
    betatr2 = zeros(0, model.n_cla);
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


function update_dump()
if opts.compute_features
  dump.x(:,end+1) = ftr;
end
if opts.compute_costs
  dump.y(end+1) = bestmove;
  dump.cost(:,end+1) = cost;
end
if opts.compute_scores
  dump.z(end+1) = maxmove;
  dump.score(:,end+1) = score;
end
end % update_dump

end % vectorparser
