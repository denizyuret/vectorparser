% Usage:
%
% model = vectorparser(model, corpus);  %% training
% [model, dump] = vectorparser(model, corpus, 'update', 0)); %% testing
% [model, dump] = vectorparser(model, corpus, 'update', 0, 'predict', 0); %% dump features

function [model, dump] = vectorparser(model, corpus, varargin)

nargout_save = nargout;
vectorparser_init();
fprintf('Processing sentences...\n');

for snum=1:numel(corpus)
  s = corpus{snum};
  h = s.head;
  n = numel(h);
  p = feval(model.parser, n);

  while 1                               % parse one sentence
    valid = p.valid_moves();
    if ~any(valid) break; end

    if p.wptr <= p.nword n0=s.form{p.wptr}; else n0='NONE'; end
    if p.sptr >= 1 s0=s.form{p.stack(p.sptr)}; else s0='NONE'; end
    % fprintf('%s(%d) | %s(%d)', s0, p.sptr, n0, p.nword-p.wptr);

    if opts.compute_bestmove
      cost = p.oracle_cost(h); 		% 1019us
      [mincost, bestmove] = min(cost);
    end

    if opts.compute_features
      f = features(p, s, model.feats);  % 1153us
      ftr = f';                         % f is a row vector, ftr column vector
    end

    if opts.compute_maxmove
      % Same matrix operation has different costs on gpu:
      % scores = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
      % scores = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree))); % 11364us
      if isempty(svtr)
        scores = zeros(1, model.n_cla);
      elseif opts.average
        scores = model.b2 + gather(sum(bsxfun(@times, betatr2, (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree))); % 7531us
      else
        scores = model.b + gather(sum(bsxfun(@times, betatr, (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree))); % 7531us
      end
      scores(cost==inf) = -inf;         % 928us
      [~,maxmove] = max(scores);        % 925us
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

    if opts.dump
      dump.x(:,end+1) = ftr;
      dump.y(end+1) = bestmove;
      if opts.compute_maxmove
        dump.z(end+1) = maxmove;
      end
    end

    if opts.predict
      p.transition(maxmove);                 % 1019us
      assert(valid(maxmove));
      % fprintf(' %d\n', maxmove);
    else
      p.transition(bestmove);
      assert(valid(bestmove), '%s %s', num2str(valid), num2str(cost));
      % fprintf(' %d\n', bestmove);
    end

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


function vectorparser_init()

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

opts.dump = (nargout_save >= 2);
if opts.dump
  fprintf('Dump mode.\n');
  dump.x = [];
  dump.y = [];
  dump.z = [];
  dump.pred = {};
end

if opts.update % Initialize model by defaults if necessary
  if ~isfield(model,'parser')
    fprintf('Using default parser archybrid.\n');
    model.parser = @archybrid;
  end
  if ~isfield(model,'n_cla')
    p = feval(model.parser, 0);
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
        ];
  end
end

opts.compute_bestmove = opts.update || opts.dump || ~opts.predict;
opts.compute_features = opts.update || opts.dump || opts.predict;
opts.compute_maxmove  = opts.update || opts.predict;

assert(isfield(model,'parser'), 'Please specify model.parser.');

if opts.compute_features
  assert(isfield(model,'feats'), 'Please specify model.feats.');
end

if opts.compute_maxmove
  assert(isfield(model,'n_cla'), 'Please specify model.n_cla.');
  assert(isfield(model,'beta'), 'Please specify model.beta.');
  assert(isfield(model,'beta2'), 'Please specify model.beta2.');
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
    assert(~isempty(model.beta) && size(model.beta) == size(model.beta2));
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
    fprintf('Loading model on GPU.\n');
    gpuDevice(1);
    svtr = gpuArray(svtr);
    betatr = gpuArray(betatr);
    betatr2 = gpuArray(betatr2);
  end

end % if compute_maxmove

end % vectorparser_init

end % vectorparser
