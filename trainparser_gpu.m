function [r, model] = trainparser_gpu(model, corpus, feats)

% Use second argout for training, single for testing
if (nargout > 1) 
  fprintf('Training mode.\n');
  training=1; 
else
  fprintf('Testing mode.\n');
  training=0;
end

% Make sure we have a poly kernel model
assert(isfield(model,'ker') && ~isempty(model.ker) && isfield(model,'kerparam'),...
       'Only kernel models supported.\n');
hp = model.kerparam;
assert(strcmp(hp.type,'poly'), 'Only poly kernel models supported.\n');

% Initialize statistics
r = struct('ntot',0,'nerr',0,'npct',0,... % token errors
           'xtot',0,'xerr',0,'xpct',0,... % transition errors
           'wtot',0,'werr',0,'wpct',0);   % non-punct errors
%ti=0; r.t = zeros(1,20);

% Initialize model if necessary
if training
  if isfield(model,'n_cla')==0
    model.n_cla=archybrid.NMOVE;
  end
  if isfield(model,'beta')==0
    model.beta=[];
    model.beta2=[];
  end
end

tic; fprintf('Loading model on GPU.\n');
gpuDevice(1);
if ~isempty(model.SV)
  assert(~isempty(model.beta));
  svtr = gpuArray(model.SV');
  betatr = gpuArray(model.beta');
  betatr2 = gpuArray(model.beta2');
else
  svtr = [];
  betatr = [];
  betatr2 = [];
end
average = (isfield(model,'beta2') && ~isempty(model.beta2) && ~training);

toc;tic;fprintf('Processing sentences...\n');
for s1=corpus
  s = s1{1};
  h = s.head;
  n = numel(h);
  p = archybrid(n);
  while ((p.sptr > 1) || (p.wptr <= p.nword))
    c = p.oracle_cost(h);               % 1019us
    assert(any(isfinite(c)));           % 915us
    f = features(p, s, feats);          % 1153us
    ftr = f';

    % Same matrix operation has different costs on gpu:
    % ti=4;wait(gpuDevice);r.t(ti)=r.t(ti)+toc;
    % [~,scores] = model_predict(f(featidx), model)
    % scores = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
    % scores = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree))); % 11364us

    if isempty(svtr)
      scores = zeros(1, model.n_cla);
    elseif average
      scores = model.b2 + gather(sum(bsxfun(@times, betatr2, (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree))); % 7531us
    else
      scores = model.b + gather(sum(bsxfun(@times, betatr, (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree))); % 7531us
    end
    % ti=5;wait(gpuDevice);r.t(ti)=r.t(ti)+toc;

    scores(c==inf) = -inf;              % 928us
    [~,move] = max(scores);             % 925us
    p.transition(move);                 % 1019us
    r.xtot = r.xtot + 1;                % 914us

    [mincost, bestmove] = min(c);
    if c(move) > mincost
      r.xerr = r.xerr + 1; 
      if training
        if isempty(svtr)
          svtr = gpuArray(f);
          betatr = zeros(1, model.n_cla, 'gpuArray');
          betatr2 = zeros(1, model.n_cla, 'gpuArray');
        else
          svtr(end+1,:) = f;
          betatr(end+1,:) = zeros(1, model.n_cla);
          betatr2(end+1,:) = zeros(1, model.n_cla);
        end
        betatr(end, bestmove) = 1;
        betatr(end, move) = -1;
      end
    end
    if training
      betatr2 = betatr2 + betatr;
    end

  end % while ((p.sptr > 1) || (p.wptr <= p.nword))

  %ti=11;wait(gpuDevice);r.t(ti)=r.t(ti)+toc;
  r.ntot = r.ntot + n;
  r.nerr = r.nerr + numel(find(h ~= p.head));
  for i=1:n
    if (isempty(regexp(s.form{i}, '^\W+$')) ||...
        ~isempty(regexp(s.form{i}, '^[\`\$]+$')))
      r.wtot = r.wtot + 1;
      if s.head(i) ~= p.head(i)
        r.werr = r.werr + 1;
      end %if
    end %if
  end %for
  %ti=12;wait(gpuDevice);r.t(ti)=r.t(ti)+toc;

  fprintf('.');
end % for s1=corpus
fprintf('\n');

r.xpct = r.xerr/r.xtot;
r.npct = r.nerr/r.ntot;
r.wpct = r.werr/r.ntot;

if training
  model.SV = gather(svtr)';
  model.beta = gather(betatr)';
  model.beta2 = gather(betatr2)';
  model = compactify(model);
end

toc;
end
