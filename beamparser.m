% Usage: beamparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = beamparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = beamparser(model, corpus, 'update', 0, 'predict', 0); %% dump features
% [model,dump] = beamparser(model, corpus, 'update', 0)); %% testing
% stats = eval_conll(corpus, dump);

function [model, dump] = beamparser(model, corpus, varargin)

m = beamparser_init(varargin, nargout);
fprintf('Processing sentences...\n');

for snum=1:numel(corpus)
  sentence = corpus{snum};
  % candidates are triples of score, state, history
  s0 = 0;
  p0 = feval(model.parser, numel(sentence.head));
  h0 = struct('x',[],'y',[],'z',[],'score',[],'cost',[]);
  candidates = { s0, p0, h0 };
  agenda = {};
  while 1
    for cnum = 1:size(candidates, 1)
      s0 = candidates{cnum,1};
      p0 = candidates{cnum,2};
      h0 = candidates{cnum,3};
      if m.dbg fprintf('cand[%d]: %g %s\n', cnum, s0, mat2str(h0.z)); end
      valid = p0.valid_moves;
      if (sum(valid) == 0) continue; end
      feats = features(p0, sentence, model.feats);
      score = compute_scores(m, feats');
      cost = p0.oracle_cost(sentence.head);
      [~, mincostmove] = min(cost);
      for move_i = 1:p0.NMOVE
        if ~valid(move_i) continue; end
        s1 = s0 + score(move_i);
        p1 = copy(p0);
        p1.transition(move_i);
        h1 = h0;
        h1.score = [h1.score, score'];  % score is a row vector
        h1.cost = [h1.cost, cost'];     % cost is a row vector
        h1.x = [h1.x, feats'];          % feats is a row vector
        h1.y = [h1.y, mincostmove];
        h1.z = [h1.z, move_i];
        agenda = [ agenda; { s1, p1, h1 } ];
        if m.dbg fprintf('agenda += %g %s\n', s1, mat2str(h1.z)); end
      end
    end
    if (numel(agenda) == 0)  break; end
    candidates = top_b(agenda, m.beam);
    agenda = {};
  end % while 1

  c1 = top_b(candidates, 1);
  s1 = c1{1,1};
  p1 = c1{1,2};
  h1 = c1{1,3};
  dump.pred{end+1} = p1.head;
  dump.x = [dump.x, h1.x];
  dump.y = [dump.y, h1.y];
  dump.z = [dump.z, h1.z];
  dump.s1 = [dump.s1, s1];
  dump.score = [dump.score, h1.score];
  dump.cost = [dump.cost, h1.cost];
  dump.sidx = [dump.sidx, numel(dump.z)];

  fprintf('.');
end % for snum=1:numel(corpus)
fprintf('\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function opts = beamparser_init(varargin_save, nargout_save)

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
   case 'beam'
    opts.beam	 = v1;
   case 'dbg'
    opts.dbg	 = v1;
   otherwise 
    error('Usage: [model, dump] = beamparser(model, corpus, opts)');
  end
end

if ~isfield(opts, 'dbg')
  opts.dbg = 0;
end

if ~isfield(opts, 'beam')
  opts.beam = 64;
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
  opts.kerparam = model.kerparam;

  if ~isfield(opts,'average')
    opts.average = (isfield(model,'beta2') && ~isempty(model.beta2) && ~opts.update);
  elseif opts.average
    assert(isfield(model,'beta2') && ~isempty(model.beta2),...
           'Please set model.beta2 for averaged model.');
  end

  if ~isfield(model,'SV') || isempty(model.SV)
    if opts.update
      opts.svtr = zeros(0, nd);
      opts.betatr = zeros(0, nc);
      opts.betatr2 = zeros(0, nc);
    else
      error('Please specify model.SV');
    end
  else
    assert(size(model.SV, 1) == nd);
    assert(size(model.beta, 1) == nc);
    assert(size(model.SV, 2) == size(model.beta, 2));
    assert(all(size(model.beta) == size(model.beta2)));
    opts.svtr = model.SV';
    opts.betatr = model.beta';
    opts.betatr2 = model.beta2';
  end

  if ~isfield(opts, 'gpu')
    opts.gpu = gpuDeviceCount(); % Use the gpu if there is one
  end

  if opts.gpu
    assert(gpuDeviceCount()>0, 'No GPU detected.');
    fprintf('Loading model on GPU.\n');
    gpuDevice(1);
    opts.svtr = gpuArray(opts.svtr);
    opts.betatr = gpuArray(opts.betatr);
    opts.betatr2 = gpuArray(opts.betatr2);
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
    dump.sidx = [];
  end
  if opts.compute_costs
    dump.y = [];
    dump.cost = [];
  end
  if opts.compute_scores
    dump.z = [];
    dump.score = [];
    dump.s1 = [];
  end
  if opts.predict
    dump.pred = {};
  end
  if opts.compute_features
    dump.feats = model.feats;
    sentence = corpus{1};
    p0 = feval(model.parser, numel(sentence.head));
    [~,dump.fidx] = features(p0, sentence, model.feats);
  end
end % if opts.dump

end % beamparser_init


%%%%%%%%%%%%%%%%%%%%%%
% function update_dump()
% if opts.compute_features
%   dump.x(:,end+1) = ftr;
% end
% if opts.compute_costs
%   dump.y(end+1) = mincostmove;
%   dump.cost(:,end+1) = cost;
% end
% if opts.compute_scores
%   dump.z(end+1) = execmove;
%   dump.score(:,end+1) = score;
% end
% end % update_dump

end % beamparser


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Same matrix operation has different costs on gpu:
% score = gather(sum(bsxfun(@times, betatr, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1)); % 7531us
% Matrix multiplication is less efficient than array mult:
% score = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
% Even the transpose makes a difference:
% score = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree),1)); % 11364us

function score = compute_scores(m, ftr)
hp = m.kerparam;
if isempty(m.svtr)
  score = zeros(1, size(m.betatr, 2));
elseif m.average
  score = gather(sum(bsxfun(@times, m.betatr2, (hp.gamma * full(m.svtr * ftr) + hp.coef0).^hp.degree),1));
else
  score = gather(sum(bsxfun(@times, m.betatr, (hp.gamma * full(m.svtr * ftr) + hp.coef0).^hp.degree),1));
end
end % compute_scores


%%%%%%%%%%%%%%%%%%%%%%%%
function c = top_b(a, b)
c = sortrows(a,-1);
c = c(1:min(b, end),:);
end
