function [model, dump] = beamparser_dbg(model, corpus, varargin)

% Usage: beamparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = beamparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = beamparser(model, corpus, 'update', 0, 'predict', 0); %% dump features
% [model,dump] = beamparser(model, corpus, 'update', 0)); %% testing
% stats = eval_conll(corpus, dump);

m = beamparser_init(varargin, nargout);
agenda = cell(m.beam * m.nmove, 2);
fprintf('Processing sentences...\n');

for snum=1:numel(corpus)
  sentence = corpus{snum};
  parser = feval(m.parser, numel(sentence.head));
  candidates = {0, struct('parser', parser)};
  while 1
    nagenda = 0;
    for cnum = 1:size(candidates, 1)
      [score, state] = candidates{cnum,:};
      parser = state.parser;
      valid = parser.valid_moves();
      if (sum(valid) == 0) continue; end;
      state.feats = features(parser, sentence, m.feats)';
      state.score = compute_scores(m, state.feats)';
      state.cost  = parser.oracle_cost(sentence.head);
      candidates{cnum,2} = state;
      for move = 1:m.nmove
        if ~valid(move) continue; end;
        newscore = score + state.score(move);
        newstate = struct('parser', copy(parser), 'prev', state);
        newstate.parser.transition(move);
        newstate.move = move;
        nagenda = nagenda + 1;
        agenda(nagenda,:) = { newscore, newstate };
      end % for move = 1:m.nmove
    end % for cnum = 1:size(candidates, 1)
    if (nagenda == 0)  break; end;
    candidates = top_b(agenda(1:nagenda,:), m.beam);
  end % while 1
  update_dump(top_b(candidates, 1));
  fprintf('.');
end % for snum=1:numel(corpus)
fprintf('\n');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = beamparser_init(varargin_save, nargout_save)

m = model;  % copy of model with extra information

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
   case 'beam'
    m.beam	 = v1;
   case 'dbg'
    m.dbg	 = v1;
   otherwise 
    error('Usage: [model, dump] = beamparser(model, corpus, opts)');
  end
end

if ~isfield(m, 'dbg')
  m.dbg = 0;
end

if ~isfield(m, 'beam')
  m.beam = 64;
end

if ~isfield(m, 'predict')
  m.predict = true;  % predict: use the model for prediction (default), otherwise follow gold moves
end
 
if ~isfield(m, 'update')
  m.update = true;   % update: train the model (default), otherwise model is not updated
end

m.dump = (nargout_save >= 2);
m.compute_costs = m.update || m.dump || ~m.predict;
m.compute_features = m.update || m.dump || m.predict;
m.compute_scores  = m.update || m.predict;

assert(isfield(model,'parser'), 'Please specify model.parser.');
tmp_s = corpus{1};
tmp_p = feval(model.parser, numel(tmp_s.head));
nc = tmp_p.NMOVE;
m.nmove = nc;

if m.compute_features
  assert(isfield(model,'feats'), 'Please specify model.feats.');
  assert(size(model.feats, 2) == 3, 'The feats matrix needs 3 columns.');
  tmp_f = features(tmp_p, tmp_s, model.feats);
  nd = numel(tmp_f);
end

if m.compute_scores
  assert(isfield(model,'kerparam') && ...
         strcmp(model.kerparam.type,'poly'), ...
         'Please specify poly kernel in model.kerparam.\n');
  m.kerparam = model.kerparam;

  if ~isfield(m,'average')
    m.average = (isfield(model,'beta2') && ~isempty(model.beta2) && ~m.update);
  elseif m.average
    assert(isfield(model,'beta2') && ~isempty(model.beta2),...
           'Please set model.beta2 for averaged model.');
  end

  if ~isfield(model,'SV') || isempty(model.SV)
    if m.update
      m.svtr = zeros(0, nd);
      m.betatr = zeros(0, nc);
      m.betatr2 = zeros(0, nc);
    else
      error('Please specify model.SV');
    end
  else
    assert(size(model.SV, 1) == nd);
    assert(size(model.beta, 1) == nc);
    assert(size(model.SV, 2) == size(model.beta, 2));
    assert(all(size(model.beta) == size(model.beta2)));
    m.svtr = model.SV';
    m.betatr = model.beta';
    m.betatr2 = model.beta2';
  end

  if ~isfield(m, 'gpu')
    m.gpu = gpuDeviceCount(); % Use the gpu if there is one
  end

  if m.gpu
    assert(gpuDeviceCount()>0, 'No GPU detected.');
    fprintf('Loading model on GPU.\n');
    gpuDevice(1);
    m.svtr = gpuArray(m.svtr);
    m.betatr = gpuArray(m.betatr);
    m.betatr2 = gpuArray(m.betatr2);
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
    dump.x = [];
    dump.sidx = [];
  end
  if m.compute_costs
    dump.y = [];
    dump.cost = [];
  end
  if m.compute_scores
    dump.z = [];
    dump.score = [];
    dump.s1 = [];
  end
  if m.predict
    dump.pred = {};
  end
  if m.compute_features
    dump.feats = model.feats;
    sentence = corpus{1};
    p0 = feval(model.parser, numel(sentence.head));
    [~,dump.fidx] = features(p0, sentence, model.feats);
  end
end % if m.dump

end % beamparser_init


%%%%%%%%%%%%%%%%%%%%%%
function update_dump(c)
[score, state] = c{:};
dump.s1(end+1) = score;
heads = state.parser.head;
dump.pred{end+1} = heads;
nstates = 2*numel(heads) - 1;
states = cell(1, nstates);
for i=nstates:-1:1
  states{i} = state;
  if i>1 
    if ~isfield(state,'prev') error('Bad state'); end;
    state = state.prev; 
  end
end
for i=1:numel(states)-1
  state = states{i};
  dump.x(:,end+1) = state.feats;
  dump.cost(:,end+1) = state.cost;
  dump.score(:,end+1) = state.score;
  dump.z(end+1) = states{i+1}.move;
  [~,dump.y(end+1)] = min(state.cost);
end
dump.sidx(end+1) = numel(dump.z);
end % update_dump

end % beamparser


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Same matrix operation has different costs on gpu:
% score = gather(sum(bsxfun(@times, betatr, (hp.gamma * full(svtr * ftr) + hp.coef0).^hp.degree),1)); % 7531us
% Matrix multiplication is less efficient than array mult:
% score = gather(b + beta * (hp.gamma * (svtr * ftr) + hp.coef0).^hp.degree); % 17310us
% Even the transpose makes a difference:
% score = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (f * sv) + hp.coef0).^hp.degree),1)); % 11364us

function score = compute_scores(m, ftr)
if 0 %DBG
  score = rand(1, size(m.betatr, 2));
  return;
end
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
