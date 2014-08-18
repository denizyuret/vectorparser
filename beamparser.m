% TODO: can test without kernelcache or dump to make sure we are on the right track.
% TODO: test all combinations of predict/update/average/dump against beamparser (and for beam=1 vectorparser)
% TODO: try pa for faster convergence?

function [model, dump] = beamparser(model, corpus, varargin)

% Usage: beamparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = beamparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = beamparser(model, corpus, 'update', 0, 'predict', 0); %% dump features
% [model,dump] = beamparser(model, corpus, 'update', 0)); %% testing
% stats = eval_conll(corpus, dump);

msg('Initializing...');
m = beamparser_init(model, corpus, varargin, nargout);
msg('Processing sentences...');
t0 = tic;

for snum=1:numel(corpus)
  sentence = corpus{snum};
  [maxscorepath, mincostpath] = parse(m, sentence);
  if m.update
    m = update(m, maxscorepath, mincostpath);
  end
  if m.dump
    m = dodump(m, maxscorepath, mincostpath);
  end
  dot(snum, numel(corpus), t0);
end % for

msg('Terminating...');
[model, dump] = beamparser_term(m, model, corpus, varargin, nargout);

end % beamparser_dbg


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [maxscorepath, mincostpath] = parse(m, sentence)

% Each state in the two paths may have the following fields:

empty_state = ...
    struct('prev', [],...       % previous state
           'lastmove', [],...   % move that led to this state from prev
           'sumscore', [],...   % cumulative score including lastmove
           'ismincost', [],...  % true if state can be on mincostpath
           'parser', [],...     % the parser state
           'cost', [],...       % costs of moves from this state
           'feats', [],...      % feature vector for parser state
           'score', []);        % scores of moves from this state

% The initialstate is common to both paths.  It has no prev or
% lastmove fields.
%
% The final states come about in two ways:
%
% 1. If the sentence is finished.  In this case it will not have
% any valid moves.  So it does not need the fields feats, valid,
% cost, score.  It will have sumscore, prev, and lastmove.
%
% 2. If we hit early stop, i.e. oracle path dropped out of beam.
%
% In both cases the final states need to be included in the path to
% retrieve the last move and the last sumscore.
% 
% At the exit, candidates(1) will be the maxscorestate and by
% following its prev links we can construct the maxscorepath.
% However mincoststate has to be tracked separately.  If the
% candidates list does not include a mincoststate we still should
% have the mincoststate that just fell out of the beam.

% candidates are the states on the beam:
candidates(m.beam) = empty_state;
candidates(1).sumscore = 0;
candidates(1).ismincost = true;
ncandidates = 1;

% agenda contains children of candidates to be sorted for next round:
agenda(m.beam * m.nmove) = empty_state;
nagenda = 0;

% fmatrix is the feature matrix that allows score calculation in parallel
fmatrix = zeros(m.ndims, m.beam);

% depth is the length of the state.prev chain
depth = 1;

while 1
  
  % Set parser:
  % Here is which fields/variables have valid values at each point:
  % +prev +lastmove +sumscore +ismincost -parser -cost -feats -score 
  % -mincoststate +agenda -fmatrix

  if isempty(candidates(1).prev)
    if (ncandidates ~= 1) error('Not initial state'); end;
    candidates(1).parser = feval(m.parser, numel(sentence.head));
  else
    for c = 1:ncandidates
      candidates(c).parser = candidates(c).prev.parser.copy();
      candidates(c).parser.transition(candidates(c).lastmove);
    end
  end

  % Track mincoststate; maxscorestate is already in candidates(1)
  % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score
  % -mincoststate +agenda -fmatrix

  mincoststate = find_mincoststate(candidates, ncandidates, agenda, nagenda);

  % Check for early stop:
  % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score
  % +mincoststate -agenda -fmatrix

  if (m.early_stop && ~any(arrayfun(@(x) isfield(x,'ismincost'), candidates(1:ncandidates))))
    break
  end
  
  % Check for end of sentence:

  if ~any(candidates(1).parser.valid_moves())
    break
  end

  % Set cost and features:
  % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score
  % +mincoststate -agenda -fmatrix

  for c = 1:ncandidates
    if m.compute_costs
      candidates(c).cost = candidates(c).parser.oracle_cost(sentence.head);
    end
    if m.compute_features
      candidates(c).feats = features(candidates(c).parser, sentence, m.feats)';
      if m.compute_scores
        fmatrix(:,c) = candidates(c).feats;
      end
    end
  end

  % Computing scores in bulk is faster on gpu.
  % +prev +lastmove +sumscore +ismincost +parser +cost +feats -score
  % +mincoststate -agenda +fmatrix

  if m.compute_scores
    scores = compute_scores(m, fmatrix(:,1:ncandidates));
    for c = 1:ncandidates
      candidates(c).score = scores(:,c);
    end
  end

  % Refill agenda with children of candidates.
  % +prev +lastmove +sumscore +ismincost +parser +cost +feats +score
  % +mincoststate -agenda -fmatrix

  depth = depth + 1;
  a = 0;

  for c = 1:ncandidates
    cc = candidates(c);
    valid = cc.parser.valid_moves();
    for move = 1:m.nmove
      if ~valid(move) continue; end;
      a = a+1;
      agenda(a).prev = cc;
      agenda(a).lastmove = move;
      if m.predict
        agenda(a).sumscore = cc.sumscore + cc.score(move);
      else
        agenda(a).sumscore = cc.sumscore - cc.cost(move);
      end
      if (m.compute_costs && isfield(cc, 'ismincost') && (cc.cost(move) == min(cc.cost)))
        agenda(a).ismincost = true;
      end
    end % for move = 1:m.nmove
  end % for c = 1:ncandidates

  nagenda = a;
  ncandidates = min(m.beam, nagenda);
  [~, index] = sort([agenda(1:nagenda).sumscore], 'descend');
  candidates(1:ncandidates) = agenda(index(1:ncandidates));

  % +prev +lastmove +sumscore +ismincost -parser -cost -feats -score
  % -mincoststate +agenda -fmatrix

end % while

if (depth == 1) error('depth == 1'); end;
maxscorestate = candidates(1);
maxscorepath = cell(1, depth);
mincostpath = cell(1, depth);
while depth > 0
  if m.compute_costs
    if isempty(mincoststate) error('isempty(mincoststate)'); end;
    mincostpath{depth} = mincoststate;
    if (depth > 1) mincoststate = mincoststate.prev; end;
  end
  if m.compute_scores
    maxscorepath{depth} = maxscorestate;
    if (depth > 1) maxscorestate = maxscorestate.prev; end;
  end
  depth = depth - 1;
end

end % parse



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = update(m, maxscorepath, mincostpath)
% TODO: split into two blocks for kernel cache

npath = numel(maxscorepath);
beta = zeros(m.nmove, 2*npath);
nbeta = 0;

for ipath = 1:npath-1
  maxscorestate = maxscorepath{ipath};
  maxscoremove = maxscorepath{ipath+1}.lastmove;
  mincoststate = mincostpath{ipath};
  mincostmove = mincostpath{ipath+1}.lastmove;
  % TODO: look at ties
  samestate = isequal(maxscorestate, mincoststate);
  samemoves = (maxscoremove == mincostmove);
  if samestate && samemoves
    continue;
  else
    m.svtr(end+1,:) = maxscorestate.feats;
    nbeta = nbeta + 1;
    beta(maxscoremove, nbeta) = -1;
    if samestate
      beta(mincostmove, nbeta) = 1;
    else
      m.svtr(end+1,:) = mincoststate.feats;
      nbeta = nbeta + 1;
      beta(mincostmove, nbeta) = 1;
    end
  end
end % for ipath = 1:npath

m.beta2 = m.beta2 + m.beta;
if nbeta > 0
  beta = beta(:,1:nbeta);
  m.beta  = [m.beta,  beta];
  m.beta2 = [m.beta2, beta];
end

end % update



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = dodump(m, maxscorepath, mincostpath)
% TODO: check compatibility with eval_conll.
if m.compute_costs
  npath = numel(mincostpath);
  for ipath=1:npath-1
    m.yfeats(:,end+1) = mincostpath{ipath}.feats;
    m.ycost(:,end+1) = mincostpath{ipath}.cost;
    m.ymove(end+1) = mincostpath{ipath+1}.lastmove;
  end
end
if m.compute_scores
  npath = numel(maxscorepath);
  for ipath=1:npath-1
    state = maxscorepath{ipath};
    m.zfeats(:,end+1) = maxscorepath{ipath}.feats;
    m.zscore(:,end+1) = maxscorepath{ipath}.score;
    m.zmove(end+1) = maxscorepath{ipath+1}.lastmove;
  end
  m.sumscore(end+1) = maxscorepath{npath}.sumscore;
end
if isempty(m.sidx)
  m.sidx(1) = npath-1;
else
  m.sidx(end+1) = m.sidx(end) + npath - 1;
end
if m.compute_scores
  m.pred{end+1} = maxscorepath{npath}.parser.head;
else
  % output even when we are not predicting
  % heads differ in non-projective sentences.
  m.pred{end+1} = mincostpath{npath}.parser.head;
end
end % dodump



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function scores = compute_scores(m, x)
% TODO: split into two blocks for kernel cache
hp = m.kerparam;
if m.average beta = m.beta2; else beta = m.beta; end;
if size(x, 2) == 1   % this ugliness cuts the time in half when beam=1
  scores = gather(sum(bsxfun(@times, beta', (hp.gamma * (m.svtr * x) + hp.coef0).^hp.degree),1))';
else
  scores = gather(beta * (hp.gamma * (m.svtr * x) + hp.coef0) .^ hp.degree);
end
end % compute_scores


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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mincoststate = find_mincoststate(candidates, ncandidates, agenda, nagenda)
mincoststate = [];
for c = 1:ncandidates
  state = candidates(c);
  if isfield(state, 'ismincost')
    mincoststate = state;
    break;
  end
end
if ~isempty(mincoststate) return; end;
for c = 1:nagenda
  state = agenda(c);
  if isfield(state, 'ismincost')
    % state.parser is not set in agenda, fix it:
    state.parser = state.prev.parser.copy();
    state.parser.transition(state.lastmove);
    mincoststate = state;
    break;
  end
end
if isempty(mincoststate) error('Cannot find mincoststate'); end;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = beamparser_init(model, corpus, varargin_save, nargout_save)
% TODO: split into two blocks for kernel cache

m = model;  % return copy of model with extra information

for vi = 1:2:numel(varargin_save)
  v = varargin_save{vi};
  v1 = varargin_save{vi+1};
  switch v
   case 'beam'
    m.beam	 = v1;
   case 'predict' 
    m.predict = v1;
   case 'update'  
    m.update  = v1;
   case 'average' 
    m.average = v1;
   case 'gpu'     
    m.gpu     = v1;
   otherwise 
    error('Usage: [model, dump] = beamparser(model, corpus, opts)');
  end
end

if ~isfield(m, 'beam')
  m.beam = 10;
end
msg('Using beam size %d.', m.beam);

if ~isfield(m, 'predict')
  m.predict = true;  % predict: use the model for prediction (default), otherwise follow gold moves
end
if m.predict
  msg('Following moves predicted by the model.');
else
  msg('Following oracle moves only.');
end

if ~isfield(m, 'update')
  m.update = true;   % update: train the model (default), otherwise model is not updated
end
if m.update
  msg('Updating the model (training mode)');
else
  msg('Model will not be updated (testing mode)');
end

m.dump = (nargout_save >= 2);
if m.dump
  msg('Parse information will be dumped.');
else
  msg('No dump.');
end

m.compute_costs = m.update || ~m.predict;
m.compute_features = m.update || m.dump || m.predict;
m.compute_scores  = m.update || m.predict;
m.early_stop = m.update && m.predict;

assert(isfield(m,'parser'), 'Please specify model.parser.');
tmp_s = corpus{1};
tmp_p = feval(m.parser, numel(tmp_s.head));
m.nmove = tmp_p.NMOVE;
msg('Number of moves: %d.', m.nmove);

if m.compute_features
  assert(isfield(m,'feats'), 'Please specify model.feats.');
  assert(size(m.feats, 2) == 3, 'The feats matrix needs 3 columns.');
  [tmp_f, m.fidx] = features(tmp_p, tmp_s, m.feats);
  m.ndims = numel(tmp_f);
  msg('Number of features: %d, dimensions: %d', size(m.feats, 1), m.ndims);
end

if m.compute_scores
  assert(isfield(m,'kerparam') && ...
         strcmp(m.kerparam.type,'poly'), ...
         'Please specify poly kernel in model.kerparam.\n');

  if ~isfield(m,'average')
    m.average = (~m.update && isfield(m,'beta2') && ~isempty(m.beta2));
  elseif m.average
    assert(isfield(m,'beta2') && ~isempty(m.beta2),...
           'Please set model.beta2 for averaged model.');
  end
  if m.average
    msg('Using the averaged coefficients to predict.');
  else
    msg('Using the final coefficients to predict.');
  end

  if ~isfield(m,'SV') || isempty(m.SV)
    msg('Initializing empty model.');
    if m.update
      m.svtr = zeros(0, m.ndims);
      m.beta = zeros(0, m.nmove);
      m.beta2 = zeros(0, m.nmove);
    else
      error('Please specify model.SV');
    end
  else
    msg('Starting with nsv:%d', size(m.SV, 2));
    assert(size(m.SV, 1) == m.ndims);
    assert(size(m.beta, 1) == m.nmove);
    assert(size(m.SV, 2) == size(m.beta, 2));
    assert(all(size(m.beta) == size(m.beta2)));
    m.svtr = m.SV';
  end

  if ~isfield(m, 'gpu')
    m.gpu = gpuDeviceCount(); % Use the gpu if there is one
  end

  if m.gpu
    assert(gpuDeviceCount()>0, 'No GPU detected.');
    msg('Initializing GPU.');
    gdev = gpuDevice();
    reset(gdev);
    msg('Loading model to GPU.');
    m.svtr = gpuArray(m.svtr);
    m.beta = gpuArray(m.beta);
    m.beta2 = gpuArray(m.beta2);
  else
    msg('Not using GPU.');
  end

end % if m.compute_scores

if m.dump
  m.sidx = [];		% last move index of each sentence
  if m.compute_costs
    m.yfeats = [];      % mincoststate.feats
    m.ycost = [];       % mincoststate.cost
    m.ymove = [];       % mincostmove
  end
  if m.compute_scores
    m.zfeats = [];      % maxscorestate.feats
    m.zscore = [];      % maxscorestate.score
    m.zmove = [];       % maxscoremove
    m.sumscore = [];    % cumulative sentence scores
  end
  m.pred = {};          % predicted heads
end % if m.dump

end % beamparser_init


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model, dump] = beamparser_term(m, model, corpus, varargin_save, nargout_save)
if m.dump dump = m; end
if m.update
  model.SV = gather(m.svtr)';
  model.beta = gather(m.beta);
  model.beta2 = gather(m.beta2);
  model = compactify(model);
end
end % beamparser_term


