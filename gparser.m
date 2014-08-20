classdef gparser < matlab.mixin.Copyable

properties (SetAccess = immutable)
parser 	% the parser object, e.g. @archybrid.
fselect	% features to be used
kerparam % kernel parameters
nmove   % number of possible transitions
ndims   % dimensionality of feature vectors
fidx    % end index of features in feats
end

properties (SetAccess = public)
update	% whether to update model parameters
predict % probability of following maxscoremove rather than mincostmove
average	% use beta2 (averaged coefficients) if true
gpu     % whether to use the gpu
output  % what to output
end

properties (SetAccess = private)
corpus  % last corpus parsed
feats 	% feature vectors representing parser states
cost	% cost of each move
score   % score of each move
move    % the moves executed
head    % the heads predicted
sidx    % sentence end indices in move array
eval    % evaluation metrics
end

properties (SetAccess = private)
SV      % support vectors
beta    % final weights
beta2   % averaged weights
end


methods (Access = public)

%%%%%%%%%%%%%%%%%%%%%%%%%
function parse(m, corpus)
initialize_model(m, corpus);
t0 = tic;
for snum=1:numel(corpus)
  s = corpus{snum};
  p = feval(m.parser, m.sentence_length(s));
  valid = p.valid_moves();
  while any(valid)
    mycost = []; myscore = [];
    if m.compute.cost
      mycost = p.oracle_cost(s.head);
      if m.output.cost m.cost(:,end+1) = mycost; end
    end
    if m.compute.feats
      frow = features(p, s, m.fselect);
      fcol = frow';
      if m.output.feats m.feats(:,end+1) = fcol; end
    end
    if m.compute.score
      myscore = compute_kernel(m, fcol);
      if m.output.score m.score(:,end+1) = myscore; end
    end
    if m.update
      perceptron_update(m, frow, mycost, myscore);
    end
    mymove = pick_move(m, valid, mycost, myscore);
    if m.output.move m.move(end+1) = mymove; end
    p.transition(mymove);
    valid = p.valid_moves();
  end % while 1
  if m.output.head m.head{end+1} = p.head; end
  if m.output.move m.sidx(end+1) = numel(m.move); end
  m.dot(snum, numel(corpus), t0);
end % for snum=1:numel(corpus)
if m.output.eval eval_model(m, corpus); end
finalize_model(m, corpus);
end % parse


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function set_model_parameters(m, model)
m.SV = model.SV;
m.beta = model.beta;
m.beta2 = model.beta2;
end % set_model_parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = gparser(parser, fselect, kerparam, corpus)
m.parser = parser;
m.fselect = fselect;
m.kerparam = kerparam;
s1 = corpus{1}; % need corpus for dims
p1 = feval(m.parser, m.sentence_length(s1));
m.nmove = p1.NMOVE;
[f1,m.fidx] = features(p1, s1, m.fselect);
m.ndims = numel(f1);
end % gparser

end % methods (Access = public)

methods (Access = private)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function initialize_model(m, corpus)

msg('gparser(%d,%d) corpus(%d)', m.nmove, m.ndims, numel(corpus));
if isempty(m.update) m.update = 1; end
if isempty(m.predict) m.predict = 1; end
if isempty(m.gpu) m.gpu = gpuDeviceCount(); end
if isempty(m.output) m.output = struct('corpus',1,'feats',1,'cost',1,'score',1,'move',1,'head',1,'eval',1); end

if m.output.corpus m.corpus = corpus; end
m.feats = [];
m.cost = [];
m.score = [];
m.move = [];
m.head = [];
m.sidx = [];
m.eval = [];

m.compute.cost = m.output.cost || m.update || ~m.predict;
m.compute.feats = m.output.feats || m.update || m.predict;
m.compute.score  = m.output.score || m.update || m.predict;

if m.compute.score
  assert(strcmp(m.kerparam.type,'poly'), 'Please specify poly kernel in model.kerparam.\n');
  if isempty(m.average)
    m.average = (~isempty(m.beta2) && ~m.update);
  elseif m.average
    assert(~isempty(m.beta2), 'Please set model.beta2 for averaged model.');
  end

  % We are using 1 and 2 for the old and the new SV blocks
  % For the final vs average model we'll use bfin vs bavg

  if isempty(m.SV)
    m.svtr1 = zeros(0, m.ndims);
    m.bfin1 = zeros(0, m.nmove);
    m.bavg1 = zeros(0, m.nmove);
  else
    assert(size(m.SV, 1) == m.ndims);
    assert(size(m.SV, 2) == size(m.beta, 2) && size(m.SV, 2) == size(m.beta2, 2));
    assert(size(m.beta, 1) == m.nmove && size(m.beta2, 1) == m.nmove);
    m.svtr1 = m.SV';
    m.bfin1 = m.beta';
    m.bavg1 = m.beta2';
  end
  m.svtr2 = zeros(0, m.ndims);
  m.bfin2 = zeros(0, m.nmove);
  m.bavg2 = zeros(0, m.nmove);

  if m.gpu
    assert(gpuDeviceCount()>0, 'No GPU detected.');
    msg('Resetting GPU.');
    gdev = gpuDevice;
    reset(gdev);
    msg('Loading model on GPU.');
    m.svtr1 = gpuArray(m.svtr1);
    m.bfin1 = gpuArray(m.bfin1);
    m.bavg1 = gpuArray(m.bavg1);
    m.svtr2 = gpuArray(m.svtr2);
    m.bfin2 = gpuArray(m.bfin2);
    m.bavg2 = gpuArray(m.bavg2);
  end

  if m.update && ~isempty(m.feats)
    msg('Computing cache scores.');tmp=tic;
    [~,scores] = perceptron(m.feats, [], m, 'update', 0, 'average', m.average);
    msg('Initializing kernel cache.');toc(tmp);tmp=tic;
    m.cache = kernelcache(m.feats, scores);
    msg('done');toc(tmp);
  end
end % if m.compute.score

msg('update=%d predict=%g average=%d gpu=%d', m.update, m.predict, m.average, m.gpu);
o = m.output;
msg('output: corpus=%d feats=%d cost=%d score=%d move=%d head=%d eval=%d', ...
    o.corpus, o.feats, o.cost, o.score, o.move, o.head, o.eval);

end % initialize_model


%%%%%%%%%%%%%%%%%%%%%%%%%%
function dot(m, cur, tot, t0)
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function n = sentence_length(m, s)
if isfield(s, 'head')
  n = numel(s.head);
elseif isfield(s, 'wvec')
  n = size(s.wvec, 2);
else
  error('Cannot get sentence length');
end
end % sentence_length


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function perceptron_update(m, frow, cost, score)
m.bavg1 = m.bavg1 + m.bfin1;
m.bavg2 = m.bavg2 + m.bfin2;
[maxscore, maxscoremove] = max(score);
[mincost, mincostmove] = min(cost);
% TODO: try other update methods
if cost(maxscoremove) > mincost
  m.svtr2(end+1,:) = frow;
  newbeta = zeros(1, m.nmove);
  newbeta(mincostmove) = 1;
  newbeta(maxscoremove) = -1;
  m.bfin2(end+1,:) = newbeta;
  m.bavg2(end+1,:) = newbeta;
end % if cost(maxscoremove) > mincost
end % perceptron_update


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function move = pick_move(m, valid, cost, score)
if rand <= m.predict
  [~,move] = max(score);
  if ~valid(move)
    % TODO: we could also choose mincostmove here
    zscore = score;
    zscore(~valid) = -inf;
    [~,move] = max(zscore);
  end
else
  [~,move] = min(cost);
end
end % pick_move


%%%%%%%%%%%%%%%%%%%%%%%%%%
function finalize_model(m, corpus)
if m.update
  m.SV = [gather(m.svtr1); gather(m.svtr2)]';
  m.beta = [gather(m.bfin1); gather(m.bfin2)]';
  m.beta2 = [gather(m.bavg1); gather(m.bavg2)]';
  compactify_model(m);
  clear m.cache;
end
clear m.svtr1 m.svtr2 m.bfin1 m.bfin2 m.bavg1 m.bavg2;
end % finalize_model


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function compactify_model(m)

% [C, ia, ic] = unique(A,'rows')
% Find the unique rows C(u,d) of A(n,d) and the index vectors ia(u,1) and ic(n,1), such that ...
% C = A(ia,:) and A = C(ic,:).

msg('Finding unique SV in %d...', size(m.SV, 2));
[~, ia, ic] = unique(m.SV', 'rows');

msg('Saving %d unique SV.', numel(ia));
b2 = ~isempty(m.beta2);
nc = size(m.beta, 1);
newbeta  = zeros(nc, numel(ia));
if b2 newbeta2 = zeros(nc, numel(ia)); end
assert(numel(ic) == size(m.beta, 2));

for oldi=1:numel(ic)
  newi = ic(oldi);
  newbeta(:,newi) = newbeta(:,newi) + m.beta(:,oldi);
  if b2 newbeta2(:,newi) = newbeta2(:,newi) + m.beta2(:,oldi); end
end

m.SV = m.SV(:,ia);
m.beta = newbeta;
if b2 m.beta2 = newbeta2; end

end % compactify_model


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function eval_model(m, corpus)

% TODO turn on move, cost, vs if eval is on
r = struct();
r.move_cnt = numel(m.move);
mincost = min(m.cost);
movecost = m.cost(sub2ind(size(m.cost), m.move, 1:numel(m.move)));
r.move_err = sum(movecost > mincost);
r.move_pct = r.move_err / r.move_cnt;

r.sent_cnt = 0; r.sent_err = 0;
r.head_cnt = 0; r.head_err = 0;
r.word_cnt = 0; r.word_err = 0;

nmove = 0;

for i=1:numel(corpus)
  s = corpus{i};
  h = s.head;
  p = m.head{i};
  nword = numel(h);
  head_err = sum(h ~= p);

  r.sent_cnt = r.sent_cnt + 1;
  if (head_err > 0) r.sent_err = r.sent_err + 1; end

  r.head_cnt = r.head_cnt + nword;
  r.head_err = r.head_err + head_err;

  if (i == 1) move1=1; else move1=m.sidx(i-1)+1; end
  move2 = m.sidx(i);
  sumcost = sum(movecost(move1:move2));
  assert(sumcost == head_err, ...
         'Discrepancy in sentence %d: %d ~= %d moves(%d,%d)', ...
         i, movecost, head_err, move1, move2);

  for j=1:numel(h)
    if (isempty(regexp(s.form{j}, '^\W+$')) ||...
        ~isempty(regexp(s.form{j}, '^[\`\$]+$')))
      r.word_cnt = r.word_cnt + 1;
      if h(j) ~= p(j)
        r.word_err = r.word_err + 1;
      end %if
    end %if
  end % for j=1:numel(h)
end % for i=1:numel(corpus)
assert(r.head_err == sum(movecost));

r.sent_pct = r.sent_err / r.sent_cnt;
r.head_pct = r.head_err / r.head_cnt;
r.word_pct = r.word_err / r.word_cnt;
m.eval = r;
end % eval_model


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score = compute_kernel(m, fcol)

% Same matrix operation has different costs on gpu:
% score = gather(sum(bsxfun(@times, btr, (hp.gamma * full(svtr * fcol) + hp.coef0).^hp.degree),1)); % 7531us
% Matrix multiplication is less efficient than array mult:
% score = gather(b + beta * (hp.gamma * (svtr * fcol) + hp.coef0).^hp.degree); % 17310us
% Even the transpose makes a difference:
% score = gather(b + sum(bsxfun(@times, beta, (hp.gamma * (frow * sv) + hp.coef0).^hp.degree),1)); % 11364us

hp = m.kerparam;
if m.average
  b1tr = m.bavg1;
  b2tr = m.bavg2;
else
  b1tr = m.bfin1;
  b2tr = m.bfin2;
end
score1 = [];
if ~isempty(m.cache)
  score1 = m.cache.get(fcol);
end
if isempty(score1)
  % score1 = gather(b1 * (hp.gamma * (m.svtr1 * fcol) + hp.coef0).^hp.degree);
  score1 = gather(sum(bsxfun(@times, b1tr, (hp.gamma * (m.svtr1 * fcol) + hp.coef0).^hp.degree),1))';
end
% score2 = gather(b2 * (hp.gamma * (m.svtr2 * fcol) + hp.coef0).^hp.degree);
score2 = gather(sum(bsxfun(@times, b2tr, (hp.gamma * (m.svtr2 * fcol) + hp.coef0).^hp.degree),1))';
score = score1 + score2;
end % compute_kernel

end % methods (Access = private)


properties (Access = private)
compute % what to compute
cache
svtr1
svtr2
bfin1
bfin2
bavg1
bavg2
end


end % classdef gparser < matlab.mixin.Copyable
