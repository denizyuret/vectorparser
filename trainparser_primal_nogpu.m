function [n,w,w2] = trainparser_primal_nogpu(sentences, heads, featidx, w, update)

n = struct();  % Return some stats as first argout
g = struct();  % Put all gpu variables in one place

% By default update the weights (training mode)
% For test mode provide a weight vector w and set update=0
if (nargin < 5) update=1; end
assert(update~=1 || nargout==3, 'Need three output arguments for training.');

tic(); fprintf('Resetting GPU...\n');
gpu=gpuDevice(1); 	                % clears the gpu


toc();tic(); fprintf('Transferring model...\n');
if nargin < 4  % We are doing training
  assert(update==1, 'Must provide weights for testing.');
  p = ArcHybrid(sentences{1});
  n.c = p.nmoves;
  if (nargin < 3) featidx = 1:numel(p.features()); end
  clear p;
  n.f = 1+numel(featidx);
  g.turan = (turan(n.f));         % need this for poly3 basis calculation
  n.t = numel(find(g.turan));             % (nf + floor((nf-1)/2)*ceil((nf-1)/2))
  n.w = n.f*n.t;                           % size of basis fn expansion
  for c=1:n.c
    g.w{c} = (zeros(1, n.w, 'single')); % actual weights
    g.u{c} = (zeros(1, n.w, 'single')); % Daume's trick for w2
  end
else
  n.c = size(w,1);
  n.w = size(w,2);
  n.f = 1+numel(featidx);
  g.turan = (turan(n.f));         % need this for poly3 basis calculation
  n.t = numel(find(g.turan));             % (nf + floor((nf-1)/2)*ceil((nf-1)/2))
  assert(n.w == n.f * n.t);
  for c=1:n.c  g.w{c} = (single(w(c,:))); end
  if update % Only need averaging for training
    for c=1:n.c  g.u{c} = (zeros(1, n.w, 'single')); end
  end
end

n.sent = length(sentences);
n.word = 0;
n.move = 0;
n.werr = 0;
n.merr = 0;

toc();tic();fprintf('Parsing in %g dims...\n', n.w);
for si=1:n.sent
  s = sentences{si};
  h = heads{si};
  p = ArcHybrid(s);
  n.word = n.word + numel(h);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    n.move = n.move + 1;
    cost = p.oracle_cost(h);            % 191us
    assert(any(isfinite(cost)));        % 46us
    [mincost, best] = min(cost);        % 46us
    feats = p.features();               % 595us
    g.f1 = (single([1;feats(featidx)])); % 1368us
    g.f2 = g.f1*g.f1';                  % 4005us
    clear g.f3;                         % 1307us
    g.f3 = g.f2(g.turan)*g.f1';          % 41132us
    g.f3 = reshape(g.f3, 1, n.w);        % 958us
    % scores = g.w * g.f3; % This takes a very long time for some reason!
    for c=1:n.c
      scores(c) = (dot(g.w{c},g.f3));
    end                                 % 43963us
    scores(cost == inf) = -inf;         % 930us TODO: is this good for learning?  try moving after max.
    [maxscore, move] = max(scores);     % 927us
    if cost(move) > mincost
      n.merr = n.merr + 1;
      if update
        g.w{best} = g.w{best} + g.f3;     % 28500us
        g.w{move} = g.w{move} - g.f3;     % 28500us
        g.u{best} = g.u{best} + n.move * g.f3;
        g.u{move} = g.u{move} - n.move * g.f3;
      end
    end
    p.transition(move);                 % 130us
  end % while
  n.werr = n.werr + numel(find(p.heads ~= h));
  fprintf('s=%d w=%d we=%.4f m=%d me=%.4f wps=%.2f ', ...
          si, n.word, n.werr/n.word, n.move, n.merr/n.move, n.word/toc());
toc();
end % for

if update
  tic();fprintf('Transferring back...\n');
    w = []; u = [];
    for c=1:n.c
      w = [w ; (g.w{c})];
      u = [u ; (g.u{c})];
    end
    w2 = w - (1/n.move) * u;
  toc();
end

end % function
