function [n,w,w2] = trainparser_primal(sentences, heads, featidx, w, update)

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
  g.turan = gpuArray(turan(n.f));       % need this for poly3 basis calculation
  n.t = numel(find(g.turan));           % (nf + floor((nf-1)/2)*ceil((nf-1)/2))
  n.w = n.f*n.t;                        % size of basis fn expansion
  for c=1:n.c
    g.w{c} = gpuArray(zeros(1, n.w, 'single')); % actual weights
    g.u{c} = gpuArray(zeros(1, n.w, 'single')); % Daume's trick for w2
  end
else
  n.c = size(w,1);
  n.w = size(w,2);
  n.f = 1+numel(featidx);
  g.turan = gpuArray(turan(n.f));       % need this for poly3 basis calculation
  n.t = numel(find(g.turan));           % (nf + floor((nf-1)/2)*ceil((nf-1)/2))
  assert(n.w == n.f * n.t);
  for c=1:n.c  g.w{c} = gpuArray(single(w(c,:))); end
  if update % Only need averaging for training
    for c=1:n.c  g.u{c} = gpuArray(zeros(1, n.w, 'single')); end
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
    cost = p.oracle_cost(h);            % 106μs
    assert(any(isfinite(cost)));
    [mincost, best] = min(cost);
    feats = p.features();               % 637μs
    g.f1 = gpuArray(single([1, feats(featidx)'])); % 975μs
    g.f2 = g.f1' * g.f1;                % 1052μs
    g.f3(:) = g.f2(g.turan) * g.f1;     % 14873μs
    % scores = g.w * g.f3;              % mtimes with skinny matrices takes a very long time for some reason!
    for c=1:n.c
      scores(c) = gather(dot(g.w{c},g.f3));
    end                                 % 16050μs
    scores(cost == inf) = -inf;
    [maxscore, move] = max(scores);
    if cost(move) > mincost
      n.merr = n.merr + 1;
      if update
        g.w{best} = g.w{best} + g.f3;   % 7980μs
        g.w{move} = g.w{move} - g.f3;   % 7947μs
        wait(gpuDevice);
        g.u{best} = g.u{best} + n.move * g.f3; % 13043μs
        wait(gpuDevice);
        g.u{move} = g.u{move} - n.move * g.f3; % 12682μs
      end
    end                                 % 9226μs (avg)
    p.transition(move);                 % 126μs
  end % while: 40245μs per iteration on average
  n.werr = n.werr + numel(find(p.heads ~= h));
  fprintf('s=%d w=%d we=%.4f m=%d me=%.4f wps=%.2f t=%.2f\n', ...
          si, n.word, n.werr/n.word, n.move, n.merr/n.move, n.word/toc(), toc());
end % for

if update
  tic();fprintf('Transferring back...\n');
    w = []; u = [];
    for c=1:n.c
      w = [w ; gather(g.w{c})];
      wait(gpuDevice);
      u = [u ; gather(g.u{c})];
      wait(gpuDevice);
    end
    w2 = w - (1/n.move) * u;
  toc();
end

end % function
