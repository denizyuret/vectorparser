function [w,w2] = trainparser_primal_12(sentences, heads, featidx, w, update)

tic(); fprintf('Resetting GPU...\n');
gpu=gpuDevice(1); 	                % clears the gpu

toc();tic(); fprintf('Transferring model...\n');
if nargin < 4 	% initialize w(nc,nw)
  p = ArcHybrid(sentences{1});
  nc = p.nmoves;
  if nargin < 3
    featidx = 1:numel(p.features());
  end
  clear p;
  nf = numel(featidx);
  nw = nf*nf*(nf+1)/2; % TODO: should improve this to nf*(nf+1)*(nf+2)/6
  g_w = cell(1,nc);                     % actual weights
%  g_u = cell(1,nc);                     % Daume's trick for w2
  for c=1:nc
    g_w{c} = gpuArray(zeros(1, nw, 'single'));
%    g_u{c} = gpuArray(zeros(1, nw, 'single'));
  end
else
  nc = size(w,1);
  nw = size(w,2);
  nf = numel(featidx);
  g_w = cell(1,nc);
  for c=1:nc
    g_w{c} = gpuArray(single(w(c,:)));
%    g_u{c} = gpuArray(zeros(1, nw, 'single'));
  end
end
g_triu = gpuArray(triu(true(nf,nf)));

toc();tic();fprintf('Training in %g dims...\n', nw);
nwords = 0;
ntrain = 0;
werr = 0;
merr = 0;
for si=1:length(sentences)
  s = sentences{si};
  h = heads{si};
  p = ArcHybrid(s);
  nwords = nwords + numel(h);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    ntrain = ntrain + 1;
    cost = p.oracle_cost(h);            % 191us
    assert(any(isfinite(cost)));        % 46us
    [mincost, best] = min(cost);    % 46us
    feats = p.features();               % 595us
    wait(gpuDevice);
    g_f1 = gpuArray(single(feats(featidx))); % 1368us
    g_f2 = g_f1*g_f1';                  % 4005us
    clear g_f3;                         % 1307us
    g_f3 = g_f2(g_triu)*g_f1';          % 41132us
    g_f3 = reshape(g_f3, 1, nw);        % 958us
    % scores = g_w * g_f3; % This takes a very long time for some reason!
    for c=1:nc
      scores(c) = gather(dot(g_w{c},g_f3));
    end                                 % 43963us
    scores(cost == inf) = -inf;         % 930us TODO: is this good for learning?  try moving after max.
    [maxscore, move] = max(scores);     % 927us
    if cost(move) > mincost
      merr = merr + 1;
      if update
      g_w{best} = g_w{best} + g_f3;     % 28500us
      g_w{move} = g_w{move} - g_f3;     % 28500us
%      g_u{best} = g_u{best} + ntrain * g_f3;
%      g_u{move} = g_u{move} - ntrain * g_f3;
      end
    end
    p.transition(move);                 % 130us
  end % while
  fprintf('%d ', si);toc();
  werr = werr + numel(find(p.heads ~= h));
  fprintf('s=%d w=%d we=%.4f m=%d me=%.4f wps=%.2f ', ...
          si, nwords, werr/nwords, ntrain, merr/ntrain, nwords/toc());
end % for
fprintf('\nSentences=%d Words=%d Words/Sec=%g\n', length(sentences), ...
        nwords, nwords/toc());
toc();

if update
tic();fprintf('Transferring back...\n');
w = [];
% u = [];
w2 = [];
for c=1:nc
  w = [w ; gather(g_w{c})];
%  u = [u ; gather(g_u{c})];
end
% w2 = w - (1/ntrain) * u;
toc();
end

end % function


