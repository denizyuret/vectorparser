function [x,y] = dumpfeatures2(sentences, feats, dump_singletons)
% do not output singleton moves (without alternative) by default
if (nargin < 3) dump_singletons = 0; end;

% figure out dimensions
s1 = sentences{1};
p1 = archybrid(numel(s1.head));
f1 = features(p1, s1, feats);
nd = numel(f1);
nmax = 0;
for s1 = sentences
  s = s1{1};
  nmax = nmax + 2 * numel(s.head) - 2;
end

x = zeros(nd, nmax);
y = zeros(1,  nmax);
nx = 0;

tic();
for s1 = sentences
  s = s1{1};
  p = archybrid(numel(s.head));
  while ((p.sptr > 1) || (p.wptr <= p.nword))
    c = p.oracle_cost(s.head);
    % there can be multiple 0 cost moves (SLR=RSL)
    % there can be no 0 cost moves (non-projective)
    % assert(((p.nword > 20) || ((sum(isfinite(c)) > 0) && (sum(c==0) > 0))), ...
    % but there should always be some finite cost moves
    assert(sum(isfinite(c)) > 0, ...
           'nw=%d nx=%d sptr=%d wptr=%d\nstack=%s\ncost=%s\nheads=\n%s\n%s\n%s\n', ...
           p.nword, nx, p.sptr, p.wptr, num2str(p.stack(1:p.sptr)), ...
           num2str(c), int2str([1:p.nword]), int2str(s.head), int2str(p.head));
    % only add a move if there is a choice
    [cost,best] = min(c);
    assert(best>0);
    assert(isfinite(cost));
    if (dump_singletons || (sum(isfinite(c)) > 1))
      nx = nx + 1;
      x(:,nx) = features(p, s, feats);
      y(nx) = best;
    end
    p.transition(best);
  end
end
toc();
% we could output less than n instances
% assert(nx == n, 'nx=%d n=%d', nx, n);
x = x(:,1:nx);
y = y(1:nx);
end  % dumpfeatures
