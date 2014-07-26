function [x,y] = dumpfeatures(trn_w, trn_h, dump_singletons)

% do not output singleton moves (without alternative) by default
if (nargin < 3) dump_singletons = 0; end;

% figure out dimensions
d = length(ArcHybrid(trn_w{1}).features);
n=0;
for i=1:length(trn_w)
  n = n + 2 * size(trn_w{i}, 2) - 2;
end

x = zeros(d, n);
y = zeros(1, n);
nx = 0;

tic();
for ns=1:length(trn_w)
  w = trn_w{ns};
  h = trn_h{ns};
  p = ArcHybrid(w);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    c = p.oracle_cost(h);
    % there can be multiple 0 cost moves (SLR=RSL)
    % there can be no 0 cost moves (non-projective)
    % assert(((p.nwords > 20) || ((sum(isfinite(c)) > 0) && (sum(c==0) > 0))), ...
    % but there should always be some finite cost moves
    assert(sum(isfinite(c)) > 0, ...
           'ns=%d nw=%d nx=%d sptr=%d wptr=%d\nstack=%s\ncost=%s\nheads=\n%s\n%s\n%s\n', ...
           ns, p.nwords, nx, p.sptr, p.wptr, num2str(p.stack(1:p.sptr)), ...
           num2str(c), int2str([1:p.nwords]), int2str(h), int2str(p.heads));
    % only add a move if there is a choice
    [cost,best] = min(c);
    assert(best>0);
    assert(isfinite(cost));
    if (dump_singletons || (sum(isfinite(c)) > 1))
      nx = nx + 1;
      x(:,nx) = p.features();
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
