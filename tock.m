function tock(cur, tot, t0, n)
persistent ncall;
if isempty(ncall) ncall = 0; end
ncall = ncall + 1;
if (nargin < 4) n = 10; end
if (nargin < 3) t1 = inf; else t1 = toc(t0); end
if (nargin < 2) tot = inf; end
if (nargin < 1) cur = ncall; end
if cur == tot
  fprintf('. %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
elseif mod(cur,n) == 0
  fprintf('.');
  if mod(cur, 10*n) == 0
    fprintf(' %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
  end
end
