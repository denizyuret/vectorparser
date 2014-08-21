function featselect(m0, trn, dev, cachefile, initfeats)

% Algorithm SFFS from P. Somol, P. Pudil, J. Novovicova, and
% P. Paclik.  Adaptive floating search methods in feature
% selection. Pattern Recognition Letters, 20(11–13):1157–1163, 1999.

% Takes a model (m0), specifying kernel type (kerparam), parser type
% (parser); training (trn) and development (dev) sets, dump structures
% with instance matrix (x), correct moves (y), list of features
% (feats), and feature indices (fidx) from vectorparser; a cachefile,
% and optional starting feature combination (initfeats).  An internal
% cache holds the dev errors of feature combinations.  The cachefile
% is a .mat file that will be created if it doesn't exist, will be
% read to initialize the cache if it does exist, and will be updated
% with new results.  The feature combinations are represented as
% feature matrix strings:
%
% fkey(f) = mat2str(sortrows(trn.feats(f,:)))

% Typical usage:
% m0 = struct(
%  'kerparam', struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3),
%  'parser', @archybrid, 'feats', fv102, 'step', 100000, 'batchsize', 1000);
% [~,trndump] = vectorparser(m0, trn, 'update', 0, 'predict', 0);
% [~,devdump] = vectorparser(m0, dev, 'update', 0, 'predict', 0);
% featselect(m0,trndump,devdump,'foo.mat');

cache = [];
featselect_init(nargin);

start_changed = true;
while start_changed
  start_changed = false;

  if nstart > 0
    update_cachefile();
    update_bestfeats();
    start_err = err(start);
    msg('# starting(%d)\t%g\t%s', nstart, start_err, fstr(start));
    if (start_err <= besterror(nstart))
      msg('# newbest(%d)\t%g\t%s', nstart, start_err, fstr(start));
    end
  end

  if nstart > 1
    msg('# backtracking(%d) with %s', nstart, fstr(start));
    best = struct('i', 0, 'e', inf, 'f', []);
    update_cachefile();
    update_bestfeats();
    for curr_i=1:nstart
      curr_f = start;
      curr_f(curr_i) = [];
      curr_e = err(curr_f);
      if curr_e < best.e
        best = struct('i', curr_i, 'e', curr_e, 'f', curr_f);
        msg('# better(%d)\t%g\t-%s\t%s', nstart-1, ...
            best.e, fstr(start(best.i)), fstr(best.f));
      end %if
    end %for
    if best.e <= besterror(nstart-1)
      msg('# newbest(%d)\t%g\t-%s\t%s', nstart-1, ...
          best.e, fstr(start(best.i)), fstr(best.f));
    end
    if best.e <= start_err
      nstart = nstart-1;
      start(best.i) = [];
      start_changed = true;
      continue;
    end
  end

  if nstart < nfeats
    msg('# children(%d) of %s', nstart, fstr(start));
    best = struct('i', 0, 'e', inf, 'f', []);
    update_cachefile();
    update_bestfeats();
    for curr_i=1:nfeats
      if ismember(curr_i, start) continue; end
      curr_f = start;
      curr_f(end+1) = curr_i;
      curr_e = err(curr_f);
      if curr_e < best.e
        best = struct('i', curr_i, 'e', curr_e, 'f', curr_f);
        msg('# better(%d)\t%g\t+%s\t%s', nstart+1, ...
            best.e, fstr(best.i), fstr(best.f));
      end
    end % for
    if best.e < besterror(nstart+1)
      msg('# newbest(%d)\t%g\t+%s\t%s', nstart+1, ...
          best.e, fstr(best.i), fstr(best.f));
    end
    update_cachefile();
    update_bestfeats();
    if best.e < start_err
      nstart = nstart + 1;
      start(end+1) = best.i;
      start_changed = true;
    elseif ~strcmp(fkey(start), bestfeats{nstart})
      msg('# reverting(%d) from %g %s to %g %s', nstart, ...
          start_err, fstr(start), besterror(nstart), bestfeats{nstart});
      start = key2f(bestfeats{nstart});
      start_changed = true;
    end
  end % if nstart < nfeats

end  % while nstart <= nfeats


%%%%%%%%%%%%%%%%%%%%%%
function fk = fkey(f)
% f is an array of indices into trn.fidx and rows of trn.feats
fk = mat2str(sortrows(trn.feats(f,:)));
end % fkey


%%%%%%%%%%%%%%%%%%%%%%
function f = key2f(fk)
% reverse of fkey
[~,f] = ismember(eval(fk), trn.feats, 'rows');
assert(strcmp(fkey(f),fk));
end % key2f


%%%%%%%%%%%%%%%%%%%%%
function fs = fstr(f)
fs = '';
for i=1:numel(f)
  fs = [fs fsymbol(trn.feats(f(i),:)) ' '];
end
end % fstr


%%%%%%%%%%%%%%%%%
function s=err(f)                       % f is an array of indices into trn.fidx
fk = fkey(f);
if ~isKey(cache, fk)                    % x matrix has an instance with all features in each column
  % msg('Computing err for %s', fstr(f));
  idx = logical([]);                    % idx is a boolean index into the rows of x matrix (features)
  for j=1:numel(f)
    fj=f(j);
    fidx2 = trn.fidx(fj);               % trn.fidx has the end position of each feature group in an x column
    if fj > 1                           
      fidx1 = trn.fidx(fj-1) + 1;       % the starting position is either one more than the end of last fgroup
    else
      fidx1 = 1;                        % or is 1 for the first fgroup.
    end
    idx(fidx1:fidx2) = true;
  end % for
  x_te = dev.x(idx,:);
  x_tr = trn.x(idx,:);
  
  t0 = tic;
  m1 = perceptron(x_tr, trn.y, m0);
  [~,~,e1] = perceptron(x_te, dev.y, m1, 'update', 0, 'average', 1);
  t1 = toc(t0);
  nsv = size(m1.beta, 2);
  fprintf('==>\t%g\t%g\t%g\t%s\n', e1, nsv/numel(trn.y), t1, fstr(f));
  cache(fk) = e1;
else
  fprintf('==>\t%g\t%d\t%g\t%s\n', cache(fk), 0, 0, fstr(f));
end % if ~isKey(cache, fk)
s = cache(fk);
end % err


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function featselect_init(nargin_save)

% Check to make sure model m0 and dumps trn and dev have all we want:
assert(isfield(m0, 'parser'), 'Please specify the parser type m0.parser.\n');
assert(isfield(m0, 'kerparam'), 'Please specify the kernel type m0.kerparam.\n');
assert(strcmp(m0.kerparam.type,'poly'), 'Only poly kernels are supported.\n');
assert(isfield(trn, 'fidx'), 'Please specify the feature indices trn.fidx.\n');
assert(isfield(trn, 'feats'), 'Please specify the feature matrix trn.feats.\n');
assert(size(trn.x, 1) == trn.fidx(end));
assert(size(trn.feats, 1) == numel(trn.fidx));
assert(all(dev.fidx(:) == trn.fidx(:)));
assert(all(dev.feats(:) == trn.feats(:)));
assert(size(dev.x, 1) == size(trn.x, 1));
% m0.step = 200;
% m0.batchsize = 500;

nfeats = numel(trn.fidx);
bestfeats = cell(1,nfeats);
besterror = inf(1,nfeats);
cache = containers.Map();
update_cachefile();
update_bestfeats();

% Initialize starting feature combination
if nargin_save >= 5
  start = key2f(mat2str(sortrows(initfeats)));
elseif any(isfinite(besterror));
  [~,nstart] = min(besterror);
  start = bestfeats{nstart};
else
  start = [];
end
nstart = numel(start);
fprintf('==>\tavg\tnsv\ttime\tfeats\n');

end % featselect_init


%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_cachefile()

% Check if someone else added data to our cachefile
% and merge with our cache if they did.
if exist(cachefile, 'file')
  tmp = load(cachefile);
  assert(isfield(tmp,'cache'), '%s does not contain a cache.\n', cachefile);
  cachekeys = keys(tmp.cache);
  for i=1:numel(cachekeys)
    key = cachekeys{i};
    if isKey(cache, key)
      assert(cache(key) == tmp.cache(key), ...
             'Cache mismatch: %g~=%g for %s', ...
             cache(key), tmp.cache(key), key);
    else
      cache(key) = tmp.cache(key);
    end
  end
end

% Try to save cache in a threadsafe manner
threadtemp = [tempname '.mat'];
save(threadtemp, 'cache');
runme = sprintf('flock -x %s -c ''mv %s %s''', cachefile, threadtemp, cachefile);
system(runme);

end % update_cachefile


%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_bestfeats()
cachekeys = keys(cache);
for i=1:numel(cachekeys)
  fstr = cachekeys{i};
  ferr = cache(fstr);
  flen = size(eval(fstr), 1);
  if ferr < besterror(flen)
    besterror(flen) = ferr;
    bestfeats{flen} = fstr;
  end
end
end % update_bestfeats

end % featselect

