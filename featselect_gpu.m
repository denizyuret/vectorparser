function featselect_gpu(m0, trn, dev, cachefile, initfeats)

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
% m0 = model_init(@compute_kernel,struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3));
% m0.parser = @archybrid;
% m0.feats = fv102;
% [~,trndump] = vectorparser(m0, trn, 'update', 0, 'predict', 0);
% [~,devdump] = vectorparser(m0, dev, 'update', 0, 'predict', 0);
% m0.step = 200; m0.batchsize = 500;
% featselect_gpu(m0,trndump,devdump,'foo.mat');

[cache,bestfeats,besterror,start,nfeats,nstart] = featselect_init(nargin);


while nstart < nfeats

  if (nstart > 1)
    start_err = err(start);
    if (start_err < besterror(nstart))
      bestfeats{nstart} = fkey(start);
      besterror(nstart) = start_err;
      fprintf('# newbest(%d)\t%g\t%s\n', nstart, start_err, fkey(start));
    end
  end

  backtrack = 1;
  while ((nstart > numel(start0)) && backtrack)
    fprintf('# backtracking(%d) with %s\n', nstart, fkey(start));
    best_try_i = 0;
    best_try_err = inf;
    for try_i=1:nstart
      new_features = start;
      new_features(try_i) = [];
      new_err = err(new_features);
      if new_err < best_try_err
        best_try_err = new_err;
        best_try_i = try_i;
      end %if
    end %for
    if best_try_err < besterror(nstart-1)
      nstart = nstart-1;
      start(best_try_i) = [];
      bestfeats{nstart} = fkey(start);
      besterror(nstart) = best_try_err;
      fprintf('# newbest(%d)\t%g\t%s\n', nstart, best_try_err, fkey(start));
    else
      backtrack = 0;
    end
  end

  best_try = 0;
  best_try_err = inf;
  fprintf('# children(%d) of %s\n', nstart, fkey(start));
  for feature_to_try=1:nfeats
    if ismember(feature_to_try, start) continue; end
    new_features = start;
    new_features(end+1) = feature_to_try;
    new_err = err(new_features);
    if new_err < best_try_err
      best_try = feature_to_try;
      best_try_err = new_err;
    end
  end % for
  nstart = nstart + 1;
  start(end+1) = best_try;
  assert(nstart == length(start));

  if best_try_err < besterror(nstart)
    bestfeats{nstart} = fkey(start);
    besterror(nstart) = best_try_err;
    fprintf('# newbest(%d)\t%g\t%s\n', nstart, best_try_err, fkey(start));
  elseif ~strcmp(fkey(start), bestfeats{nstart})
    fprintf('# reverting(%d) from %g %s to %g %s\n', nstart, best_try_err, ...
            fkey(start), besterror(nstart), bestfeats{nstart});
    start = key2f(bestfeats{nstart});
  end

end  % while nstart < nfeats


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
end

%%%%%%%%%%%%%%%%%
function s=err(f)                       % f is an array of indices into trn.fidx

fk = fkey(f);
if ~isKey(cache, fk)                    % x matrix has an instance with all features in each column
  fprintf('Computing err for %s\n', fk);
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
  
  tic();
  gpuDevice(1);
  m1 = perceptron(x_tr, trn.y, m0);
  [~,~,e0] = perceptron(x_te, dev.y, m1, 'update', 0, 'average', 0);
  [~,~,e1] = perceptron(x_te, dev.y, m1, 'update', 0, 'average', 1);
  t1 = toc();
  nsv = size(m1.beta, 2);
  fprintf('%g\t%g\t%g\t%g\t%s\n', e0, e1, nsv/numel(trn.y), t1, fk);
  cache(fk) = e1;
  save(cachefile, 'cache');
else
  fprintf('%g\t%g\t%d\t%g\t%s\n', nan, cache(fk), nan, 0, fk);
end % if ~isKey(cache, fk)
s = cache(fk);
end % err


%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cache,bestfeats,besterror,start,nfeats,nstart] = featselect_init(nargin_save)

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

% Init cache
if exist(cachefile, 'file')
  load(cachefile);
  assert(exist('cache') ~= 0, '%s does not contain a cache.\n', cachefile);
else
  cache = containers.Map();
  save(cachefile, 'cache');
end

% Init bestfeats
nfeats = numel(trn.fidx);
bestfeats = cell(1,nfeats);
besterror = inf(1,nfeats);
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

% Initialize starting feature combination
if nargin_save < 5
  start0 = [];
else
  start0 = key2f(mat2str(sortrows(initfeats)));
end
start = start0;
nstart = numel(start);

fprintf('last\tavg\tnsv\ttime\tfeats\n');

end % featselect_init

end % featselect_gpu
