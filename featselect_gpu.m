function [best_feats, best_feats_err, cache] = featselect_gpu(m0, trn, dev, initfeats, cache)

% Algorithm SFFS from P. Somol, P. Pudil, J. Novovicova, and
% P. Paclik.  Adaptive floating search methods in feature
% selection. Pattern Recognition Letters, 20(11–13):1157–1163, 1999.

% Takes a model (specifying kernel type (kerparam), parser type
% (parser); training and development sets (dump structures with
% instance matrix (x), correct moves (y), list of features (feats),
% and feature indices (fidx) from vectorparser), and optional starting
% feature combination (initfeats) which should be an array of indices
% into trn.fidx; and an optional cache of older results.  Returns the
% best feature combination of each size, their error rates (averaged
% model on dev set), and the new cache of results.  Feature
% combinations are represented by sorted index arrays into trn.fidx.

% Typical usage:
% m0 = model_init(@compute_kernel,struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3));
% m0.parser = @archybrid;
% m0.feats = fv102;
% [~,trndump] = vectorparser(m0, trn, 'update', 0, 'predict', 0);
% [~,devdump] = vectorparser(m0, dev, 'update', 0, 'predict', 0);
% m0.step = 200; m0.batchsize = 500;
% [a,b,c] = featselect_gpu(m0,trndump,devdump);


nargin_save = nargin;
featselect_init();
start = initfeats;
nstart = numel(start);
nfeats = numel(trn.fidx);


while nstart < nfeats

  if (nstart > 1)
    start_err = err(start);
    if (start_err < best_feats_err(nstart))
      best_feats{nstart} = start;
      best_feats_err(nstart) = start_err;
    end
  end

  backtrack = 1;
  while ((nstart > 1) && backtrack)
    fprintf('# backtracking with %s\n', fkey(start));
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
    if best_try_err < best_feats_err(nstart-1)
      nstart = nstart-1;
      start(best_try_i) = [];
      best_feats{nstart} = start;
      best_feats_err(nstart) = best_try_err;
      fprintf('# best_try=%d best_try_err=%g\n', best_try_i, best_try_err);
    else
      backtrack = 0;
    end
  end

  best_try = 0;
  best_try_err = inf;
  fprintf('# Trying children of %s\n', fkey(start));
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
  if (best_try_err >= best_feats_err(nstart))
    start = best_feats{nstart};
    fprintf('# best_try_err=%g best(k)=%g reverting to %s\n', best_try_err, best_feats_err(nstart), fkey(start));
    continue;
  end
  start(end+1) = best_try;
  assert(nstart == length(start));
  best_feats{nstart} = start;
  fprintf('# best_try_err=%g\n', best_try_err);

end  % while nstart < nfeats


%%%%%%%%%%%%%%%%%%%%%%
function fk = fkey(f)
% f is an array of indices into trn.fidx
fk = mat2str(sort(f));
end % fkey


%%%%%%%%%%%%%%%%%
function s=err(f)                       % f is an array of indices into trn.fidx

fk = fkey(f);
if ~isKey(cache, fk)                    % x matrix has an instance with all features in each column
  fprintf('Computing err for %s\n', fk);
  idx = logical([]);                    % idx is a boolean index into the rows of x matrix (features)
  for fj=f                              
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
  m1 = k_perceptron_multi_train_gpu(x_tr,trn.y,m0);
  z0 = model_predict_gpu(x_te,m1,0);
  z1 = model_predict_gpu(x_te,m1,1);
  t1 = toc();
  e0 = numel(find(z0 ~= dev.y))/numel(dev.y)*100;
  e1 = numel(find(z1 ~=dev.y))/numel(dev.y)*100;
  nsv = size(m1.beta, 2);
  fprintf('%.2f\t%.2f\t%d\t%.2f\t%s\n', ...
          e0, e1, nsv, t1, fk);
  cache(fk) = e1;
else
  fprintf('%.2f\t%.2f\t%d\t%.2f\t%s\n', ...
          nan, cache(fk), nan, 0, fk);
end % if ~isKey(cache, fk)
s = cache(fk);
end % err


%%%%%%%%%%%%%%%%%%%%%%%%%%
function featselect_init()

% The last two args are optional, here are the defaults:
if nargin_save < 4
  initfeats = [];
end
if nargin_save < 5
  cache = containers.Map();
end

% Check to make sure model m0 and dumps trn and dev have all we want:
assert(isfield(m0, 'parser'), 'Please specify the parser type m0.parser.\n');
assert(isfield(m0, 'kerparam'), 'Please specify the kernel type m0.kerparam.\n');
assert(isfield(trn, 'fidx'), 'Please specify the feature indices trn.fidx.\n');
assert(strcmp(m0.kerparam.type,'poly'));
assert(size(trn.x, 1) == trn.fidx(end));
assert(all(dev.fidx(:) == trn.fidx(:)));
assert(size(dev.x, 1) == dev.fidx(end));
% m0.step = 200;
% m0.batchsize = 500;

% Initialize argout:
nfeats = numel(trn.fidx);
best_feats = cell(1,nfeats);
best_feats_err = inf(1,nfeats);
cachekeys = keys(cache);
for i=1:numel(cachekeys)
  feats = eval(cachekeys{i});
  fterr = cache(key);
  nf = numel(feats);
  if fterr < best_feats_err(nf)
    best_feats_err(nf) = fterr;
    best_feats{nf} = feats;
  end
end

fprintf('last\tavg\tnsv\ttime\tfeats\n');

end % featselect_init

end % featselect_gpu
