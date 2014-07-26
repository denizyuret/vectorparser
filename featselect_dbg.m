% Algorithm SFFS from P. Somol, P. Pudil, J. Novovicova, and
% P. Paclik.  Adaptive floating search methods in feature
% selection. Pattern Recognition Letters, 20(11–13):1157–1163, 1999.

% Takes training and testing sets, polynomial kernel degree,
% starting feature combination, and cache of older results.
% Returns best feature combination of each size, their error rates
% (averaged model on test set), and the new cache of results.

function [best_feats, best_feats_err, cache] = featselect_dbg(x_tr,y_tr,x_te,y_te,degree,initfeats,cache)

% The last three args are optional, here are the defaults:
if nargin < 5
  degree = 3;
end
if nargin < 6
  initfeats = '';
end
if nargin < 7
  cache = containers.Map();
end

% Feature indices maps feature names to ranges of indices
[~,fidx,nidx] = featureindices();
assert(nidx == size(x_tr,1));
fidxkeys = keys(fidx);
fidxvals = values(fidx, fidxkeys);
nfeats = length(fidxkeys);

% Initialize argout:
best_feats = cell(1,nfeats);
best_feats_err = inf(1,nfeats);
cachekeys = keys(cache);
for i=1:numel(cachekeys)
  key = cachekeys{i};
  val = cache(key);
  feats = strsplit(key,',');
  nf = numel(feats);
  if val < best_feats_err(nf)
    best_feats_err(nf) = val;
    best_feats{nf} = feats;
  end
end

% Initialize model
hp = struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', degree)
model_bak = model_init(@compute_kernel,hp);
model_bak.step = 200;
model_bak.batchsize = 500;

fprintf('last\tavg\tnsv\ttime\tfeats\n');
if strcmp(initfeats,'')
  start = cell(0);
  nstart = 0;
else
  start = strsplit(initfeats,',');
  nstart = length(start);                 % current feature set size
  start_err = err(start);
  if start_err < best_feats_err(nstart)
    best_feats{nstart} = start;
    best_feats_err(nstart) = start_err;
  end    
end

while nstart < nfeats

  backtrack = 1;
  while ((nstart > 3) && backtrack)
    fprintf('# backtracking with %s\n', strjoin(start,','));
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

  best_try = '';
  best_try_err = inf;
  fprintf('# Trying children of %s\n', strjoin(start,','))
  for try_f=1:nfeats
    feature_to_try = fidxkeys{try_f};
    if find(strcmp(feature_to_try, start))
      continue;
    end
    new_features = start;
    new_features{end+1} = feature_to_try;
    new_err = err(new_features);
    if new_err < best_try_err
      best_try = feature_to_try;
      best_try_err = new_err;
    end
  end % for
  nstart = nstart + 1;
  if (best_try_err >= best_feats_err(nstart))
    start = best_feats{nstart};
    fprintf('# best_try_err=%g best(k)=%g reverting to %s\n', best_try_err, best_feats_err(nstart), strjoin(start,','));
    continue;
  end
  start{end+1} = best_try;
  assert(nstart == length(start));
  best_feats{nstart} = start;
  fprintf('# best_try_err=%g\n', best_try_err);
end

function s=err(f)                     % f is a cell array of feature names

fkey = strjoin(sort(f),',');
if ~isKey(cache, fkey)
  idx = [];
  for fj=1:length(f)
    idx = [idx, fidx(f{fj})];
  end % for
  x_te_idx = x_te(idx,:);
  x_tr_idx = x_tr(idx,:);
  
  tic();
  gpuDevice(1);
  model_perceptron = k_perceptron_multi_train_gpu(x_tr_idx,y_tr,model_bak);
  pred_perceptron_last = model_predict_gpu(x_te_idx,model_perceptron,0);
  pred_perceptron_av = model_predict_gpu(x_te_idx,model_perceptron,1);
  telapsed = toc();
  err_last = numel(find(pred_perceptron_last~= y_te))/numel(y_te)*100;
  err_av = numel(find(pred_perceptron_av~=y_te))/numel(y_te)*100;
  nsv = size(model_perceptron.beta, 2);
  fprintf('%.2f\t%.2f\t%d\t%.2f\t%s\n', ...
          err_last, err_av, nsv, telapsed, fkey);
  cache(fkey) = err_av;
else
  fprintf('%.2f\t%.2f\t%d\t%.2f\t%s\n', ...
          nan, cache(fkey), nan, 0, fkey);
end % if ~isKey(cache, fkey)
s = cache(fkey);
end % err

end % featselect_gpu
