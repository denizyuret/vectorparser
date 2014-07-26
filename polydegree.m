% load conllWSJToken_wikipedia2MUNK-50.mat;
% Elapsed time is 9.625437 seconds.
% [x5k,y5k]=dumpfeatures(trn_w(1:5000),trn_h(1:5000));
% Elapsed time is 341.514392 seconds.

[fi,i] = featureindices();
assert(i == size(x5k,1));
feats = {'n0','n0l','n0r','n0+','n0-', ...
         's0','s0l','s0r','s0+','s0-', ...
         's1','s1l','s1r','s1+','s1-', ...
         'n1','n1l','n1r','n1+','n1-', ...
         'n0l1','n0l1l','n0l1r','n0l1+','n0l1-', ...
         's0r1','s0r1l','s0r1r','s0r1+','s0r1-', ...
         's0l1','s0l1l','s0l1r','s0l1+','s0l1-', ...
         's1r1','s1r1l','s1r1r','s1r1+','s1r1-', ...
         'n0s0','s0s1'};
idx = [];
for i=1:length(feats) idx = [idx, fi(feats{i})]; end

x_te = x5k(idx,1:10000);
y_te = y5k(1:10000);
x_tr = x5k(idx,10001:end);
y_tr = y5k(10001:end);

hp.type = 'poly';
hp.gamma = 1;
hp.coef0 = 1;

fprintf('degree\tlast\tavg\tnsv\ttime\n');
for d=1:9
  hp.degree = d;
  model = model_init(@compute_kernel, hp);
  model.step = 1000000;
  tic();
  model = k_perceptron_multi_train(x_tr, y_tr, model);
  telapsed = toc();
  nsv = size(model.beta, 2);
  last = numel(find(y_te ~= model_predict(x_te, model, 0)))/numel(y_te)*100;
  avg = numel(find(y_te ~= model_predict(x_te, model, 1)))/numel(y_te)*100;
  fprintf('%d\t%.2f\t%.2f\t%d\t%.2f\n', hp.degree, last, avg, nsv, telapsed);
end
