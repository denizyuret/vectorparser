% load conllWSJToken_wikipedia2MUNK-50.mat;
% Elapsed time is 9.625437 seconds.
% [x5k,y5k]=dumpfeatures(trn_w(1:5000),trn_h(1:5000));
% Elapsed time is 341.514392 seconds.
% load dump5000-model.mat; % loads model
% Elapsed time is 0.704188 seconds.

[fi,i] = featureindices();
assert(i == size(x5k,1));
feats = {'n0','s0','s1','n1','n0l1','s0r1','s0l1','s1r1','s0r'};
idx = [];
for i=1:length(feats)
  idx = [idx, fi(feats{i})];
end % for

x_te = x5k(idx,1:10000);
y_te = y5k(1:10000);
x_tr = x5k(idx,10001:end);
y_tr = y5k(10001:end);
return
model.step = 100;
m = model_sparsify(x_tr, y_tr, model, 1);

p0 = model_predict(x_te, model, 1);
numel(find(p0 ~= y_te))
p = model_predict(x_te, m, 0);
numel(find(p ~= y_te))
