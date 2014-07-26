path('dogma',path);
load logs/biyofiz
load logs/dumpfeatures
idx804 = featureindices({'n0','s0','s1','n1','n0l1','s0r1','s0l1','s1r1','s0r'});
idx708=featureindices({'n0','n0l1','n0l2','n1','s0','s0r','s0r1','s0s1','s1'});
x_te = x5k(idx,1:10000);
y_te = y5k(1:10000);
x_tr = x5k(idx,10001:end);
y_tr = y5k(10001:end);
model5k.step = 1000;
p = struct('x_te', x_te, 'y_te', y_te, 'epsilon', 0.1, 'margin', 1, 'eta', 0.5);
hp = struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3);
