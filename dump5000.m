% load conllWSJToken_wikipedia2MUNK-50.mat;
% [x,y]=dumpfeatures(trn_w(1:5000),trn_h(1:5000));

[fi,i] = featureindices();
assert(i == size(x,1));
feats = {'n0','s0','s1','n1','n0l1','s0r1','s0l1','s1r1','s0r'};
idx = [];
for i=1:length(feats)
  idx = [idx, fi(feats{i})];
end % for
fprintf('Using poly kernel on %s\n', strjoin(feats,','));

hp.type = 'poly';
hp.gamma = 1;
hp.coef0 = 1;
hp.degree = 3;
disp(hp);

x_te = x(idx,1:10000);
y_te = y(1:10000);

fprintf('train\tlast\tavg\tnsv\ttime\n');
for train=10000:10000:length(y)-10000
x_tr = x(idx,10001:10000+train);
y_tr = y(10001:10000+train);
[last,avg,nsv,time,model] = run_perceptron(x_tr, y_tr, x_te, y_te, hp);
fprintf('%d\t%.2f\t%.2f\t%d\t%.2f\n', length(y_tr), last, avg, nsv, time);
end
x_tr = x(idx,10001:end);
y_tr = y(10001:end);
[last,avg,nsv,time,model] = run_perceptron(x_tr, y_tr, x_te, y_te, hp);
fprintf('%d\t%.2f\t%.2f\t%d\t%.2f\n', length(y_tr), last, avg, nsv, time);
