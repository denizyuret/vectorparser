tic(); r3=testparser(model03, dev_w, dev_h, idx); toc();
r3
r3.nerr/r3.ntot
r3.xerr/r3.xtot
x_te = x5k(idx,1:10000);
y_te = y5k(1:10000);
x_tr = x5k(idx,10001:end);
y_tr = y5k(10001:end);
p4 = struct('x_te', x_te, 'y_te', y_te, epsilon, 0.1, margin, 1, eta, 0.5);
tic(); model04 = model_sparsify(model03, x_tr, y_tr, p4); toc();
model04
tic(); [a4,b4]=model_predict_gpu(dev_x(idx,:), model04, 0); toc();
numel(find(a4 ~= dev_y))/numel(dev_y)
tic(); r4=testparser_par(model04, dev_w, dev_h, idx); toc();
r4
r4.nerr/r4.ntot
r4.xerr/r4.xtot
