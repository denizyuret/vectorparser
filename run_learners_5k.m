x_tr = trn_x(idx, 1:217603);
y_tr = trn_y(1:217603);
x_te = dev_x(idx, :);
y_te = dev_y;
hp = struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3)
m0 = model_init(@compute_kernel,hp);
fprintf('last\tavg\tnsv\ttime\talgo\n');

tic();
m1 = k_perceptron_multi_train(x_tr, y_tr, m0)
m1ttm = toc();
m1nsv = size(m1.beta, 2);
m1lst = model_predict(x_te, m1, 0);
m1avg = model_predict(x_te, m1, 1);
m1lst_err = numel(find(m1lst ~= y_te))/numel(y_te)*100;
m1avg_err = numel(find(m1avg ~= y_te))/numel(y_te)*100;
fprintf('%.2f\t%.2f\t%d\t%.2f\tk_perceptron_multi_train\n', ...
        m1lst_err, m1avg_err, m1nsv, m1ttm);

tic();
m2 = k_pa_multi_train(x_tr, y_tr, m0)
m2ttm = toc();
m2nsv = size(m2.beta, 2);
m2lst = model_predict(x_te, m2, 0);
m2avg = model_predict(x_te, m2, 1);
m2lst_err = numel(find(m2lst ~= y_te))/numel(y_te)*100;
m2avg_err = numel(find(m2avg ~= y_te))/numel(y_te)*100;
fprintf('%.2f\t%.2f\t%d\t%.2f\tk_pa_multi_train\n', ...
        m2lst_err, m2avg_err, m2nsv, m2ttm);

tic();
m3 = k_projectron2_multi_train(x_tr, y_tr, m0)
m3ttm = toc();
m3nsv = size(m3.beta, 2);
m3lst = model_predict(x_te, m3, 0);
m3avg = model_predict(x_te, m3, 1);
m3lst_err = numel(find(m3lst ~= y_te))/numel(y_te)*100;
m3avg_err = numel(find(m3avg ~= y_te))/numel(y_te)*100;
fprintf('%.2f\t%.2f\t%d\t%.2f\tk_projectron2_multi_train\n', ...
        m3lst_err, m3avg_err, m3nsv, m3ttm);

path('libsvm-3.18/matlab', path);
tic();
m4 = svmtrain(y_tr', x_tr', '-t 1 -d 3 -g 1 -r 1 -c 1')
m4ttm = toc();
m4nsv = m4.totalSV;
[m4lst, m4acc, m4val] = svmpredict(y_te', x_te', m4);
m4lst_err = numel(find(m4lst ~= y_te'))/numel(y_te)*100;
fprintf('%.2f\t%.2f\t%d\t%.2f\tsvmtrain\n', ...
        m4lst_err, 0, m4nsv, m4ttm);
