m0=model_init(@compute_kernel,hp);m0.batchsize=1250; m0.step=8;
tic();m1=k_perceptron_multi_train_gpu(trn_x(idx708,:),trn_y,m0);toc();
show_m1=m1
tic(); [a,b]=model_predict_gpu(dev_x(idx708,:), m1, 1); toc();
numel(find(a ~= dev_y))/numel(dev_y)
tic(); [a,b]=model_predict_gpu(dev_x(idx708,:), m1, 0); toc();
numel(find(a ~= dev_y))/numel(dev_y)
tic(); r1=testparser(m1, dev_w, dev_h, idx708); toc();
show_r1=r1
r1.nerr/r1.ntot
r1.xerr/r1.xtot
