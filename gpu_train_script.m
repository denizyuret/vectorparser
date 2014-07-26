gpuDevice(1);
m0=model_init(@compute_kernel,hp);m0.batchsize=1250; m0.step=8;
tic();m1=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m0);toc();
tic();m2=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m1);toc();
tic();m3=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m2);toc();
tic();m4=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m3);toc();
tic();m5=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m4);toc();
