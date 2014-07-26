hp = struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3);
m0 = model_init(@compute_kernel, hp);
m0.step = 10;
for bs=500:500:3000
  batchsize = bs
  m0.batchsize = bs;
  gpuDevice(1);
  tic();m1 = k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m0); ...
             toc();
end
