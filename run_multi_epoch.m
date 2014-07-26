m10=compactify(m10)

tic;m11=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m10)
toc;tic;m11 = compactify(m11)
toc;tic;[a11,b11]=model_predict_gpu(dev_x(idx,:), m11, 1);
toc;tic;gtrans11 = numel(find(a11 ~= dev_y))/numel(dev_y)
toc;tic;r11=trainparser_gpu(m11, dev1, fv804)
toc;tic;save -v7.3 logs/m11 m11 a11 b11 r11
toc;

tic;m12=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m11)
toc;tic;m12 = compactify(m12)
toc;tic;[a12,b12]=model_predict_gpu(dev_x(idx,:), m12, 1);
toc;tic;gtrans12 = numel(find(a12 ~= dev_y))/numel(dev_y)
toc;tic;r12=trainparser_gpu(m12, dev1, fv804)
toc;tic;save -v7.3 logs/m12 m12 a12 b12 r12
toc;

tic;m13=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m12)
toc;tic;m13 = compactify(m13)
toc;tic;[a13,b13]=model_predict_gpu(dev_x(idx,:), m13, 1);
toc;tic;gtrans13 = numel(find(a13 ~= dev_y))/numel(dev_y)
toc;tic;r13=trainparser_gpu(m13, dev1, fv804)
toc;tic;save -v7.3 logs/m13 m13 a13 b13 r13
toc;

tic;m14=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m13)
toc;tic;m14 = compactify(m14)
toc;tic;[a14,b14]=model_predict_gpu(dev_x(idx,:), m14, 1);
toc;tic;gtrans14 = numel(find(a14 ~= dev_y))/numel(dev_y)
toc;tic;r14=trainparser_gpu(m14, dev1, fv804)
toc;tic;save -v7.3 logs/m14 m14 a14 b14 r14
toc;

tic;m15=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m14)
toc;tic;m15 = compactify(m15)
toc;tic;[a15,b15]=model_predict_gpu(dev_x(idx,:), m15, 1);
toc;tic;gtrans15 = numel(find(a15 ~= dev_y))/numel(dev_y)
toc;tic;r15=trainparser_gpu(m15, dev1, fv804)
toc;tic;save -v7.3 logs/m15 m15 a15 b15 r15
toc;

tic;m16=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m15)
toc;tic;m16 = compactify(m16)
toc;tic;[a16,b16]=model_predict_gpu(dev_x(idx,:), m16, 1);
toc;tic;gtrans16 = numel(find(a16 ~= dev_y))/numel(dev_y)
toc;tic;r16=trainparser_gpu(m16, dev1, fv804)
toc;tic;save -v7.3 logs/m16 m16 a16 b16 r16
toc;

tic;m17=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m16)
toc;tic;m17 = compactify(m17)
toc;tic;[a17,b17]=model_predict_gpu(dev_x(idx,:), m17, 1);
toc;tic;gtrans17 = numel(find(a17 ~= dev_y))/numel(dev_y)
toc;tic;r17=trainparser_gpu(m17, dev1, fv804)
toc;tic;save -v7.3 logs/m17 m17 a17 b17 r17
toc;

tic;m18=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m17)
toc;tic;m18 = compactify(m18)
toc;tic;[a18,b18]=model_predict_gpu(dev_x(idx,:), m18, 1);
toc;tic;gtrans18 = numel(find(a18 ~= dev_y))/numel(dev_y)
toc;tic;r18=trainparser_gpu(m18, dev1, fv804)
toc;tic;save -v7.3 logs/m18 m18 a18 b18 r18
toc;

tic;m19=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m18)
toc;tic;m19 = compactify(m19)
toc;tic;[a19,b19]=model_predict_gpu(dev_x(idx,:), m19, 1);
toc;tic;gtrans19 = numel(find(a19 ~= dev_y))/numel(dev_y)
toc;tic;r19=trainparser_gpu(m19, dev1, fv804)
toc;tic;save -v7.3 logs/m19 m19 a19 b19 r19
toc;

tic;m20=k_perceptron_multi_train_gpu(trn_x(idx,:),trn_y,m19)
toc;tic;m20 = compactify(m20)
toc;tic;[a20,b20]=model_predict_gpu(dev_x(idx,:), m20, 1);
toc;tic;gtrans20 = numel(find(a20 ~= dev_y))/numel(dev_y)
toc;tic;r20=trainparser_gpu(m20, dev1, fv804)
toc;tic;save -v7.3 logs/m20 m20 a20 b20 r20
toc;
