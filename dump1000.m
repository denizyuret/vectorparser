% path('dogma', path);
% 

tic();
fprintf('Loading dump1000 data set\n');
load dump1000.mat;

x_te = x(:,1:10000);
y_te = y(1:10000);
x_tr = x(:,10001:end);
y_tr = y(10001:end);
%x_tr = x(:,10001:20000);
%y_tr = y(10001:20000);
toc();

% for i=1:9
% hp.type = 'poly';
% hp.gamma = 1;
% hp.coef0 = 1;
% hp.degree = i;
% fprintf('Using %dth degree poly kernel\n', hp.degree);

for i=[.07,.09]
hp.type = 'rbf';
hp.gamma = i;
fprintf('Using rbf kernel with gamma=%f\n', hp.gamma);

model_bak = model_init(@compute_kernel,hp);

tic();
%% PERC
% train Perceptron
fprintf('Training Perceptron model...\n');
model_perceptron = k_perceptron_multi_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',size(model_perceptron.beta, 2));
fprintf('Number of support vectors averaged solution:%d\n',size(model_perceptron.beta2, 2));
fprintf('Testing last solution...');
pred_perceptron_last = model_predict(x_te,model_perceptron,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_perceptron_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_perceptron_av = model_predict(x_te,model_perceptron,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_perceptron_av~=y_te))/numel(y_te)*100);
toc();
end
return

%% PA_I
tic();
fprintf('Training PA-I model...\n');
model_pa1 = k_pa_multi_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',size(model_pa1.beta, 2));
fprintf('Number of support vectors averaged solution:%d\n',size(model_pa1.beta2, 2));
fprintf('Testing last solution...');
pred_pa1_last = model_predict(x_te,model_pa1,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_pa1_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_pa1_av = model_predict(x_te,model_pa1,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_pa1_av~=y_te))/numel(y_te)*100);
toc();

%% PA_II
tic();
fprintf('Training PA-II model...\n');
model_pa2 = model_bak;
model_pa2.update = 2;
model_pa2 = k_pa_multi_train(x_tr,y_tr,model_pa2);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',size(model_pa2.beta, 2));
fprintf('Number of support vectors averaged solution:%d\n',size(model_pa2.beta2, 2));
fprintf('Testing last solution...');
pred_pa2_last = model_predict(x_te,model_pa2,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_pa2_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_pa2_av = model_predict(x_te,model_pa2,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_pa2_av~=y_te))/numel(y_te)*100);
toc();


%% PROJ++ eta=0.1
tic();
% train Projectron++
fprintf('Training Projectron++ model...\n');
model_projectron2 = k_projectron2_multi_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',size(model_projectron2.beta, 2));
fprintf('Number of support vectors averaged solution:%d\n',size(model_projectron2.beta2, 2));
fprintf('Testing last solution...');
pred_projectron2_last = model_predict(x_te,model_projectron2,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n',numel(find(pred_projectron2_last~=y_te))/numel(y_te)*100);
fprintf('Testing averaged solution...');
pred_projectron2_av = model_predict(x_te,model_projectron2,1);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_projectron2_av~=y_te))/numel(y_te)*100);
toc();

%% RBP
tic();
% set maximum number of support vectors for RBP
model_bak.maxSV = size(model_projectron2.beta,2);
% train RBP
fprintf('Training RBP...\n');
model_rbp = k_perceptron_multi_train(x_tr,y_tr,model_bak);
fprintf('Done!\n');
fprintf('Number of support vectors last solution:%d\n',size(model_rbp.beta, 2));
fprintf('Testing last solution...');
pred_rbp_last = model_predict(x_te,model_rbp,0);
fprintf('Done!\n');
fprintf('%5.2f%% of errors on the test set.\n\n',numel(find(pred_rbp_last~=y_te))/numel(y_te)*100);
toc();

% tic();
% fprintf('Plotting error and SV curves...\n');
% % plot error curves
% figure(1)
% plot(model_pa1.aer(100:end),'k')
% hold on
% plot(model_perceptron.aer(100:end),'c')
% plot(model_projectron.aer(100:end),'b')
% plot(model_projectron2.aer(100:end),'m')
% plot(model_rbp.aer(100:end),'r')
% plot(model_forgetron.aer(100:end),'y')
% plot(model_oisvm.aer(100:end),'g')
% grid
% legend('PA-I','Perceptron','Projectron','Projectron++','RBP','Forgetron','OISVM')
% xlabel('Number of Samples')
% ylabel('Average Online Error')

% % plot support vector curves
% figure(2)
% plot(model_pa1.numSV,'k')
% hold on
% plot(model_perceptron.numSV,'c')
% plot(model_projectron.numSV,'b')
% plot(model_projectron2.numSV,'m')
% plot(model_rbp.numSV,'r')
% plot(model_forgetron.numSV,'y')
% plot(model_oisvm.numSV,'g')
% legend('PA-I','Perceptron','Projectron','Projectron++','RBP','Forgetron','OISVM','Location','NorthWest')
% grid
% xlabel('Number of Samples')
% ylabel('Number of Support Vectors')
% toc();
