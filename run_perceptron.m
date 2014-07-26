% path('dogma',path);
function [last,avg,nsv,time,model] = run_perceptron(x_tr, y_tr, x_te, y_te, kernel)
model = model_init(@compute_kernel, kernel);
model.step = 1000000;
tic();
model = k_perceptron_multi_train(x_tr,y_tr,model);
pred_perceptron_last = model_predict(x_te,model,0);
pred_perceptron_av = model_predict(x_te,model,1);
time = toc();
last = numel(find(pred_perceptron_last~= y_te))/numel(y_te)*100;
avg = numel(find(pred_perceptron_av~=y_te))/numel(y_te)*100;
nsv = size(model.beta, 2);
end
