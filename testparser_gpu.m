function r = testparser_gpu(model, sentences, heads, featidx, gpos)

r = struct('ntot',0,'nerr',0,'npct',0,... % token errors
           'xtot',0,'xerr',0,'xpct',0,... % transition errors
           'wtot',0,'werr',0,'wpct',0); % non-punct errors

assert(((isfield(model,'beta') || isfield(model,'beta2')) && ...
	(~isempty(model.ker)) && ...
	strcmp(model.kerparam.type, 'poly')), ...
       'Only poly kernel models supported.\n');

if nargin < 5 gpos=[]; end

tic(); fprintf('Loading model on GPU.\n');
gpuDevice(1);
if isfield(model,'X')
  svtr = gpuArray(model.X(:,model.S)');
else
  svtr = gpuArray(model.SV');
end
if (~isfield(model,'beta2') || isempty(model.beta2))
  beta = gpuArray(model.beta);
  betatr = beta';
  b = model.b;
else
  beta = gpuArray(model.beta2);
  betatr = beta';
  b = model.b2;
end

hp = model.kerparam;
r.t = zeros(1,20);

toc();tic();fprintf('Processing sentences...\n');
for si=1:length(sentences)
  s = sentences{si};
  h = heads{si};
  p = ArcHybrid(s);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    %ti=1;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    c = p.oracle_cost(h);               % 1019us
    %ti=2;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    assert(any(isfinite(c)));           % 915us
    %ti=3;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    f = p.features();                   % 1581us

    % bsxfun is better than skinny matrix multiply on the GPU:

    %ti=4;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    % [~,scores] = model_predict(f(featidx), model)
    % scores = gather(b + beta * (hp.gamma * (svtr * f(featidx)) + hp.coef0).^hp.degree); % 15207us

    scores = gather(b + sum(bsxfun(@times, betatr, (hp.gamma * (svtr * f(featidx)) + hp.coef0).^hp.degree))); % 7189us

%     %ti=10;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     ff=gpuArray(f(featidx));            % 1012us
%     ti=11;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     kx1=svtr * ff;                      % 5096us
%     ti=12;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     kx2=hp.gamma * kx1;                 % 1076us
%     ti=13;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     kx3=kx2 + hp.coef0;                 % 1068us
%     ti=14;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     kx4=kx3 .^ hp.degree;               % 1140us
%     ti=15;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     % kx5=beta * kx4;                     % 8749us
%     kx5=sum(bsxfun(@times, betatr, kx4)); % 1353us
%     ti=16;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     kx6=b + kx5;                        % 1063us
%     ti=17;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
%     scores = gather(kx6);               % 996us
%     ti=18;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();

    %ti=5;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    scores(c==inf) = -inf;              % 928us
    %ti=6;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    [~,move] = max(scores);             % 925us
    %ti=7;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    p.transition(move);                 % 1019us
    %ti=8;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    r.xtot = r.xtot + 1;                % 914us
    %ti=9;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
    if c(move) > min(c) r.xerr = r.xerr + 1; end; % 918us
    % ti=20;wait(gpuDevice);r.t(ti)=r.t(ti)+toc();
  end
  r.ntot = r.ntot + numel(h);
  r.nerr = r.nerr + numel(find(h ~= p.heads));
  if ~isempty(gpos)
    gi = gpos{si};                      % gold parts of speech of sentence i
    np = true(1, numel(gi));            % non-punctuation index
    for gj=1:numel(gi)
      g = gi{gj};                       % part of speech of word j
      if any(strcmp(g, {',', ':', '.', '``', ''''''}))
        np(gj) = false;
      end
    end
    r.nptot = r.nptot + numel(h(np));
    r.nperr = r.nperr + numel(find(h(np) ~= p.heads(np)));
  end
end
toc();
end
