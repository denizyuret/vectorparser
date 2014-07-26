function r = testparser_par(model, sentences, heads, featidx)

fprintf('Checking matlabpool.\n');
if matlabpool('size') == 0
  matlabpool('open')
end

fprintf('Creating shared matrix.\n');
hp = model.kerparam;
svkey=101;
btkey=102;
try,sharedmatrix('free',svkey);catch,end;
try,sharedmatrix('free',btkey);catch,end;
if isfield(model,'X')
  sharedmatrix('clone', svkey, model.X(:,model.S));
else
  sharedmatrix('clone', svkey, model.SV);
end
if (~isfield(model,'beta2') || isempty(model.beta2))
  sharedmatrix('clone', btkey, model.beta);
  b = model.b;
else
  sharedmatrix('clone', btkey, model.beta2);
  b = model.b2;
end

ns = length(sentences);
ntot = zeros(1,ns);
nerr = zeros(1,ns);
xtot = zeros(1,ns);
xerr = zeros(1,ns);

fprintf('Starting parfor.\n');
parfor si=1:ns
  fprintf('Starting sentence %d\n', si);
  beta = sharedmatrix('attach', btkey);
  sv = sharedmatrix('attach', svkey);
  s = sentences{si};
  h = heads{si};
  p = ArcHybrid(s);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    c = p.oracle_cost(h);
    assert(any(isfinite(c)));
    f = p.features();
    scores = b + beta * (hp.gamma * (sv' * f(featidx)) + hp.coef0).^hp.degree;
    scores(c==inf) = -inf;
    [~,move] = max(scores);
    p.transition(move);
    xtot(si) = xtot(si) + 1;
    if c(move) > min(c) xerr(si) = xerr(si) + 1; end;
  end
  ntot(si) = ntot(si) + numel(h);
  nerr(si) = nerr(si) + numel(find(h ~= p.heads));
  sharedmatrix('detach', btkey, beta);
  sharedmatrix('detach', svkey, sv);
end

sharedmatrix('free',svkey);
sharedmatrix('free',btkey);

r = struct('ntot', sum(ntot), 'nerr', sum(nerr), 'xtot', sum(xtot), 'xerr', sum(xerr));

end
