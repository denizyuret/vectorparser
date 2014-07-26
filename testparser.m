function r = testparser(model, sentences, heads, featidx)
r.ntot = 0;
r.nerr = 0;
r.xtot = 0;
r.xerr = 0;
for si=1:length(sentences)
  s = sentences{si};
  h = heads{si};
  p = ArcHybrid(s);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    c = p.oracle_cost(h);
    assert(any(isfinite(c)));
    f = p.features();
    [~,scores] = model_predict(f(featidx), model);
    scores(c==inf) = -inf;
    [~,move] = max(scores);
    p.transition(move);
    r.xtot = r.xtot + 1;
    if c(move) > min(c) r.xerr = r.xerr + 1; end;
  end
  r.ntot = r.ntot + numel(h);
  r.nerr = r.nerr + numel(find(h ~= p.heads));
end
end
