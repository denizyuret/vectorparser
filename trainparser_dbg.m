function model = trainparser_dbg(model, sentences, heads, featidx)
for si=1:length(sentences)
  s = sentences{si};
  h = heads{si};
  p = ArcHybrid(s);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    cost = p.oracle_cost(h);
    assert(any(isfinite(cost)));
    [mincost, bestmove] = min(cost);
    feat = p.features();
    feat = feat(featidx);
    model = k_perceptron_multi_train(feat, bestmove, model);
    % scores = model.pred(:,model.iter);
    % scores(cost == inf) = -inf;
    % [~, move] = max(scores);
    p.transition(bestmove);
  end
end
end
