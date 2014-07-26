function sv = perceptron(trn_w, trn_h)

% savedata();
% load('conllWSJToken_wikipedia2MUNK-50.mat');
w1 = trn_w{1};
p1 = ArcHybrid(w1);
f1 = p1.features();
nf = length(f1);

tic();
sv = cell(1, p1.nmoves);
nsv = zeros(1, p1.nmoves);
smax = 100000;
for i=1:p1.nmoves
  sv{i} = zeros(smax, nf);
end
toc();

tic();
for i=1:100
  w = trn_w{i};
  h = trn_h{i};
  p = ArcHybrid(w);
  while ((p.sptr > 1) || (p.wptr <= p.nwords))
    c = p.oracle_cost(h);
    assert(sum(isfinite(c)) > 0);
    f = p.features();
    s = scores(f);
    s(isinf(c)) = -Inf;
    [~, pred] = max(s);
    s(c > min(c)) = -Inf;
    [~, best] = max(s);
    if (pred ~= best) 
      update(best, pred, f); 
    end
    p.transition(pred);
  end
  disp(nsv);
end
toc();

function s = scores(f)
for i=1:p.nmoves
  s(i) = sum((sv{i}(1:nsv(i),:) * f + 1) .^ 4);
end
end  % scores

function update(best, pred, f)
nsv(best) = nsv(best) + 1;
sv{best}(nsv(best),:) = f;
nsv(pred) = nsv(pred) + 1;
sv{pred}(nsv(pred),:) = -f;
end  % update

end  % perceptron