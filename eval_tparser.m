function r = eval_tparser(m, corpus)

  r = struct();
  r.move_cnt = numel(m.move);
  mincost = min(m.cost);
  movecost = m.cost(sub2ind(size(m.cost), m.move, 1:numel(m.move)));
  r.move_err = sum(movecost > mincost);
  r.move_pct = r.move_err / r.move_cnt;

  r.sent_cnt = 0; r.sent_err = 0;
  r.head_cnt = 0; r.head_err = 0;
  r.word_cnt = 0; r.word_err = 0;

  nmove = 0;

  for i=1:numel(corpus)
    s = corpus{i};
    h = s.head;
    p = m.head{i};
    nword = numel(h);
    head_err = sum(h ~= p);

    r.sent_cnt = r.sent_cnt + 1;
    if (head_err > 0) r.sent_err = r.sent_err + 1; end

    r.head_cnt = r.head_cnt + nword;
    r.head_err = r.head_err + head_err;

    if (i == 1) move1=1; else move1=m.sidx(i-1)+1; end
    move2 = m.sidx(i);
    sumcost = sum(movecost(move1:move2));
    assert(sumcost == head_err, ...
           'Discrepancy in sentence %d: %d ~= %d moves(%d,%d)', ...
           i, movecost, head_err, move1, move2);

    for j=1:numel(h)
      if (isempty(regexp(s.form{j}, '^\W+$')) ||...
          ~isempty(regexp(s.form{j}, '^[\`\$]+$')))
        r.word_cnt = r.word_cnt + 1;
        if h(j) ~= p(j)
          r.word_err = r.word_err + 1;
        end %if
      end %if
    end % for j=1:numel(h)
  end % for i=1:numel(corpus)
  assert(r.head_err == sum(movecost));

  r.sent_pct = r.sent_err / r.sent_cnt;
  r.head_pct = r.head_err / r.head_cnt;
  r.word_pct = r.word_err / r.word_cnt;

end % eval_tparser


