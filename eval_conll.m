function r = eval_conll(corpus, dump)

% sum(dump.y ~= dump.z) here gives a different result
% but moves of equal cost with gold move should be counted as correct
% should confirm total zcost == total head error

r.move_tot = numel(dump.y);
ycost = dump.cost(sub2ind(size(dump.cost), dump.y, 1:numel(dump.y)));
zcost = dump.cost(sub2ind(size(dump.cost), dump.z, 1:numel(dump.z)));
r.move_err = sum(ycost < zcost);
r.move_pct = r.move_err / r.move_tot;

r.sent_cnt = 0; r.sent_err = 0;
r.head_cnt = 0; r.head_err = 0;
r.word_cnt = 0; r.word_err = 0;

nmove = 0;

for i=1:numel(corpus)
  s = corpus{i};
  h = s.head;
  p = dump.pred{i};
  nword = numel(h);
  head_err = sum(h ~= p);

  r.sent_cnt = r.sent_cnt + 1;
  if (head_err > 0) r.sent_err = r.sent_err + 1; end

  r.head_cnt = r.head_cnt + nword;
  r.head_err = r.head_err + head_err;

  % Every sentence has 2*nword-2 moves
  move1 = nmove + 1;
  move2 = nmove + 2*nword - 2;
  nmove = move2;
  movecost = sum(zcost(move1:move2));
  assert(movecost == head_err, ...
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
  end %for

end
assert(nmove == numel(dump.z));
assert(r.head_err == sum(zcost));

r.sent_pct = r.sent_err / r.sent_cnt;
r.head_pct = r.head_err / r.head_cnt;
r.word_pct = r.word_err / r.word_cnt;
end
