function e = eval_model(m, tst, tstdump, epoch)
[~,mtst] = vectorparser(m, tst, 'update', 0);
e = eval_conll(tst, mtst);
[~,s] = perceptron(tstdump.x, tstdump.y, m, 'update', 0);
[~,z] = max(s);
e.stat_tot = numel(z);
e.stat_err = numel(find(z ~= tstdump.y));
e.stat_pct = e.stat_err / e.stat_tot;
if nargin >= 4 fprintf('epoch\t'); end
fprintf('stat\tmove\thead\tword\tsent\n');
if nargin >= 4 fprintf('%d\t', epoch); end
fprintf('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', e.stat_pct, ...
        e.move_pct, e.head_pct, e.word_pct, e.sent_pct);
end % eval_model
