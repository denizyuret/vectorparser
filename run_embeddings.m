path('dogma', path);
fv804 = [
%n0 s0 s1 n1 n0l1 s0r1 s0l1 s1r1 s0r
  0 -1 -2  1  0   -1   -1   -2   -1;
  0  0  0  0 -1    1   -1    1    0;
  0  0  0  0  0    0    0    0    2;
];
hp = struct('type', 'poly', 'gamma', 1, 'coef0', 1, 'degree', 3);
m0=model_init(@compute_kernel,hp);m0.batchsize=1250; m0.step=8;

root = 'embedded';
em = dir(root);
for i=1:numel(em)
  emi = em(i).name;                     % e.g. conllToken_rcv1UNK
  if strcmp(emi(1),'.') continue; end
  if any(strcmp(emi, {
'conllWSJToken_bansal100',
'conllWSJToken_wikipedia2MUNK-200',
'conllWSJToken_wikipedia2MUNK-100',
'conllToken_wikipedia2MUNK-50',
'conllToken_wikipedia2MUNK-25',
'conll_bansal100',
'conllWSJToken_rcv1UNK50',
'conllToken_rcv1UNK',
'conllWSJToken_rcv1UNK',
'conllWSJToken_rcv1UNK25',
'conllWSJToken_dep-100',
'conll_cw100scaled',
'conllWSJToken_huang-0.1',
'conll_4scode+f50',
'conll_2scode+f100',
'conll_3scode+f100',
'conll_6scode+f50',
'conll_4scode+f100',
'conll_6scode+f100',
'conll_5scode+f100',
'conllWSJToken_dep-WSJ',
'conll_10scode+f50',
'conll_7scode+f50',
'conll_2scode+f50',
'conll_1scode+f100',
'conll_1scode+f50',
'conll_9scode+f50',
'conll_3scode+f50',
'conll_5scode+f50',
'conll_8scode+f50',
'conll_3scode+f25',
'conllWSJToken_cw50scaled',
'conll_cw50scaled',
'conllWSJToken_ungar2-0.1',
'conllWSJToken_hlbl50scaled',
'conll_cw',
'conllWSJToken_mikolovWiki-0.1',
'conll_dep-100',
'conll_dep-WSJ',
'conll_cw25+cw25',
'conll_cw25',
'conllWSJToken_cw25',
'conllWSJToken_stratos50k5000scaled01',
'conllWSJToken_gsng1M',
'conllWSJToken_mikolovRCV50scaled01',
'conllWSJToken_cw25scaled',
'conll_cw25scaled',
'conllWSJToken_gsngAE',
'conllWSJToken_gsngALL',
'conll_dep100+wiki25',
'conllWSJToken_yogamata52scaled01',
'conllWSJToken_yogamata52',
'conllWSJToken_stratos50k200scaled01',
'conll_german+german',
'conllWSJToken_manaal48',
'conllWSJToken_hpca50scaled01',
'conllWSJToken_manaal48scaled01',
'conll_german',
'conllWSJToken_rnn80',
'conll_dep100+mik',
'conllWSJToken_levy300',
'conllWSJToken_mikolovGoogleNews300',
'conllWSJToken_murphy50',
'conllWSJToken_murphy50scaled01',
'conllWSJToken_wikipedia2MUNK-25',
'conll_10scode+f50',
'conll_german2',

                     }))
    fprintf('Skipping %s\n', emi);
    continue; 
  end

  fprintf('Reading %s\n', emi);
  tic;
  trn_i = {};
  dev_i = {};
  if isdir([root '/' emi '/22'])
    for j=2:21                          % read train
      dir_j = sprintf('%s/%s/%02d', root, emi, j); % e.g. embedded/conllToken_rcv1UNK/02
      fprintf('Reading %s\n', dir_j);
      dp = dir(dir_j);  % e.g. dir('embedded/conllToken_rcv1UNK/02')
      for k=1:numel(dp);
        if dp(k).isdir continue; end
        dpk = dp(k).name;               % e.g. wsj_0201.dp
        if (~strcmp(dpk(end-2:end), '.dp')) continue; end
        trn_k = loadCoNLL([dir_j '/' dpk]); % e.g. embedded/conllToken_rcv1UNK/02/wsj_0201.dp
        trn_i = [trn_i trn_k];
      end % for k=1:numel(dp);
    end % for j=2:21
    dir_j = sprintf('%s/%s/%02d', root, emi, 22);
    fprintf('Reading %s\n', dir_j);
    dp = dir(dir_j);    % e.g. dir('embedded/conllToken_rcv1UNK/22')
    for k=1:numel(dp);
      if dp(k).isdir continue; end
      dpk = dp(k).name;                 % e.g. wsj_2201.dp
      if (~strcmp(dpk(end-2:end), '.dp')) continue; end
      dev_k = loadCoNLL([dir_j '/' dpk]);
      dev_i = [dev_i dev_k];
    end % for k=1:numel(dp)

  elseif isdir([root '/' emi '/01'])
    dir_j = sprintf('%s/%s/%02d', root, emi, 0);
    fprintf('Reading %s\n', dir_j);
    dp = dir(dir_j);
    assert(numel(dp) == 3, dir_j);
    dp3 = dp(3).name;
    assert(strcmp(dp3(end-2:end), '.dp'), dir_j);
    trn_i = loadCoNLL([dir_j '/' dp3]);

    dir_j = sprintf('%s/%s/%02d', root, emi, 1);
    fprintf('Reading %s\n', dir_j);
    dp = dir(dir_j);
    assert(numel(dp) == 3, dir_j);
    dp3 = dp(3).name;
    assert(strcmp(dp3(end-2:end), '.dp'), dir_j);
    dev_i = loadCoNLL([dir_j '/' dp3]);

  else
    fprintf('%s is empty\n', emi);
    continue;
  end %if
  toc;
  fprintf('dumpfeatures...\n');
  tic; [trn_x, trn_y] = dumpfeatures2(trn_i, fv804); toc;
  tic; [dev_x, dev_y] = dumpfeatures2(dev_i, fv804); toc;
  fprintf('k_perceptron_multi_train_gpu...\n');
  gpuDevice(1);
  tic;m1=k_perceptron_multi_train_gpu(trn_x,trn_y,m0); toc;
  tic;[a,b]=model_predict_gpu(dev_x, m1, 1); toc;
  score = numel(find(a ~= dev_y))/numel(dev_y);
  fprintf('score:%g\t%s\n', score, emi);
end % for i=1:numel(em)
