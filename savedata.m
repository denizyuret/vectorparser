clear;
dir = '/ai/home/vcirik/eParse/run/embedded/conllWSJToken_wikipedia2MUNK-50/';
tic();
[trn_w, trn_h] = loadCoNLL([dir '00/wsj_0001.dp']); % 989860 lines, wsj 02-21
toc(); tic();
[dev_w, dev_h] = loadCoNLL([dir '01/wsj_0101.dp']); % 41817 lines, wsj 22
toc(); tic();
[tst_w, tst_h] = loadCoNLL([dir '02/wsj_0201.dp']); % 191297 lines, wsj 00, 01, 23, 24
toc(); tic();
save('conllWSJToken_wikipedia2MUNK-50');
toc();
