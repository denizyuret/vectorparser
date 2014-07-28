% Example input: /ai/home/vcirik/eParse/run/embedded/conllWSJToken_cw25/01/wsj_0101.dp

% 11 tab separated columns: 1.id, 2.form, 3.lemma, 4.cpostag, 5.postag,
% 6.feats, 7.head, 8.deprel, 9.phead, 10.pdeprel, 11.wvec
% head=0 represents root.  words start from id=1.

% TODO: Use single precision for word vecs to save space
% TODO: Longest sentence is 249 words but we'll use uint16 for heads just
% in case.  

function c = loadCoNLL(filename)
c = {};  % a corpus is a cell array of sentences
s = struct();  % a sentence is a struct array
fid = fopen(filename, 'r');
line = fgetl(fid);
while ischar(line)
  if length(line) == 0
    c{end+1} = s;
    s = struct();
    i = 0;
  else
    [tok,pos] = textscan(line, '%s', 10);
    tok = tok{1};
    i = str2num(tok{1});
    if ~strcmp(tok{2},'_') s.form{i} = tok{2}; end
    if ~strcmp(tok{3},'_') s.lemma{i} = tok{3}; end
    if ~strcmp(tok{4},'_') s.cpostag{i} = tok{4}; end
    if ~strcmp(tok{5},'_') s.postag{i} = tok{5}; end
    if ~strcmp(tok{6},'_') s.feats{i} = tok{6}; end
    if ~strcmp(tok{7},'_') s.head(i) = str2num(tok{7}); end
    if ~strcmp(tok{8},'_') s.deprel{i} = tok{8}; end
    if ~strcmp(tok{9},'_') s.phead{i} = tok{9}; end
    if ~strcmp(tok{10},'_') s.pdeprel{i} = tok{10}; end
    s.wvec(:,i) = sscanf(line(pos+1:end), '%f');
  end
  line = fgetl(fid);
end
if (i > 0) c{end+1} = s; end  % in case final newline missing
fclose(fid);
end
