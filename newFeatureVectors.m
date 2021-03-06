function f = newFeatureVectors()

% 0.0418668 archybrid_conll07EnglishToken_wikipedia2MUNK-100_rbf376678_1014_cache.mat (trn/tst) (rbf-gamma:0.376747095368119)
f.fv022b = [-3 0 8;-2 -1 -4;-2 -1 5;-2 0 -9;-2 0 -4;-2 0 4;-2 0 7;-2 1 1;-1 -1 -1;-1 -1 4;-1 0 -9;-1 0 -4;-1 0 4;-1 0 6;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4];

% 0.0325506937033084 archybrid_conllWSJToken_wikipedia2MUNK-100_rbf333140_cache.mat (rbf-gamma:0.333140404105682)
f.fv021a = [-3 0 4;-3 1 -2;-2 0 4;-2 0 6;-2 1 -2;-1 -1 -4;-1 -1 2;-1 -1 4;-1 0 -4;-1 0 -1;-1 0 2;-1 0 4;-1 1 2;-1 1 4;0 -1 -4;0 -1 -1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4];

% 0.0443168273919168 archybrid_conll07EnglishToken_wikipedia2MUNK-100_rbf372064_cache.mat (rbf-gamma:0.372063662109375)
f.fv019 = [-3 0 8;-2 -1 -4;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 6;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4];

% 0.03492252681764 archybrid_conllWSJToken_wikipedia2MUNK-100_rbf339506_cache.mat (rbf-gamma:0.339506144311523)
f.fv022a = [-3 0 4;-3 1 -2;-2 0 -4;-2 0 4;-2 0 6;-2 1 -2;-1 -1 -4;-1 -1 2;-1 -1 4;-1 0 -4;-1 0 -1;-1 0 2;-1 0 4;-1 1 2;-1 1 4;0 -1 -4;0 -1 -1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4];

% 0.0444685231336006 archybrid13_conll07EnglishToken_wikipedia2MUNK-100_rbf3.720637e+05_cache.mat
f.fv023a = [-3 0 8;-2 -1 -4;-2 -1 -1;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-2 1 4;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 5;-1 0 6;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4];

% BUGGY: 0.0325637 Best with rbf=0.339506144311523, archybrid_conllWSJToken_wikipedia2MUNK-100_fv130_dump.mat.
f.fv023 = [-3 0 4;-3 1 -2;-2 -1 -4;-2 0 -4;-2 0 4;-2 0 6;-2 1 -2;-1 -1 -4;-1 -1 2;-1 -1 4;-1 0 -4;-1 0 -1;-1 0 2;-1 0 4;-1 1 2;-1 1 4;0 -1 -4;0 -1 -1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4];

% BUGGY: 0.0443493 Best with run_featselect_rbf('archybrid13', 'conll07EnglishToken_wikipedia2MUNK-100', 'fv136', 'fv021', 'rbf3721');
f.fv022 = [-3 0 8;-2 -1 -1;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-2 1 4;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 6;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 -1 5;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4];

% 0.0445990 Best with rbf372111 archybrid_conll07EnglishToken_wikipedia2MUNK-100_fv136_dump.mat
f.fv021 = [-3 0 8;-2 -1 -1;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-1 -1 -1;-1 -1 4;-1 0 -4;-1 0 4;-1 0 5;-1 0 6;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4];

% Best on run_featselect22('archybrid', 'conllWSJToken_wikipedia2MUNK-100', 'fv130', 'fv018');
f.fv017a = [-3 0 -4;-3 0 4;-2 0 -4;-2 0 4;-2 1 -2;-1 -1 -4;-1 -1 4;-1 0 -4;-1 0 4;-1 1 -2;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4];

% This one does best on archybrid_conll07EnglishToken_wikipedia2MUNK-100_d5 (degree=5 kernel)
f.fv018a = [-3 0 8;-2 0 -9;-2 0 -4;-2 0 4;-2 1 1;-1 -1 1;-1 -1 4;-1 0 -4;-1 0 4;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 -2;0 0 4;1 0 -4;1 0 4];

% This one does best on archybrid_conll07EnglishToken_wikipedia2MUNK-100 (degree=3 kernel)
f.fv015a = [-3 0 8;-2 0 -4;-2 0 4;-2 1 1;-1 -1 4;-1 0 -4;-1 0 4;-1 1 -4;-1 1 4;0 -1 -4;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4];

% This one is a local minimum on archybrid_conll07EnglishToken_wikipedia2MUNK-100 (degree=3 kernel)
f.fv031a = [-3 0 -8;-3 0 -4;-3 0 -1;-3 0 8;-3 1 -4;-3 1 4;-2 -1 -4;-2 -1 4;-2 0 -8;-2 0 -4;-2 0 4;-2 0 8;-2 1 1;-2 1 2;-1 -1 -4;-1 -1 1;-1 -1 4;-1 0 -8;-1 0 -4;-1 0 2;-1 0 4;-1 0 8;-1 1 -4;-1 1 4;0 -1 -4;0 -1 1;0 -1 4;0 0 -4;0 0 4;1 0 -4;1 0 4];

f.fv034 = [];
for a=-3:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for buffer words other than n0
    if ((a <= -2) && (b < 0)) continue; end  % not interested in ldeps beyond s0
    if ((a <= -3) && (b > 0)) continue; end  % not interested in rdeps beyond s1
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  % do not use the word+context combination, keep them separate
      if (abs(c) == 5 || abs(c) == 6) continue; end  % just use = encoding for child count
      if ((abs(c) == 2) && (b ~= 0)) continue; end % not interested in grand children
      if (abs(c) == 1) continue; end % existence not useful
      if (abs(c) == 9) continue; end % head not useful in archybrid
      if ((c == 3) || (c == -7)) continue; end  % just use >= encoding for distance
      if ((a > 0) && ~ismember(c, [1,-1,4,-4])) continue; end      % no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && ~ismember(c, [1,-1,4,-4,-2,-5,-6])) continue; end  % no rdeps/dist/in-between/head for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  % no dist/in-between/head for deps
      f.fv034 = [f.fv034; [a b c]];
    end
  end
end



f.fv039 = [
0 0 4; 		% [-196]    'n0w'       
-1 0 4; 	% [-119]    's0w'       
1 0 4; 		% [ -73]    'n1w'       
-2 0 4; 	% [ -43]    's1w'       
0 -1 4; 	% [ -20]    'n0l1w'     
0 0 -4; 	% [ -20]    'n0c'       
1 0 -4; 	% [ -19]    'n1c'       
-1 0 -4; 	% [ -16]    's0c'       
-1 -1 4; 	% [ -11]    's0l1w'     
-2 0 -4; 	% [ -10]    's1c'       
-3 -1 -4;     	% [  -8]    's2l1c'     
-1 -1 -4;     	% [  -8]    's0l1c'     
0 -1 -4; 	% [  -8]    'n0l1c'     
-3 -1 2; 	% [  -7]    's2l1r='    
-3 0 8; 	% [  -7]    's2aw'      
-2 1 1; 	% [  -7]    's1r1+'     
-1 0 -8; 	% [  -7]    's0ac'      
-1 0 7; 	% [  -7]    's0d>'      
-1 -1 1; 	% [  -6]    's0l1+'     
-1 0 8; 	% [  -6]    's0aw'      
-3 0 -4; 	% [  -5]    's2c'       
-3 0 4; 	% [  -5]    's2w'       
-2 -1 -4;     	% [  -5]    's1l1c'     
-1 1 4; 	% [  -5]    's0r1w'     
-3 0 -8; 	% [  -4]    's2ac'      
0 -1 1; 	% [  -4]    'n0l1+'     
-3 0 -1; 	% [  -3]    's2-'       
-3 0 7; 	% [  -3]    's2d>'      
-3 1 4; 	% [  -3]    's2r1w'     
-2 -1 4; 	% [  -3]    's1l1w'     
-2 0 8; 	% [  -3]    's1aw'      
-2 1 2; 	% [  -3]    's1r1r='    
-2 -1 -2;     	% [  -2]    's1l1l='    
-1 -1 -2;     	% [  -2]    's0l1l='    
-2 0 -8; 	% [  -1]    's1ac'      
-1 -1 2; 	% [  -1]    's0l1r='    
-1 1 -4; 	% [  -1]    's0r1c'     
-3 1 -4; 	% [   0]    's2r1c'     
-1 0 2; 	% [   0]    's0r='      
];


f.fv012 = [
0 0 4; 		% [-203]    n0w       
-1 0 4; 	% [-125]    s0w       
-2 0 4; 	% [ -59]    s1w       
1 0 4; 		% [ -59]    n1w       
0 -1 4; 	% [ -41]    n0l1w     
-1 -1 4;	% [ -21]    s0l1w     

0 0 -4; 	% [ -17]    n0c       
-1 0 -4;	% [ -16]    s0c       
-2 0 -4;	% [ -13]    s1c       
1 0 -4; 	% [ -15]    n1c       
0 -1 -4; 	% [ -11]    n0l1c     
-1 -1 -4;	% [ -12]    s0l1c     
];


f.fv017 = [       % 0.0476108, 'archybrid', 'conll07EnglishToken_wikipedia2MUNK-100'
0 0 4; 		% [-203]    n0w       
-1 0 4; 	% [-125]    s0w       
-2 0 4; 	% [ -59]    s1w       
1 0 4; 		% [ -59]    n1w       
0 -1 4; 	% [ -41]    n0l1w     
-2 1 -2;	% [ -30]    s1r1l=    
-1 -1 4;	% [ -21]    s0l1w     
-1 1 4; 	% [ -21]    s0r1w     
-1 1 -2;	% [ -19]    s0r1l=    
0 0 -4; 	% [ -17]    n0c       
-1 0 -4;	% [ -16]    s0c       
1 0 -4; 	% [ -15]    n1c       
-2 0 -4;	% [ -13]    s1c       
-3 0 4; 	% [ -12]    s2w       
-1 -1 -4;	% [ -12]    s0l1c     
-1 1 -4; 	% [ -12]    s0r1c     
0 -1 -4; 	% [ -11]    n0l1c     
];


f.fv018 = [       % another version of fv808 that splits token features to word + context
    0 0 4;
    0 0 -4;
    -1 0 4;
    -1 0 -4;
    -2 0 4;
    -2 0 -4;
    1 0 4;
    1 0 -4;
    0 -1 4;
    0 -1 -4;
    -1 1 4;
    -1 1 -4;
    -2 1 -2;
    -1 -1 4;
    -1 -1 -4;
    -1 1 -2;
    -3 0 4;
    -3 0 -4;
];


f.fv084 = [];
for a=-3:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for buffer words other than n0
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  % do not use the word+context combination, keep them separate
      if (abs(c) == 5 || abs(c) == 6) continue; end  % just use = encoding for child count
      if ((c == 3) || (c == -7)) continue; end  % just use >= encoding for distance
      if ((a > 0) && ~ismember(c, [1,-1,4,-4])) continue; end      % no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && ~ismember(c, [1,-1,4,-4,-2,-5,-6])) continue; end  % no rdeps/dist/in-between/head for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  % no dist/in-between/head for deps
      f.fv084 = [f.fv084; [a b c]];
    end
  end
end


% Better initial starting point
f.fv008w = [
    0  0 4; % n0w
    -1 0 4; % s0w
    1  0 4; % n1w
    -2 0 4; % s1w
    0 -1 4; % n0l1w
    -1 1 4; % s0r1w
    -2 1 4; % s1r1w
    -1 -1 4; % s0l1w
         ];

% Good initial starting point
f.fv008 = [
    0 0 4;      % n0
    -1 0 4;     % s0
    1 0 4;      % n1
    -2 0 4;     % s1
    0 -1 4;     % n0l1
    -1 1 4;     % s0r1
    0 0 -4;     % n0c
    -1 0 -4;	% s0c
];

% Include the head feature for stack words:
f.fv136 = [];
for a=-3:2
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for buffer words other than n0
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  % do not use the word+context combination, keep them separate
      if ((a > 0) && ~ismember(c, [1,-1,4,-4])) continue; end      % no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && ~ismember(c, [1,-1,4,-4,-2,-5,-6])) continue; end  % no rdeps/dist/in-between/head for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  % no dist/in-between/head for deps
      f.fv136 = [f.fv136; [a b c]];
    end
  end
end

% All legal features for s2..n2:
f.fv130 = [];
for a=-3:2
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for a>0
    for c=-8:8
      if ((a > 0) && ismember(c, [2,-2,3,-3,5,-5,6,-6,7,-7,8,-8])) continue; end      % no deps/dist/in-between for a>0
      if ((a == 0) && (b == 0) && ismember(c, [2,3,-3,5,6,7,-7,8,-8])) continue; end  % no rdeps/dist/in-between for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8])) continue; end  % no dist/in-between for deps
      if ((c == 0) || (c == -3)) continue; end  % do not use the word+context combination, keep them separate
      f.fv130 = [f.fv130; [a b c]];
    end
  end
end

f.fv102 = [];
for a=-2:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for a>0
    for c=-8:8
      if ((a > 0) && ismember(c, [2,-2,3,-3,5,-5,6,-6,7,-7,8,-8])) continue; end      % no deps/dist/in-between for a>0
      if ((a == 0) && (b == 0) && ismember(c, [2,3,-3,5,6,7,-7,8,-8])) continue; end  % no rdeps/dist/in-between for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8])) continue; end  % no dist/in-between for deps
      f.fv102 = [f.fv102; [a b c]];
    end
  end
end

f.fv1768 = [
%n0            n1             n2             s0             s1             s2             s0l1           s0l2           s0r1           s0r2           s1l1           s1l2           s1r1           s1r2           n0l1           n0l2          n0s0 s0s1
0  0  0  0  0  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3 -3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2  0  0  0  0  0  0  0  0  0  0 -1   -2;
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  0    0;
0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  3    3;
]';

f.fv804 = [
%n0 s0 s1 n1 n0l1 s0r1 s0l1 s1r1 s0r
  0 -1 -2  1  0   -1   -1   -2   -1;
  0  0  0  0 -1    1   -1    1    0;
  0  0  0  0  0    0    0    0    2;
]';

f.fv708 = [
%n0 n0l1 n0l2 n1 s0 s0r s0r1 s0s1 s1
  0  0    0    1 -1 -1  -1   -2   -2;
  0 -1   -2    0  0  0   1    0    0;
  0  0    0    0  0  2   0    3    0;
]';

f.fv808 = [
%n0 s0 s1 n1 n0l1 s0r1 s1r1l s0l1 s0r1l s2
  0 -1 -2  1  0   -1   -2    -1   -1    -3;
  0  0  0  0 -1    1    1    -1    1     0;
  0  0  0  0  0    0   -2     0   -2     0;
]';

end
