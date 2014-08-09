fv084 = [];
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
      fv084 = [fv084; [a b c]];
    end
  end
end


% Better initial starting point
fv008w = [
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
fv008 = [
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
fv136 = [];
for a=-3:2
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for buffer words other than n0
    for c=-9:9
      if ((c == 0) || (c == -3)) continue; end  % do not use the word+context combination, keep them separate
      if ((a > 0) && ~ismember(c, [1,-1,4,-4])) continue; end      % no deps/dist/in-between/head for a>0
      if ((a == 0) && (b == 0) && ~ismember(c, [1,-1,4,-4,-2,-5,-6])) continue; end  % no rdeps/dist/in-between/head for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8,9,-9])) continue; end  % no dist/in-between/head for deps
      fv136 = [fv136; [a b c]];
    end
  end
end

% All legal features for s2..n2:
fv130 = [];
for a=-3:2
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for a>0
    for c=-8:8
      if ((a > 0) && ismember(c, [2,-2,3,-3,5,-5,6,-6,7,-7,8,-8])) continue; end      % no deps/dist/in-between for a>0
      if ((a == 0) && (b == 0) && ismember(c, [2,3,-3,5,6,7,-7,8,-8])) continue; end  % no rdeps/dist/in-between for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8])) continue; end  % no dist/in-between for deps
      if ((c == 0) || (c == -3)) continue; end  % do not use the word+context combination, keep them separate
      fv130 = [fv130; [a b c]];
    end
  end
end

fv102 = [];
for a=-2:1
  for b=-1:1
    if ((a >= 0) && (b > 0)) continue; end   % no rdeps for buffer words
    if ((a > 0)  && (b < 0)) continue; end   % no ldeps for a>0
    for c=-8:8
      if ((a > 0) && ismember(c, [2,-2,3,-3,5,-5,6,-6,7,-7,8,-8])) continue; end      % no deps/dist/in-between for a>0
      if ((a == 0) && (b == 0) && ismember(c, [2,3,-3,5,6,7,-7,8,-8])) continue; end  % no rdeps/dist/in-between for a=0
      if ((b ~= 0) && ismember(c, [3,-3,7,-7,8,-8])) continue; end  % no dist/in-between for deps
      fv102 = [fv102; [a b c]];
    end
  end
end

fv1768 = [
%n0            n1             n2             s0             s1             s2             s0l1           s0l2           s0r1           s0r2           s1l1           s1l2           s1r1           s1r2           n0l1           n0l2          n0s0 s0s1
0  0  0  0  0  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3 -3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2  0  0  0  0  0  0  0  0  0  0 -1   -2;
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  1  1  1  1  1  2  2  2  2  2 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2  0    0;
0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  0 -2  2  1 -1  3    3;
]';

fv804 = [
%n0 s0 s1 n1 n0l1 s0r1 s0l1 s1r1 s0r
  0 -1 -2  1  0   -1   -1   -2   -1;
  0  0  0  0 -1    1   -1    1    0;
  0  0  0  0  0    0    0    0    2;
]';

fv708 = [
%n0 n0l1 n0l2 n1 s0 s0r s0r1 s0s1 s1
  0  0    0    1 -1 -1  -1   -2   -2;
  0 -1   -2    0  0  0   1    0    0;
  0  0    0    0  0  2   0    3    0;
]';

fv808 = [
%n0 s0 s1 n1 n0l1 s0r1 s1r1l s0l1 s0r1l s2
  0 -1 -2  1  0   -1   -2    -1   -1    -3;
  0  0  0  0 -1    1    1    -1    1     0;
  0  0  0  0  0    0   -2     0   -2     0;
]';
