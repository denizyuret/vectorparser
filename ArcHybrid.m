% ArcHybrid.m, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
% http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
% Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.

classdef ArcHybrid < handle

properties (Constant = true)
SHIFT = 1;
RIGHT = 2;
LEFT = 3;
nmoves = 3;
end

properties
nwords  % number of words in sentence
words   % dxn matrix of word vectors
heads   % 1xn vector of heads
stack   % 1xn vector for stack of indices
ldeps   % nxn matrix for left dependents
rdeps   % nxn matrix for right dependents
wptr    % index of first word in buffer
sptr    % index of top of stack
lptr    % lptr(h): number of left deps for h
rptr    % rptr(h): number of right deps for h
end % properties

methods
function s = ArcHybrid(words)
n = size(words, 2);
s.nwords = n;
s.words = words;       s.wptr = 1;
s.stack = zeros(1, n); s.sptr = 0;
s.ldeps = zeros(n, n); s.lptr = zeros(1, n);
s.rdeps = zeros(n, n); s.rptr = zeros(1, n);
s.heads = zeros(1, n);
s.transition(s.SHIFT);
end % ArcHybrid

function add_arc(s, h, d)
s.heads(d) = h;
if d < h
  s.lptr(h) = s.lptr(h) + 1;
  s.ldeps(h, s.lptr(h)) = d;
else
  s.rptr(h) = s.rptr(h) + 1;
  s.rdeps(h, s.rptr(h)) = d;
end % if
end % add_arc

function transition(s, op)
switch op
 case s.SHIFT
  s.sptr = s.sptr + 1;
  s.stack(s.sptr) = s.wptr;
  s.wptr = s.wptr + 1;
 case s.RIGHT
  s.add_arc(s.stack(s.sptr - 1), s.stack(s.sptr));
  s.sptr = s.sptr - 1;
 case s.LEFT
  s.add_arc(s.wptr, s.stack(s.sptr));
  s.sptr = s.sptr - 1;
 otherwise
  error('Move %d is not supported.\n', op);
end % switch
end % transition

function v = valid_moves(s)
v(s.SHIFT) = (s.wptr <= s.nwords);
v(s.RIGHT) = (s.sptr >= 2);
v(s.LEFT)  = (s.sptr >= 1);
end % valid_moves

% Oracle counts gold arcs that become impossible after moves:
% 1. SHIFT moves b0 to s: (s\s0,b0) + (b0,s)
% 2. RIGHT adds (s1,s0) : (s0,b) + (b,s0)
% 3. LEFT  adds (b0,s0) : (s0,b) + (b\b0,s0) + (s1,s0)

function c = oracle_cost(s, gold)
c = Inf(1, s.nmoves);
if (s.wptr <= s.nwords)
  c(s.SHIFT) = sum(gold(s.stack(1:s.sptr)) == s.wptr) ...
      + sum(s.stack(1:s.sptr-1) == gold(s.wptr));
end
if (s.sptr >= 1)
  s0 = s.stack(s.sptr);
  s0h = gold(s0);
  s0b = sum(gold(s.wptr:end) == s0);
  if (s.wptr <= s.nwords)
    c(s.LEFT) = s0b + (s0h > s.wptr);
  end
  if (s.sptr >= 2) 
    c(s.RIGHT) = s0b + (s0h >= s.wptr);
    c(s.LEFT) = c(s.LEFT) + (s0h == s.stack(s.sptr-1));
  end
end
end % oracle_cost


function f = features(s)
d = size(s.words, 1);                   % word vector dimensionality
dx = d + 10;                            % bit features: l0, l1, l2, l3+, r0, r1, r2, r3+, exists, not
f = zeros(16 * dx + 8, 1);              % feature vector
i = 0;                                  % index into feature vector

function addWord(wi)  % nested function
% i+1 .. i+d contains the word vector
% i+d+1 .. i+d+4 are bits for left child count l0, l1, l2, l3+
% i+d+5 .. i+d+8 are bits for right child count r0, r1, r2, r3+
% i+d+9,i+d+10 is set to 1 if word exists / does not exist (wi==0)
if wi == 0
  f(i+d+10) = 1;
else
  f(i+d+9) = 1;
  f(i+1:i+d) = s.words(:,wi);
  l = s.lptr(wi); if (l > 3) l = 3; end;
  f(i+d+1+l) = 1;
  r = s.rptr(wi); if (r > 3) r = 3; end;
  f(i+d+5+r) = 1;
end % if
i = i + dx;
end % addWord

function addChildren(wi,dir,num)  % nested function
if wi == 0
  i = i + dx * num; % no features set for deps of nonexistent words
elseif dir == s.LEFT
  nc = s.lptr(wi);
  for c=1:num
    if (nc >= c) addWord(s.ldeps(wi,nc-c+1)); else addWord(0); end
  end % for
elseif dir == s.RIGHT
  nc = s.rptr(wi);
  for c=1:num
    if (nc >= c) addWord(s.rdeps(wi,nc-c+1)); else addWord(0); end
  end % for
end % if
end % addChildren

% The first three words of the buffer (n0, n1, n2)
if (s.wptr <= s.nwords)   n0 = s.wptr;   else n0 = 0; end
if (s.wptr+1 <= s.nwords) n1 = s.wptr+1; else n1 = 0; end
if (s.wptr+2 <= s.nwords) n2 = s.wptr+2; else n2 = 0; end
addWord(n0); addWord(n1); addWord(n2);

% The top three words of the stack (s0, s1, s2)
if (s.sptr >= 1) s0 = s.stack(s.sptr);   else s0 = 0; end
if (s.sptr >= 2) s1 = s.stack(s.sptr-1); else s1 = 0; end
if (s.sptr >= 3) s2 = s.stack(s.sptr-2); else s2 = 0; end
addWord(s0); addWord(s1); addWord(s2);

% The two leftmost children of s0 (s0l1, s0l2);
% The two rightmost children of s0 (s0r1, s0r2);
addChildren(s0, s.LEFT, 2);
addChildren(s0, s.RIGHT, 2);

% The two leftmost children of s1 (s1l1, s1l2);
% The two rightmost children of s1 (s1r1, s1r2);
addChildren(s1, s.LEFT, 2);
addChildren(s1, s.RIGHT, 2);

% The two leftmost children of n0 (n0l1, n0l2)
addChildren(n0, s.LEFT, 2);

% n0-s0 distance: 1,2,3,4+
if (s0 > 0 && n0 > 0)
  z = n0 - s0; if (z > 4) z = 4; end
  f(i+z) = 1;
end
i = i + 4;

% s0-s1 distance: 1,2,3,4+
if (s0 > 0 && s1 > 0)
  z = s0 - s1; if (z > 4) z = 4; end
  f(i+z) = 1;
end
i = i + 4;
assert(i == numel(f));
end % features

end % methods
end % classdef
