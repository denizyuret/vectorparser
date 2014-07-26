% ArcHybrid.m, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
% http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
% Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.

classdef archybrid < handle

properties (Constant = true)
NMOVE = 3;
SHIFT = 1;
RIGHT = 2;
LEFT = 3;
end

properties
nword   % number of words in sentence
head    % 1xn vector of heads
stack   % 1xn vector for stack of indices
ldep    % nxn matrix for left dependents
rdep    % nxn matrix for right dependents
wptr    % index of first word in buffer
sptr    % index of last word (top) of stack
lcnt    % lcnt(h): number of left deps for h
rcnt    % rcnt(h): number of right deps for h
end % properties

methods

function p = archybrid(n)
p.nword = n;	       p.wptr = 1;
p.stack = zeros(1, n); p.sptr = 0;
p.ldep = zeros(n, n); p.lcnt = zeros(1, n);
p.rdep = zeros(n, n); p.rcnt = zeros(1, n);
p.head = zeros(1, n);
p.transition(p.SHIFT);
end % ArcHybrid


function add_arc(p, h, d)
p.head(d) = h;
if d < h
  p.lcnt(h) = p.lcnt(h) + 1;
  p.ldep(h, p.lcnt(h)) = d;
else
  p.rcnt(h) = p.rcnt(h) + 1;
  p.rdep(h, p.rcnt(h)) = d;
end % if
end % add_arc


function transition(p, op)
switch op
 case p.SHIFT
  p.sptr = p.sptr + 1;
  p.stack(p.sptr) = p.wptr;
  p.wptr = p.wptr + 1;
 case p.RIGHT
  p.add_arc(p.stack(p.sptr - 1), p.stack(p.sptr));
  p.sptr = p.sptr - 1;
 case p.LEFT
  p.add_arc(p.wptr, p.stack(p.sptr));
  p.sptr = p.sptr - 1;
 otherwise
  error('Move %d is not supported.\n', op);
end % switch
end % transition


% Oracle counts gold arcs that become impossible after moves:
% 1. SHIFT moves n0 to s: (s\s0,n0) + (n0,s)
% 2. RIGHT adds (s1,s0) : (s0,b) + (b,s0)
% 3. LEFT  adds (n0,s0) : (s0,b) + (b\n0,s0) + (s1,s0)

function c = oracle_cost(p, gold)
assert(numel(gold) == p.nword);
c = Inf(1, p.NMOVE);
n0 = p.wptr;
if (n0 <= p.nword)
  c(p.SHIFT) = sum(gold(p.stack(1:p.sptr)) == n0) ...
      + sum(p.stack(1:p.sptr-1) == gold(n0));
end
if (p.sptr >= 1)
  s0 = p.stack(p.sptr);
  s0h = gold(s0);
  s0b = sum(gold(n0:end) == s0);
  if (n0 <= p.nword)
    c(p.LEFT) = s0b + (s0h > n0);
  end
  if (p.sptr >= 2) 
    c(p.RIGHT) = s0b + (s0h >= n0);
    c(p.LEFT) = c(p.LEFT) + (s0h == p.stack(p.sptr-1));
  end
end
end % oracle_cost


function v = valid_moves(p)
v(p.SHIFT) = (p.wptr <= p.nword);
v(p.RIGHT) = (p.sptr >= 2);
v(p.LEFT)  = (p.sptr >= 1);
end % valid_moves

end % methods
end % classdef
