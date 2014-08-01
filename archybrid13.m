% archybrid13.m, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
% http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
% Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.

classdef archybrid13 < handle

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

% [w1,...,wn] are the words in the sentence
% [n0,n1,...] are the words in the buffer 
% [...,s1,s0] are the words in the stack.
%
% In (Goldberg and Nivre, 2013) the initial configuration has an empty
% stack, and a buffer with special symbol ROOT to the left of all the
% words.  We don't use an explicit ROOT, instead just use head=0 with
% the RIGHT move to get the same effect.  Also, when w1 is at
% position n0, SHIFT is the only legal move and we perform it
% during initialization.  So our initial state is stack=[w1],
% buffer=[w2...wn].

function p = archybrid13(n)
p.nword = n;	       p.wptr = 1;
p.stack = zeros(1, n); p.sptr = 0;
p.ldep = zeros(n, n);  p.lcnt = zeros(1, n);
p.rdep = zeros(n, n);  p.rcnt = zeros(1, n);
p.head = zeros(1, n);
p.transition(p.SHIFT);  % SHIFT is the only legal first move
end % archybrid13


% transition:
%
% SHIFT moves n0 to s0
% RIGHT adds (s1,s0) and pops s0 (pop with head=0 if no s1)
% LEFT adds (n0,s0) and pops s0

function transition(p, op)
switch op
 case p.SHIFT
  p.sptr = p.sptr + 1;
  p.stack(p.sptr) = p.wptr;
  p.wptr = p.wptr + 1;
 case p.RIGHT
  if (p.sptr > 1)
    p.add_arc(p.stack(p.sptr - 1), p.stack(p.sptr));
  end  % otherwise we already have head==0
  p.sptr = p.sptr - 1;
 case p.LEFT
  p.add_arc(p.wptr, p.stack(p.sptr));
  p.sptr = p.sptr - 1;
 otherwise
  error('Move %d is not supported.\n', op);
end % switch
end % transition


% valid_moves: 
%
% We need to have n0 for shift and s0 for right, both for left.
% No need for the last RIGHT move.

function v = valid_moves(p)
v(p.SHIFT) = (p.wptr <= p.nword);
v(p.RIGHT) = ((p.sptr > 1) || ((p.sptr == 1) && (p.wptr <= p.nword)));
v(p.LEFT)  = ((p.sptr >= 1) && (p.wptr <= p.nword));
end % valid_moves


% oracle_cost counts gold arcs that become impossible after possible
% transitions.  Tokens start their lifecycle in the buffer without
% links.  They move to the top of the buffer (n0) with SHIFT moves.
% There they acquire left dependents using LEFT moves.  After that a
% single SHIFT moves them to the top of the stack (s0).  There they
% acquire right dependents with SHIFT-RIGHT pairs.  Finally from s0
% they acquire a head with a LEFT or RIGHT move.  Any token from the
% buffer may become the head but only s1 from the stack (or root=0 if
% there is no s1) may become a left head.  The parser terminates when
% there is a single word at s0 whose head is ROOT (2n-2 moves).
%
% 1. SHIFT moves n0 to s0: n0 cannot acquire left dependents after a
% shift.  Also it can no longer get a head from the stack to the left
% of s0 or get a root head if there is s0: (0+s\s0,n0) + (n0,s)
%
% 2. RIGHT adds (s1,s0): s0 cannot acquire a head or dependent from
% the buffer after right: (s0,b) + (b,s0)
%
% 3. LEFT adds (n0,s0): s0 cannot acquire s1 or 0 (if there is no s1)
% or ni (i>0) as head.  It also cannot acquire any more right
% children: (s0,b) + (b\n0,s0) + (s1 or 0,s0)

function c = oracle_cost(p, gold)

assert(numel(gold) == p.nword);
c = Inf(1, p.NMOVE);
isvalid = valid_moves(p);
n0 = p.wptr;

if isvalid(p.SHIFT)
  n0h = gold(n0);
  c(p.SHIFT) = sum(gold(p.stack(1:p.sptr)) == n0) + ...
      + sum(p.stack(1:p.sptr-1) == n0h) ...
      + ((n0h == 0) && (p.sptr >= 1));
end

if p.sptr >= 1
  s0 = p.stack(p.sptr);
  s0h = gold(s0);
  s0b = sum(gold(n0:end) == s0);

  if isvalid(p.RIGHT)
    c(p.RIGHT) = s0b + (s0h >= n0);
  end

  if isvalid(p.LEFT)
    c(p.LEFT) = s0b + ...
        ((s0h > n0) || ...
         ((p.sptr == 1) && (s0h == 0)) || ...
         ((p.sptr >  1) && (s0h == p.stack(p.sptr-1))));
  end
end

end % oracle_cost


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


end % methods
end % classdef
