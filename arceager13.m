% arceager13.m, Deniz Yuret, July 28, 2014: Transition based greedy arc-eager parser based on:
% http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
% Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.

classdef arceager13 < matlab.mixin.Copyable

properties (Constant = true)
NMOVE 	= 4;
SHIFT 	= 1;	% push stack
RIGHT 	= 2;	% right child
LEFT 	= 3;	% left child
REDUCE	= 4;	% pop stack
end

properties
nword   % number of words in sentence
stack   % 1xn vector for stack of indices
head    % 1xn vector of heads
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
% In the arc-eager system (Nivre, 2003), a configuration c= (σ,β,A)
% consists of a stack σ, a buffer β, and a set A of dependency arcs.
% In (Goldberg and Nivre, 2013) the initial configuration has an empty
% stack, and a buffer with special symbol ROOT to the right of all the
% words.  We don't use an explicit ROOT, instead just use head=0 with
% the LEFT move to get the same effect.  Also, when w1 is at position
% n0, SHIFT is the only legal move and we perform it during
% initialization.  So our initial state is stack=[w1],
% buffer=[w2...wn].

function p = arceager13(n)
p.nword = n;	       p.wptr = 1;
p.stack = zeros(1, n); p.sptr = 0;
p.ldep = zeros(n, n); p.lcnt = zeros(1, n);
p.rdep = zeros(n, n); p.rcnt = zeros(1, n);
p.head = zeros(1, n);
p.transition(p.SHIFT); % Only possible first move.
end % arceager13


% transition:
%
% SHIFT[(σ, b|β, A)] = (σ|b, β, A)
% RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
% LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})
% REDUCE[(σ|s, β, A)] = (σ, β, A)

function transition(p, op)
v = valid_moves(p);
assert(v(op));
switch op
 case p.SHIFT                   % move n0 to s; same as archybrid
  p.sptr = p.sptr + 1;		
  p.stack(p.sptr) = p.wptr;
  p.wptr = p.wptr + 1;
 case p.RIGHT                   % (s0,n0) followed by a shift
  p.add_arc(p.stack(p.sptr), p.wptr);
  p.sptr = p.sptr + 1;
  p.stack(p.sptr) = p.wptr;
  p.wptr = p.wptr + 1;
 case p.LEFT                    % (n0,s0) and pop s0; same as archybrid
  if (p.wptr <= p.nword)        % if no n0, leave the head as ROOT=0
    p.add_arc(p.wptr, p.stack(p.sptr)); 
  end
  p.sptr = p.sptr - 1;
 case p.REDUCE                  % pop s0
  p.sptr = p.sptr - 1;
 otherwise
  error('Move %d is not supported.\n', op);
end % switch
end % transition


% valid_moves:
%
% There is a precondition on the RIGHT and SHIFT transitions to be
% legal only when b ~= ROOT (p.wptr <= p.nword), and for LEFT, RIGHT
% and REDUCE to be legal only when the stack is non-empty (p.sptr >=
% 1). Moreover, LEFT is only legal when s does not have a parent in A,
% and REDUCE when s does have a parent in A.
%
% We terminate when the buffer is empty (i.e. contains root) and there
% is a single word in stack (with head=0), we do not perform the
% mandatory last LEFT move.  This ensures 2n-2 moves for each
% sentence.

function v = valid_moves(p)
v(p.SHIFT)  = (p.wptr <= p.nword);
v(p.RIGHT)  = ((p.sptr >= 1) && (p.wptr <= p.nword));
v(p.LEFT)   = ((p.sptr >= 1) && (p.head(p.stack(p.sptr)) == 0) && ...
               ~((p.sptr == 1) && (p.wptr > p.nword)));
v(p.REDUCE) = ((p.sptr >= 1) && (p.head(p.stack(p.sptr)) ~= 0));
end % valid_moves


% oracle_cost counts gold arcs that become impossible after a move.
% In the arc-eager system: A token starts life without any arcs in the
% buffer.  It moves to the head of the buffer (n0) with shift+right
% moves.  Left deps are acquired while in n0 position.  Then rdeps are
% acquired while in stack.  Head is acquired at s0 before (if
% left-head) or after (if right-head) the rdeps.  

function c = oracle_cost(p, gold)

assert(numel(gold) == p.nword); % gold is the correct head vector.
c = Inf(1, p.NMOVE);            % cost for illegal moves is inf.
isvalid = valid_moves(p);

if (p.sptr >= 1)

  n0 = p.wptr;
  s0 = p.stack(p.sptr);
  s = p.stack(1:p.sptr);
  ss = p.stack(1:p.sptr-1);
  s0b = sum(gold(n0:end) == s0);
  n0s = sum((gold(s)==n0) & (p.head(s)==0));

  if isvalid(p.SHIFT)
    % SHIFT moves n0 to s: n0 gets no more ldeps or lhead
    c(p.SHIFT) = n0s + sum(s == gold(n0));
  end
  if isvalid(p.RIGHT)
    % RIGHT adds (s0,n0): n0 gets no more ldeps, rhead, 0head, or lhead<s0
    c(p.RIGHT) = n0s + (gold(n0) > n0) + (gold(n0) == 0) + sum(ss == gold(n0));
  end
  if isvalid(p.LEFT)
    % LEFT  adds (n0,s0): s0 gets no more rdeps, rhead>n0 or 0head
    c(p.LEFT) = s0b + (gold(s0) > n0) + ((gold(s0) == 0) && (n0 <= p.nword));
  end
  if isvalid(p.REDUCE)
    % REDUCE pops s0: s0 gets no more rdeps
    c(p.REDUCE) = s0b;
  end
elseif isvalid(p.SHIFT)
  % SHIFT is the only legal move when stack empty
  c(p.SHIFT) = 0; 

end % if (p.sptr >= 1)

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
