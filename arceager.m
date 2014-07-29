% arceager.m, Deniz Yuret, July 28, 2014: Transition based greedy arc-eager parser based on:
% http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
% Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.

classdef arceager < handle

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
function p = arceager(n)
p.nword = n;	       p.wptr = 1;
p.stack = zeros(1, n); p.sptr = 0;
p.ldep = zeros(n, n); p.lcnt = zeros(1, n);
p.rdep = zeros(n, n); p.rcnt = zeros(1, n);
p.head = zeros(1, n);
p.transition(p.SHIFT); % Only possible first move.
end % arceager


% In the arc-eager system (Nivre, 2003), a configuration c= (σ,β,A)
% consists of a stack σ, a buffer β, and a set A of dependency arcs.
%
% SHIFT[(σ, b|β, A)] = (σ|b, β, A)
% RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
% LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})
% REDUCE[(σ|s, β, A)] = (σ, β, A)

function transition(p, op)
switch op
 case p.SHIFT                   % same as archybrid
  p.sptr = p.sptr + 1;		
  p.stack(p.sptr) = p.wptr;
  p.wptr = p.wptr + 1;
 case p.RIGHT                   % (s,w) followed by a shift
  p.add_arc(p.stack(p.sptr), p.wptr);
  p.sptr = p.sptr + 1;
  p.stack(p.sptr) = p.wptr;
  p.wptr = p.wptr + 1;
 case p.LEFT                    % (w,s) followed by reduce; same as archybrid

  % p.wptr can be nword+1 for root!  Our convention is root=0.  We
  % also start all p.head==0.  This conflates words with no head
  % yet and words with root head.  However the second case is only
  % possible with LEFT and results in the word being popped, so in
  % effect all words in stack+buffer with head==0 have no head and
  % all words off of stack+buffer have root head.

  if (p.wptr <= p.nword)
    p.add_arc(p.wptr, p.stack(p.sptr)); 
  end
  p.sptr = p.sptr - 1;
 case p.REDUCE                  % pops the stack
  p.sptr = p.sptr - 1;
 otherwise
  error('Move %d is not supported.\n', op);
end % switch
end % transition


% There is a precondition on the RIGHT and SHIFT transitions to be
% legal only when b ~= ROOT , and for LEFT , RIGHT and REDUCE to be
% legal only when the stack is non-empty. Moreover, LEFT is only legal
% when s does not have a parent in A, and REDUCE when s does have a
% parent in A.  The parser terminates when p.wptr == p.nword+1 and
% p.sptr == 0.

function v = valid_moves(p)
v(p.SHIFT)  = (p.wptr <= p.nword);
v(p.RIGHT)  = ((p.sptr >= 1) && (p.wptr <= p.nword));
v(p.LEFT)   = ((p.sptr >= 1) && (p.head(p.stack(p.sptr)) == 0));
v(p.REDUCE) = ((p.sptr >= 1) && (p.head(p.stack(p.sptr)) ~= 0));
end % valid_moves


% In the arc-eager system:
% A token starts life without any arcs in the buffer.
% It becomes n0 after a number of shift+right.
% n0 acquires ldeps using left+reduce.
% It becomes s0 using shift (s0s) if right-head.
% It becomes s0 using right (s0r) if left-head.
% s0 acquires rdeps using right+reduce.
% s0r ends with reduce.
% s0s ends with left.
%
% i.e. left deps are acquired first while in buffer.
% then rdeps are acquired while in stack.
% head is acquired in stack before or after the rdeps.
% before if left-head, after if right-head.

% In the arc-eager system, an arc (h,d) is reachable from a
% configuration c if one of the following conditions hold: 
% (1) (h,d) is already derived ((h,d) ∈ A_c); 
% (2) h and d are in the buffer; 
% (3) h is on the stack and d is in the buffer; 
% (4) d is on the stack and is not assigned a head and h is in the buffer.

% Oracle counts gold arcs that become impossible after a move:
% 1. SHIFT moves n0 to s: [(n0,s) & head(s)==0] + [gold(n0) in s]
% 2. RIGHT adds (s0,n0) : [(n0,s) & head(s)==0] + [gold(n0) ~= s0]
% 3. LEFT  adds (n0,s0) : (s0,b) + (b\n0,s0)
% 4. REDUCE pops s0     : (s0,b)

function c = oracle_cost(p, gold)

assert(numel(gold) == p.nword); % gold is the correct head vector.
c = Inf(1, p.NMOVE);            % cost for illegal moves is inf.

if (p.sptr >= 1)
  s0 = p.stack(p.sptr);
  n0 = p.wptr;
  s0b = sum(gold(n0:end) == s0);
  if (p.head(s0) == 0)
    c(p.LEFT) = s0b + (gold(s0) > p.wptr) + (gold(s0) == 0);
  else
    c(p.REDUCE) = s0b;
  end
  if (n0 <= p.nword)
    s = p.stack(1:p.sptr);
    n0s = sum((gold(s)==n0) & (p.head(s)==0));
    c(p.SHIFT) = n0s + sum(s == gold(n0));
    c(p.RIGHT) = n0s + (gold(n0) ~= s0);
  end
elseif p.wptr <= p.nword
  c(p.SHIFT) = 0;
end

v = valid_moves(p);
assert(all(isfinite(c(v))) && all(isinf(c(~v))));

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
