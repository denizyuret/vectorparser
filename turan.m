% The goal is to get all third degree monomials of x by matrix
% operations that GPU can perform.  This can be achieved (redundantly)
% by vec(x*x')*x'.  We need to filter the entries of x*x' before
% multiplying with x' again, otherwise we have too many redundant
% entries.
% 
% - Ideally the number of triples should be n(n+1)(n+2)/6
% - (x*x')*x' has n^3 entries.
% - triu(x*x')*x' has n*n*(n+1)/2, better but not very good.
% - turan(x*x')*x' has n*⌊(n-1)/2⌋*⌈(n-1)/2⌉, probably best with matrices.
% Thanks to Sule for telling me how to construct this matrix using
% Turan's thorem.
function m = turan(n)
m = false(n,n);
for i=1:n
  m(i,i)=true;
end
n2 = floor(n/2);
for i=1:n2
  for j=i:n2
    m(i,j)=true;
  end
end
for i=n2+1:n
  for j=i:n
    m(i,j)=true;
  end
end
end
