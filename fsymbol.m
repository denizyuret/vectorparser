function s = fsymbol(f)
a = {'s3','s2','s1','s0','n0','n1','n2','n3'};
b = {'l3','l2','l1','','r1','r2','r3'};
c = {'h-','ac','d<','l<','l>','c','at','l=','-','t','+','r=','d=','w','r>','r<','d>','aw','h+'};
s = [ a{f(1)+5} b{f(2)+4} c{f(3)+10} ];
end

