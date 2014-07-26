function [idx,fidx,n] = featureindices(feats)
fidx = containers.Map(); i=0;
fidx('n0') = i+1:i+100; i=i+100;
fidx('n0l') = i+1:i+4; i=i+4;
fidx('n0r') = i+1:i+4; i=i+4;
fidx('n0+') = i+1:i+1; i=i+1;
fidx('n0-') = i+1:i+1; i=i+1;

fidx('n1') = i+1:i+100; i=i+100;
fidx('n1l') = i+1:i+4; i=i+4;
fidx('n1r') = i+1:i+4; i=i+4;
fidx('n1+') = i+1:i+1; i=i+1;
fidx('n1-') = i+1:i+1; i=i+1;

fidx('n2') = i+1:i+100; i=i+100;
fidx('n2l') = i+1:i+4; i=i+4;
fidx('n2r') = i+1:i+4; i=i+4;
fidx('n2+') = i+1:i+1; i=i+1;
fidx('n2-') = i+1:i+1; i=i+1;

fidx('s0') = i+1:i+100; i=i+100;
fidx('s0l') = i+1:i+4; i=i+4;
fidx('s0r') = i+1:i+4; i=i+4;
fidx('s0+') = i+1:i+1; i=i+1;
fidx('s0-') = i+1:i+1; i=i+1;

fidx('s1') = i+1:i+100; i=i+100;
fidx('s1l') = i+1:i+4; i=i+4;
fidx('s1r') = i+1:i+4; i=i+4;
fidx('s1+') = i+1:i+1; i=i+1;
fidx('s1-') = i+1:i+1; i=i+1;

fidx('s2') = i+1:i+100; i=i+100;
fidx('s2l') = i+1:i+4; i=i+4;
fidx('s2r') = i+1:i+4; i=i+4;
fidx('s2+') = i+1:i+1; i=i+1;
fidx('s2-') = i+1:i+1; i=i+1;

fidx('s0l1') = i+1:i+100; i=i+100;
fidx('s0l1l') = i+1:i+4; i=i+4;
fidx('s0l1r') = i+1:i+4; i=i+4;
fidx('s0l1+') = i+1:i+1; i=i+1;
fidx('s0l1-') = i+1:i+1; i=i+1;

fidx('s0l2') = i+1:i+100; i=i+100;
fidx('s0l2l') = i+1:i+4; i=i+4;
fidx('s0l2r') = i+1:i+4; i=i+4;
fidx('s0l2+') = i+1:i+1; i=i+1;
fidx('s0l2-') = i+1:i+1; i=i+1;

fidx('s0r1') = i+1:i+100; i=i+100;
fidx('s0r1l') = i+1:i+4; i=i+4;
fidx('s0r1r') = i+1:i+4; i=i+4;
fidx('s0r1+') = i+1:i+1; i=i+1;
fidx('s0r1-') = i+1:i+1; i=i+1;

fidx('s0r2') = i+1:i+100; i=i+100;
fidx('s0r2l') = i+1:i+4; i=i+4;
fidx('s0r2r') = i+1:i+4; i=i+4;
fidx('s0r2+') = i+1:i+1; i=i+1;
fidx('s0r2-') = i+1:i+1; i=i+1;

fidx('s1l1') = i+1:i+100; i=i+100;
fidx('s1l1l') = i+1:i+4; i=i+4;
fidx('s1l1r') = i+1:i+4; i=i+4;
fidx('s1l1+') = i+1:i+1; i=i+1;
fidx('s1l1-') = i+1:i+1; i=i+1;

fidx('s1l2') = i+1:i+100; i=i+100;
fidx('s1l2l') = i+1:i+4; i=i+4;
fidx('s1l2r') = i+1:i+4; i=i+4;
fidx('s1l2+') = i+1:i+1; i=i+1;
fidx('s1l2-') = i+1:i+1; i=i+1;

fidx('s1r1') = i+1:i+100; i=i+100;
fidx('s1r1l') = i+1:i+4; i=i+4;
fidx('s1r1r') = i+1:i+4; i=i+4;
fidx('s1r1+') = i+1:i+1; i=i+1;
fidx('s1r1-') = i+1:i+1; i=i+1;

fidx('s1r2') = i+1:i+100; i=i+100;
fidx('s1r2l') = i+1:i+4; i=i+4;
fidx('s1r2r') = i+1:i+4; i=i+4;
fidx('s1r2+') = i+1:i+1; i=i+1;
fidx('s1r2-') = i+1:i+1; i=i+1;

fidx('n0l1') = i+1:i+100; i=i+100;
fidx('n0l1l') = i+1:i+4; i=i+4;
fidx('n0l1r') = i+1:i+4; i=i+4;
fidx('n0l1+') = i+1:i+1; i=i+1;
fidx('n0l1-') = i+1:i+1; i=i+1;

fidx('n0l2') = i+1:i+100; i=i+100;
fidx('n0l2l') = i+1:i+4; i=i+4;
fidx('n0l2r') = i+1:i+4; i=i+4;
fidx('n0l2+') = i+1:i+1; i=i+1;
fidx('n0l2-') = i+1:i+1; i=i+1;

fidx('n0s0') = i+1:i+4; i=i+4;
fidx('s0s1') = i+1:i+4; i=i+4;
n = i;
idx = [];
if nargin > 0
  for i=1:length(feats) 
    idx = [idx, fidx(feats{i})]; 
  end
end
end % featureindices
