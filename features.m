% Given a parser state p and a sentence s returns a feature vector
% feats is a 3xn matrix that determines which features to extract
% Each column of feats consists of the following three values:
% 1. anchor word: 0:n0, 1:n1, 2:n2, ..., -1:s0, -2:s1, -3:s2, ...
% 2. target word: 0:self, 1:rightmost child, 2:second rightmost, -1:leftmost, -2:second leftmost ...
% 3. feature: 0:wvec, +-1: exists/doesn't, +-2: right/left child count, +3: distance to right

function f = features(p, s, feats)

ndim = size(s.wvec,1);                  % word vector dimensionality
imax = 10000;                           % maximum number of features
f = zeros(1,imax);                      % feature vector
i = 0;                                  % index into feature vector

for feat = feats                        % 16.10us/iter

  % identify the anchor a: 8.81us
  if feat(1) >= 0
    a = p.wptr + feat(1);
    if (a > p.nword) a = 0; end
  else
    ax = p.sptr + feat(1) + 1;
    if (ax > 0) 
      a = p.stack(ax);
      assert(a >= 1 && a <= p.nword);
    else 
      a = 0; 
    end
  end

  % identify the target b: 3.13us
  if a == 0
    b = 0;
  elseif feat(2) == 0
    b = a;
  elseif feat(2) > 0
    nc = p.rcnt(a);
    bx = nc - feat(2) + 1;
    if (bx > 0)
      b = p.rdep(a, bx);
      assert(b >= 1 && b <= p.nword && p.head(b) == a);
    else
      b = 0;
    end
  elseif feat(2) < 0
    nc = p.lcnt(a);
    bx = nc + feat(2) + 1;
    if (bx > 0)
      b = p.ldep(a, bx);
      assert(b >= 1 && b <= p.nword && p.head(b) == a);
    else
      b = 0;
    end
  end

  % generate the feature: 4.16us
  switch feat(3)
   case 0 
    if (b > 0) f(i+1:i+ndim) = s.wvec(:,b); end; i=i+ndim;
   case 1 
    if (b > 0) f(i+1) = 1; end; i=i+1;
   case -1  % for backward compatibility, can be simplified
    if ((b == 0) && (((a == 0) && (feat(2) == 0)) || ((a ~= 0) && (feat(2) ~= 0))))
      f(i+1) = 1; 
    end; i=i+1;
   case 2 
    if (b > 0) 
      nc = p.rcnt(b); if (nc > 3) nc = 3; end;
      f(i+1+nc) = 1;
    end; i=i+4;
   case -2
    if (b > 0)
      nc = p.lcnt(b); if (nc > 3) nc = 3; end;
      f(i+1+nc) = 1;
    end; i=i+4;
   case 3
    assert(feat(1) < 0 && feat(2) == 0,...
           'Distance only available for stack words');
    if (b > 0)
      if feat(1) == -1
        if (p.wptr <= p.nword) c = p.wptr;
        else c = 0; end
      else
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if (c > 0)
        assert(c > b);
        d = c - b; if (d > 4) d = 4; end
        f(i+d) = 1;
      end
    end; 
    i=i+4;
   otherwise
    error('Unknown feature %d.\n', feat(3));
  end % switch feat(3)
  assert(i <= imax, 'max feature vector length exceeded, need to increase imax');
end % for feat = feats

f = f(1:i);

end % features
