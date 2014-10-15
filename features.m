function [f, fidx] = features(p, s, fselect)

% Given a parser state p and a sentence s returns a feature vector
% fselect is a nx3 matrix whose rows determine which features to extract
% Each row of fselect consists of the following three values:
% 1. anchor word: 0:n0, 1:n1, 2:n2, ..., -1:s0, -2:s1, -3:s2, ...
% 2. target word: 0:self, 1:rightmost child, 2:second rightmost, -1:leftmost, -2:second leftmost ...
% 3. feature: One of the features listed below.
%
% 0:wvec (word+context if token, dim depends on encoding)
% +-1: exists/doesn't (one bit)
% +-2: right/left child count (4 bits, == encoding: 0,1,2,3+)
% +3: distance to right (4 bits, == encoding: 1,2,3,4+, root is 4+)
% -3: average of in-between tokens to the right (dim:wvec)
% +-4: word/context half of vector (dim:wvec/2, only valid for token encoding, assumes first half=word, second half=context)
% +-5: right/left child count (4 bits, >= encoding: >=1, >=2, >=3, >=4)
% +-6: right/left child count (4 bits, <= encoding: <=0, <=1, <=2, <=3)
% +7: distance to right, >= encoding (8 bits, >= encoding: >=2, >=3, >=4, >=6, >=8, >=12, >=16, >=20)
% -7: distance to right, <= encoding (8 bits, <= encoding: <=1, <=2, <=3, <=4, <=6, <=8, <=12, <=16)
% +-8: average of in-between word/context vectors to the right (dim:wvec/2)
% +-9: head exists/doesn't (one bit)

ndim = size(s.wvec,1);                  % token vector dimensionality
ndim2 = ndim/2;                         % for token encodings the first half is the word vector, the second half is the context vector
nfeat = size(fselect, 1);               % number of features
imax = nfeat*ndim;                      % maximum number of dimensions
smax = 1000;                            % maximum distance in sentence
f = zeros(1,imax,'like',s.wvec);        % feature vector
i = 0;                                  % index into feature vector
fidx = zeros(1,nfeat,'uint32');         % fidx(ifeat): end of feature fselect(ifeat,:) in f

for ifeat=1:nfeat
  feat = fselect(ifeat,:);

  % identify the anchor a: 8.81us
  if feat(1) >= 0                       % buffer word
    a = p.wptr + feat(1);
    if (a > p.nword) a = 0; end;
  else                                  % stack word
    ax = p.sptr + feat(1) + 1;
    if (ax > 0) 
      a = p.stack(ax);
      if ((a < 1) || (a >= p.wptr)) error('Bad anchor'); end;
    else 
      a = 0; 
    end
  end

  % identify the target b: 3.13us
  if a == 0
    b = 0;
  elseif feat(2) == 0                   % self
    b = a;                              
  elseif feat(2) > 0                    % right-child
    if (a >= p.wptr) error('buffer words do not have rdeps'); end;
    nc = p.rcnt(a);
    bx = nc - feat(2) + 1;
    if (bx > 0)
      b = p.rdep(a, bx);
      if ((b <= a) || (b > p.nword) || (p.head(b) ~= a)) error('Bad rdep'); end;
    else
      b = 0;
    end
  elseif feat(2) < 0                    % left-child
    if (a > p.wptr) error('buffer words other than n0 do not have ldeps'); end;
    nc = p.lcnt(a);
    bx = nc + feat(2) + 1;
    if (bx > 0)
      b = p.ldep(a, bx);
      if ((b < 1) || (b >= a) || p.head(b) ~= a) error('Bad ldep'); end;
    else
      b = 0;
    end
  end

  % generate the feature: 4.16us
  switch feat(3)
   case 0	% wvec
    if (b > 0) f(i+1:i+ndim) = s.wvec(:,b); end;
    i=i+ndim; fidx(ifeat)=i;

   case 1       % exists
    if (b > 0) f(i+1) = 1; end 
    i=i+1; fidx(ifeat)=i;

   case -1      % does not exist
    if (b == 0) f(i+1) = 1; end 
    i=i+1; fidx(ifeat)=i;

   case 2       % rdep count, 4 bits, == encoding
    if (b > 0) 
      if (b >= p.wptr) error('buffer words do not have rdeps'); end;
      nc = p.rcnt(b); if (nc > 3) nc = 3; end;
      f(i+1+nc) = 1;
    end 
    i=i+4; fidx(ifeat)=i;

   case -2      % ldep count, 4 bits, == encoding
    if (b > 0)
      if (b > p.wptr) error('buffer words other than n0 do not have ldeps'); end;
      nc = p.lcnt(b); if (nc > 3) nc = 3; end;
      f(i+1+nc) = 1;
    end 
    i=i+4; fidx(ifeat)=i;

   case 3       % distance to the right
    if (~(feat(1) < 0 && feat(2) == 0)) error('distance only available for stack words'); end
    if (b > 0)
      if feat(1) == -1 % s0n0 distance
        if (p.wptr <= p.nword) c = p.wptr;
        else c = b+smax; end % root is far far away...
      else      % s(i)-s(i-1) distance
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if (c <= b) error('c <= b'); end;
      d = c - b; if (d > 4) d = 4; end;
      f(i+d) = 1;
    end 
    i=i+4; fidx(ifeat)=i;

   case 4       % word (first) half of token vector
    if (b > 0) f(i+1:i+ndim2) = s.wvec(1:ndim2,b); end 
    i=i+ndim2; fidx(ifeat)=i;

   case -4      % context (second) half of token vector
    if (b > 0) f(i+1:i+ndim2) = s.wvec(ndim2+1:end,b); end 
    i=i+ndim2; fidx(ifeat)=i;

   case 5       % rdep count, 4 bits, >= encoding
    if (b > 0) 
      if (b >= p.wptr) error('buffer words do not have rdeps'); end;
      nc = p.rcnt(b);
      for ic=1:4 
        if nc >= ic
          f(i+ic) = 1;
        end
      end
    end 
    i=i+4; fidx(ifeat)=i;

   case -5      % ldep count, 4 bits, >= encoding
    if (b > 0)
      if (b > p.wptr) error('buffer words other than n0 do not have ldeps'); end;
      nc = p.lcnt(b);
      for ic=1:4 
        if nc >= ic
          f(i+ic) = 1;
        end
      end
    end 
    i=i+4; fidx(ifeat)=i;

   case 6       % rdep count, 4 bits, <= encoding
    if (b > 0) 
      if (b >= p.wptr) error('buffer words do not have rdeps'); end;
      nc = p.rcnt(b);
      for ic=0:3
        if nc <= ic
          f(i+1+ic) = 1;
        end
      end
    end 
    i=i+4; fidx(ifeat)=i;

   case -6      % ldep count, 4 bits, <= encoding
    if (b > 0)
      if (b > p.wptr) error('buffer words other than n0 do not have ldeps'); end;
      nc = p.lcnt(b);
      for ic=0:3 
        if nc <= ic
          f(i+1+ic) = 1;
        end
      end
    end 
    i=i+4; fidx(ifeat)=i;

   case 7       % distance to the right, >= encoding, 8 bits
    if (~(feat(1) < 0 && feat(2) == 0)) error('distance only available for stack words'); end;
    if (b > 0)
      if feat(1) == -1 % s0n0 distance
        if (p.wptr <= p.nword) c = p.wptr;
        else c = b+smax; end % root is far far away...
      else      % s(i)-s(i-1) distance
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if (c <= b) error('c <= b'); end;
      d = c - b;
      dmin = [2,3,4,6,8,12,16,20];
      for id=1:numel(dmin)
        if d >= dmin(id)
          f(i+id) = 1;
        end
      end
    end 
    i=i+8; fidx(ifeat)=i;

   case -7       % distance to the right, <= encoding, 8 bits
    if (~(feat(1) < 0 && feat(2) == 0)) error('distance only available for stack words'); end;
    if (b > 0)
      if feat(1) == -1 % s0n0 distance
        if (p.wptr <= p.nword) c = p.wptr;
        else c = b+smax; end % root is far far away...
      else      % s(i)-s(i-1) distance
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if (c <= b) error('c <= b'); end;
      d = c - b;
      dmax = [1,2,3,4,6,8,12,16];
      for id=1:numel(dmax)
        if d <= dmax(id)
          f(i+id) = 1;
        end
      end
    end 
    i=i+8; fidx(ifeat)=i;

   case -3      % avg of in-between token vectors to the right
    if (~(feat(1) < 0 && feat(2) == 0)) error('in-between only available for stack words'); end;
    if (b > 0)
      if feat(1) == -1 % s0n0 interval
        if (p.wptr <= p.nword) c = p.wptr;
        else c = b; end % no in-between with root
      else      % s(i)-s(i-1) interval
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if c > b+1
        avec = zeros(ndim,1,'like',s.wvec);
        for bc=(b+1):(c-1)
          avec = avec + s.wvec(:,bc);
        end
        f(i+1:i+ndim) = avec / (c-b-1);
      end
    end
    i=i+ndim; fidx(ifeat)=i;
    
   case 8      % avg of in-between word vectors to the right
    if (~(feat(1) < 0 && feat(2) == 0)) error('in-between only available for stack words'); end;
    if (b > 0)
      if feat(1) == -1 % s0n0 interval
        if (p.wptr <= p.nword) c = p.wptr;
        else c = b; end % no in-between with root
      else      % s(i)-s(i-1) interval
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if c > b+1
        avec = zeros(ndim2,1,'like',s.wvec);
        for bc=(b+1):(c-1)
          avec = avec + s.wvec(1:ndim2,bc);
        end
        f(i+1:i+ndim2) = avec / (c-b-1);
      end
    end
    i=i+ndim2; fidx(ifeat)=i;
    
   case -8      % avg of in-between context vectors to the right
    if (~(feat(1) < 0 && feat(2) == 0)) error('in-between only available for stack words'); end;
    if (b > 0)
      if feat(1) == -1 % s0n0 interval
        if (p.wptr <= p.nword) c = p.wptr;
        else c = b; end % no in-between with root
      else      % s(i)-s(i-1) interval
        cx = p.sptr + feat(1) + 2;
        c = p.stack(cx);
      end
      if c > b+1
        avec = zeros(ndim2,1,'like',s.wvec);
        for bc=(b+1):(c-1)
          avec = avec + s.wvec(ndim2+1:end,bc);
        end
        f(i+1:i+ndim2) = avec / (c-b-1);
      end
    end
    i=i+ndim2; fidx(ifeat)=i;

   case 9       % head exists
    if (b > 0) f(i+1) = (p.head(b) > 0); end 
    i=i+1; fidx(ifeat)=i;

   case -9      % head does not exist
    if (b > 0) f(i+1) = (p.head(b) == 0); end 
    i=i+1; fidx(ifeat)=i;

   otherwise
    error('Unknown feature %d.\n', feat(3));
  end % switch feat(3)
end % for ifeat=1:nfeat

assert(fidx(end) == i);
f = f(1:i);

end % features
