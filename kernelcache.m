classdef kernelcache < handle

properties
size;
keys;
vals;
rvec;
mean;
std;
hit;
miss;
nkeys;
end % properties

methods

function c = kernelcache(keys, vals);
ndims = size(keys, 1);
nkeys = size(keys, 2);
c.size = 100 * nkeys;
c.keys = cell(1, c.size);
c.vals = cell(1, c.size);
c.rvec = rand(1, ndims);
normval = c.rvec * keys;
c.mean = mean(normval);
c.std = std(normval);
cnum = cellnum(c, keys, normval);
c.nkeys = 0;
for i=1:nkeys
  j = cnum(i);
  if isempty(c.keys{j}) c.nkeys = c.nkeys + 1; end;
  c.keys{j} = keys(:,i);
  c.vals{j} = vals(:,i);
end
c.hit = 0;
c.miss = 0;
end

function cnum = cellnum(c, keys, normval)
if nargin < 3
  normval = c.rvec * keys;
end
unifval = normcdf(normval, c.mean, c.std);
cnum = ceil(c.size * unifval);
end

function val = get(c, key)
cnum = cellnum(c, key);
ckey = c.keys{cnum};
if (~isempty(ckey) && all(ckey == key))
  val = c.vals{cnum};
  c.hit = c.hit + 1;
else
  val = [];
  c.miss = c.miss + 1;
end
end

end % methods

end % classdef
