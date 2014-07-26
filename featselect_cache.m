function c = featselect_cache(file)
fid = fopen(file);
a = textscan(fid, '%f\t%s');
err = a{1};
str = a{2};
c = containers.Map();
for i=1:numel(err)
  c(str{i})=err(i);
end
end
