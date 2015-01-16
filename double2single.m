function y = double2single(x)
if isa(x, 'double')
    y = single(x);
elseif isa(x, 'gpuArray')
    y = gpuArray(single(gather(x)));
elseif isa(x, 'cell')
    y = x;
    for i=1:numel(y)
        y{i} = double2single(y{i});
    end
elseif isa(x, 'struct')
    y = x;
    names = fieldnames(y);
    for i=1:numel(names)
        f = names{i};
        y = setfield(y, f, double2single(getfield(y, f)));
    end
else   % char, single, etc.
    y = x;
end
end
