function fval = checkfield(x, f)

% isfield does not work for objects
% isprop does not work for structs
% getfield works for both but gives error if field not defined
% fieldnames work for both
% if any(strcmp(f, fieldnames(x)))
% but this might be faster

  if isfield(x, f) || isprop(x, f)
    fval = x.(f);
  else
    fval = [];
  end
end
