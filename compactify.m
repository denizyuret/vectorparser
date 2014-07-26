function model = compactify(model)

newbeta = [];
newbeta2 = [];
newSV = [];
newS = [];

fprintf('Discovering redundant SV.\n');
idx = [];tic;
for i=1:numel(model.S)
  if (mod(i, 1000) == 0) fprintf('.'); end
  si = model.S(i);
  if (si <= numel(idx) && idx(si) ~= 0)
    xi = idx(si);
    assert(all(model.SV(:,i) == newSV(:,xi)), '%d: %d~=%d\n', si, ...
           i, oldidx(si));
    newbeta(:,xi) = newbeta(:,xi) + model.beta(:,i);
    newbeta2(:,xi) = newbeta2(:,xi) + model.beta2(:,i);
  else
    newS(end+1) = si;
    idx(si) = numel(newS);
    oldidx(si) = i;
    newSV(:,end+1) = model.SV(:,i);
    newbeta(:,end+1) = model.beta(:,i);
    newbeta2(:,end+1) = model.beta2(:,i);
  end
end
fprintf('\n');
model.S = newS;
model.SV = newSV;
model.beta = newbeta;
model.beta2 = newbeta2;

end
