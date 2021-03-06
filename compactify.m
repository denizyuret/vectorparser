function model = compactify(model)

% [C, ia, ic] = unique(A,'rows')
% Find the unique rows C(u,d) of A(n,d) and the index vectors ia(u,1) and ic(n,1), such that ...
% C = A(ia,:) and A = C(ic,:).

fprintf('Finding unique SV in %d...\n', size(model.SV, 2));
[~, ia, ic] = unique(model.SV', 'rows');

fprintf('Saving %d unique SV.\n', numel(ia));
b2 = isfield(model, 'beta2');
nc = size(model.beta, 1);
newbeta  = zeros(nc, numel(ia), 'like', model.beta);
if b2 newbeta2 = zeros(nc, numel(ia), 'like', model.beta2); end
assert(numel(ic) == size(model.beta, 2));

for oldi=1:numel(ic)
  newi = ic(oldi);
  newbeta(:,newi) = newbeta(:,newi) + model.beta(:,oldi);
  if b2 newbeta2(:,newi) = newbeta2(:,newi) + model.beta2(:,oldi); end
end

model.SV = model.SV(:,ia);
model.beta = newbeta;
if b2 model.beta2 = newbeta2; end

end
