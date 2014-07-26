% measure how long it takes to convert vectors to poly3 basis vectors

function test_primal(x)
  for i=1:size(x,2)
    z = poly3basis(x(:,i));
  end
end
