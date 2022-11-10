function B = normRow(A)
% ----------------------------------------------------------------------
%   Normalize row vectors of matrix A
%    20190218  y.yoshimura
%    Inputs: A: matrix, mxn matrix
%   Outputs: B: matrix has unit row vectors, mxn matrix
%   related function files:
%   note:
%   cf:
%   revisions;
%   
%   (c) 2019 yasuhiro yoshimura
%----------------------------------------------------------------------

B = A ./ vecnorm(A, 2, 2);

end