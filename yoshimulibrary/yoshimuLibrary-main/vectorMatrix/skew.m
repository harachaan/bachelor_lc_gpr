function S = skew(x)
% ----------------------------------------------------------------------
%   2011/12/26 yy make skew-symmetric matrix from vector x.
%    Input: x, 3x1 vector or 1x3 vector
%   Output: S, 3x3 matrix
%           S = [0 -x(3) x(2)
%                x(3) 0 -x(1)
%                -x(2) x(1) 0];
%   related function files:
%   note:
%   revisions;
%   S = skew(x);
%   (c) 2016 yasuhiro yoshimura
%----------------------------------------------------------------------
%  

S = [0 -x(3) x(2)
    x(3) 0 -x(1)
    -x(2) x(1) 0];

end
