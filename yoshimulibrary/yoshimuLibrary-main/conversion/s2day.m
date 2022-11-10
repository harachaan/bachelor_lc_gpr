function day = s2day(s)
% ----------------------------------------------------------------------
%   transform second to days
%    20210215  y.yoshimura
%    Inputs: s, second, nx1 vector
%   Outputs: day, day, nx1 vector
%   related function files:
%   note:
%   cf:
%   revisions;
%   function day = s2day(sec)
%   (c) 2021 yasuhiro yoshimura
%----------------------------------------------------------------------

day = s ./ 24 ./ 60 ./ 60;


end