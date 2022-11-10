function out = hms2deg(hour, min, sec)
% ----------------------------------------------------------------------
%   hour, min, and sec angles to deg
%    20210419  y.yoshimura
%    Inputs: hour angle
%            min. angle
%            sec. angle
%   Outputs: out: angle, deg
%   related function files:
%   note:
%   cf:
%   revisions;
%   
%   (c) 2021 yasuhiro yoshimura
%----------------------------------------------------------------------

out = hour .* 15.0 + (min .* 60 + sec) ./ 3600;


end