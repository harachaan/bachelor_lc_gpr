function b = igrf12(year, alt, lat, lon)
% ----------------------------------------------------------------------
%   calculate magnetic field with IGRF12 model
%    20190216  y.yoshimura
%    Inputs: year
%            alt: altitude, km, above sea level
%            lat: north latitude, rad,
%            lon: east longitude, rad
%   Outputs: b: magnetic vector, 1x3 vector, nT, @NED coordinates
%   related function files:
%   note: use the mex file generated from igrfsyn12.c
%   cf:
%   revisions;
%   
%   (c) 2019 yasuhiro yoshimura
%----------------------------------------------------------------------

b = igrfsyn12(year, alt, rad2deg(lat), rad2deg(lon));

end