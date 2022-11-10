function b = geodeticIGRF(jd, lat, lon, alt, coefs)
% ----------------------------------------------------------------------
%   calculate geocentric magnetic vector with IGRF model
%    20210428  y.yoshimura
%    Inputs: jd: Julian day, day
%            lat: geodetic latitude, rad
%            lon: geodetic longitude, rad
%            alt: geodetic altitude, km
%   Outputs: b: magnetic field, [northward, eastward, downward], nT
%   related function files:
%   note:
%   cf:
%   revisions;
%   
%   (c) 2021 yasuhiro yoshimura
%----------------------------------------------------------------------

[year, month, day, hour, minute, second] = jd2gc(jd);
time = datenum([year month day hour minute second]);

% inuput arguments are scalar → use igrfs.m
[bx, by, bz] = igrfs(time, rad2deg(lat), rad2deg(lon), alt, 'geodetic', coefs);
b = [bx, by, bz];
end