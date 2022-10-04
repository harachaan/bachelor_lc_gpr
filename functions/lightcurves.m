function m_app = lightcurves(u_sun, u_obs, s, d, rho, ...
    F_o, surface_reflection, A, C_sun, u, altitude)

% calculation for the magnitude of lightcurves

% parameters
n_u = surface_reflection(1,1); n_v = surface_reflection(1,2);
u_u = u(:,1); u_v = u(:,2); u_n = u(:,3);

% 太陽光ベクトルと観測者ベクトルの二等分ベクトル
u_h = (u_sun + u_obs); u_h = u_h ./ norm(u_h);
% Rs, Rdを求めるのに必要な定数たち
k1 = sqrt((n_u + 1) * (n_v + 1) / (8 * pi));
k2 = (28*rho / 23*pi) * (1 - s*F_o);
z = (n_u*dot(u_h, u_u)^2 + n_v*dot(u_h, u_v)^2) / (1 - dot(u_h, u_n)^2);
% フレネル係数
F_reflect = s*F_o + (1 - s*F_o)*(1 - dot(u_sun, u_h))^5;
% Ashikhmin-shirley Model
R_s = k1 * dot(u_h, u_n)^z / (dot(u_sun, u_h)...
    * max([dot(u_obs, u_n) dot(u_sun, u_n)])) * F_reflect;
R_d = k2 * (1 - (1 - dot(u_obs, u_n)/2)^5)...
    * (1 - (1 - dot(u_sun, u_n)/2)^5);
% BRDF
f_r = s*R_s + d*R_d;
% LightCurves model
F_sun = C_sun * f_r * dot(u_sun, u_n);
F_obs = (F_sun * A * dot(u_obs, u_n)) / (altitude^2);
% 宇宙機の等級
% F_obs / C_sun
m_app = -26.7 - 2.5 * log10(abs(F_obs / C_sun)); % -26.7: 太陽光の見かけの等級
