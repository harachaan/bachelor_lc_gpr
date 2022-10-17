% lightcurves：軌道位置と姿勢の適用


% -------------------------------------------------------------------------
clc
clear 
close all


% 手入力パラメータたち
%--------------------------------------------------------------------------
% for attitude, orbit

% time span 
t0 = 0; tf = 3600*2; % sec;
    tspan = [t0 tf];

% 形状パラメータ（平板:密度m一定）
shape = [1.0 1.0];
    a = shape(1, 1); b = shape(1, 2); % [m]
m = 1.0; % [kg/m^2]
    % 平板の面積，質量
    A = a * b; M = m * A;
    % 慣性テンソル（平板(a=1, b=1)）
    J = 1/3 * M * [b^2 0 0
                   0 a^2 0
                   0 0 a^2+b^2];

% 軌道パラメータ
r_earth = 6378.14 * 10^3; % [m]
altitude = 600 * 10^3; %[km]
mu = 398600.5 * 10^9; % 地心重力定数[m^3/s^2]

% 太陽の半径
% r_sun 
% 地球と太陽の距離
distance_sun_earth = 14960 * 10^4 * 10^3; % [m]


% 初期条件
% 姿勢(クォータニオン)
q0 = [0
      0
      0
      1];
    q0 = q0 / norm(q0); % 正規化

% 角速度(機体固定?) [rad/s]
w0 = [pi/3600
      0
      0];

% 軌道位置[km]，速度[km/s](cowell's formulation)IJK系
r0 = [5492.00034
      3984.00140
      2.95881] .* 10^3; 
v0 = [-3.931046491
      5.498676921
      3.665980697] .* 10^3; % from いそべさん
% r0 = rand(3,1); r0 = r0 ./ norm(r0);
% r0 = r0 * (r_earth + altitude);
% v0 = rand(3,1) .* 10;

% ode45のためにまとめる
y0 = [w0
      q0
      r0
      v0];
% トルクはフィードバック制御する時とかに逐一変えていくから関数の方に入れるのが一般的
% tau = [1
%        0
%        0];


%--------------------------------------------------------------------------

% これで，角速度，姿勢，軌道位置と速度の時間履歴を求めれる
[t, y] = ode45(@(t, y) eom_attitude_orbit(t, y, J, mu), tspan, y0);

%--------------------------------------------------------------------------

% odeによって得られた時間履歴(初期値は縦ベクトルだが，odeの解は時間が行，その瞬間の値が列に入ることに注意！)

% 角速度ベクトル
w1 = y(:,1); w2 = y(:,2); w3 = y(:,3);
w = [w1 w2 w3]'; % 各要素が縦ベクトルになるように転置

% quaternions 
    % q = [e*sin(theta/2) e:回転軸ベクトル
    %      cos(theta/2)]; theta:回転角
q1 = y(:,4); q2 = y(:,5); q3 = y(:,6); q4 = y(:,7);
q = [q1 q2 q3 q4]';
for i = 1:1:length(q)
    q(:,i) = q(:,i) / norm(q(:,i)); % 正規化(quaternionsのノルム拘束条件)
end
q_inv = [-q(1:3,:)
          q(4,:)]; % 共役クォータニオン(逆クォータニオン) 

% 角運動量ベクトル(h_b:機体固定座標系から見た，h_i:慣性系から見た)

% h_b = zeros(3, length(w)); % 事前割り当て
% h_i_q = zeros(4, length(w));
% for i = 1:1:length(w)
%     h_b(:,i) = J * w(:,i);
%     h_b_q = [h_b(:,i)
%                0]; % クォータニオン積のために4行ベクトル化
%     q_bi = q (:, i); % i系からb系への回転を表すクォータニオン
%     q_bi_inv = q_inv(:,i);
%     h_i_q(:,i) = q_pro(q_pro(q_bi_inv, h_b_q), q_bi);
% end
% h_i1 = h_i_q(1,:); h_i2 = h_i_q(2,:); h_i3 = h_i_q(3,:);
% h_i = [h_i1
%        h_i2
%        h_i3];

h_b = zeros(3, length(w)); % 事前割り当て
for i = 1:1:length(w)
    h_b(:,i) = J * w(:,i);
end
h_i = transform_b_to_i(q, q_inv, h_b);
h_i1 = h_i(1,:); h_i2 = h_i(2,:); h_i3 = h_i(3,:);

% 角運動量ベクトルの大きさ(角運動量保存則の確認)
h_b_abs = zeros(1, length(h_b)); % 事前割り当て
h_i_abs = zeros(1, length(h_i));
for i = 1:1:length(h_i)
    h_b_abs(1,i) = norm(h_b(:,i));
    h_i_abs(1,i) = norm(h_i(:,i));
end

% 軌道位置・速度(from Cowell's Formulation)
x_cowell = y(:,8); y_cowell = y(:,9); z_cowell = y(:,10);
r_cowell = [x_cowell y_cowell z_cowell]'; % 転置に注意
xdot_cowell = y(:,11); ydot_cowell = y(:,12); zdot_cowell = y(:,13);
v_cowell = [xdot_cowell ydot_cowell zdot_cowell];


% -------------------------------------------------------------------------
% parameters for LightCurves

% 表面特性パラメータ
% 表面反射率s, 拡散率d, 拡散反射率rho
s = 0.7; d = 1-s; rho = 0.3;
% フレネル反射率
F_o = 0.3;
% 異方性考慮できるように
surface_reflection = [1000 1000];
    n_u = surface_reflection(1,1); n_v = surface_reflection(1,2);
% 物体表面に当たる可視光域太陽光の強さ
C_sun = 455; % [W/m^2]

% 機体固定座標系単位ベクトル
u_u = [1
       0
       0];
u_v = [0
       1
       0];
u_n = [0
       0
       1]; 
u = [u_u u_v u_n];
% 太陽光方向単位ベクトル(慣性系)
% u_sun_i = rand(3,1); u_sun_i = u_sun_i ./ norm(u_sun_i)
u_sun_i = [10 10 0]'; u_sun_i = u_sun_i ./ norm(u_sun_i); % IJK系のXY平面上においた
% 観測者方向ベクトル(慣性系)
% obs_vector = rand(3,1); obs_vector = obs_vector ./ norm(obs_vector);
% obs_vector = r_earth * obs_vector;
% SC_vector = rand(3,1); SC_vector = SC_vector ./ norm(SC_vector); % SC:Space Craft
% SC_vector = (r_earth + altitude) * SC_vector;
SC_vector = [3.8017 4.0097 4.2618]' * 1.0e+06;
obs_vector = [4.9188 4.0590 0.1029]' .* 1.0e+06;
u_obs_i = obs_vector - SC_vector;
h_t = norm(u_obs_i); % 観測者から見た宇宙機の相対位置ベクトルの大
u_obs_i = u_obs_i ./ h_t;
% u_obs_i = rand(3,1); u_obs_i = u_obs_i ./ norm(u_obs_i);
% u_obs_i = [0.7150 0.3636 0.5978]';
% それぞれ機体固定座標系へ変換
u_sun_i_hist = zeros(3, length(q)); % 事前割り当て
u_obs_i_hist = zeros(3, length(q));
for i = 1:1:length(q)
    u_sun_i_hist(:,i) = u_sun_i;
    u_obs_i_hist(:,i) = u_obs_i;
end
u_sun = transform_i_to_b(q, q_inv, u_sun_i_hist);
u_obs = transform_i_to_b(q, q_inv, u_obs_i_hist);

% u_sun = [0
%          0
%          1];
% u_obs = [0
%          1 / sqrt(2)
%          1 / sqrt(2)]; % 太陽と観測者同じでいいの？
%--------------------------------------------------------------------------
% calculation for the magnitude of lightcurves

% % 太陽光ベクトルと観測者ベクトルの二等分ベクトル
% u_h = (u_sun + u_obs); u_h = u_h ./ norm(u_h);
% % Rs, Rdを求めるのに必要な定数たち
% k1 = sqrt((n_u + 1) * (n_v + 1) / (8 * pi));
% k2 = (28*rho / 23*pi) * (1 - s*F_o);
% z = (n_u*dot(u_h, u_u)^2 + n_v*dot(u_h, u_v)^2) / (1 - dot(u_h, u_n)^2);
% % フレネル係数
% F_reflect = s*F_o + (1 - s*F_o)*(1 - dot(u_sun, u_h))^5;
% % Ashikhmin-shirley Model
% R_s = k1 * dot(u_h, u_n)^z / (dot(u_sun, u_h)...
%     * max([dot(u_obs, u_n) dot(u_sun, u_n)])) * F_reflect;
% R_d = k2 * (1 - (1 - dot(u_obs, u_n)/2)^5)...
%     * (1 - (1 - dot(u_sun, u_n)/2)^5);
% % BRDF
% f_r = s*R_s + d*R_d;
% % LightCurves model
% F_sun = C_sun * f_r * dot(u_sun, u_n);
% F_obs = (F_sun * A * dot(u_obs, u_n)) / (altitude^2);
% % 宇宙機の等級
% % F_obs / C_sun
% m_app = -26.7 - 2.5 * log10(abs(F_obs / C_sun)) % -26.7: 太陽光の見かけの等級

% u_sun, u_obsに姿勢運動による時間履歴を反映させてmagnitudeを求める
% u_h = zeros(3, length(q)); % 事前割り当て
% m_app = zeros(1, length(q));
% for i = 1:1:length(q)
%     u_h(:,i) = u_sun_b(:,i) + u_obs_b(:,i);
%     u_h(:,i) = u_h(:,i) ./ norm(u_h(:,i));
%     k1 = sqrt(n_u + 1) * (n_v + 1) / (8 * pi);
%     k2 = (28*rho / 23*pi) * (1 - s*F_o);
%     z = (n_u*dot(u_h(:,i), u_u)^2 + n_v*dot(u_h(:,i), u_v)^2) ...
%         / (1 - dot(u_h(:,i), u_n)^2);
%     F_reflect = s*F_o + (1 - s*F_o)*(1 - dot(u_sun_b(:,i), u_h(:,i)))^5;
%     R_s = k1 * dot(u_h(:,i), u_n)^z / (dot(u_sun_b(:,i), u_h(:,i))...
%         *max([dot(u_obs_b(:,i), u_n), dot(u_sun_b(:,i), u_n)])) * F_reflect;
%     R_d = k2 * (1 - (1 - dot(u_obs_b(:,i), u_n)/2)^5)...
%     * (1 - (1 - dot(u_sun_b(:,i), u_n)/2)^5);
%     % BRDF
%     f_r = s*R_s + d*R_d;
%     % LightCurves model
%     F_sun = C_sun * f_r * dot(u_sun_b(:,i), u_n);
%     F_obs = (F_sun * A * dot(u_obs_b(:,i), u_n)) / (altitude^2);
%     % 宇宙機の等級
%     m_app(:,i) = -26.7 - 2.5 * log10(abs(F_obs / C_sun)) % -26.7: 太陽光の見かけの等級
% end

% LightCurvesの計算の関数化
m_app = zeros(1, length(q));
for i = 1:1:length(q)
    m_app(1,i) = lightcurves(u_sun(:,i), u_obs(:,i), s, d, rho, ...
        F_o, surface_reflection, A, C_sun, u, h_t);
end    


%--------------------------------------------------------------------------



%--------------------------------------------------------------------------
% f1 = figure; f2 = figure;
% 
% figure(f1);
% plot(t, h_b_abs, '-b')
% title('角運動量ベクトル(b系)の大きさ')
% 
% figure(f2);
% plot(t, h_i_abs, '-r')
% title('角運動量ベクトル(系)の大きさ')
f3 = figure;
figure(f3);
plot(t, m_app, '-b');
title('LightCurves');
xlabel('t'); ylabel('magnitude');


