%---------------------------------------------------
% Light curve analysis 
% ・トルクの導入
% 20190821 yuri matsushita(M1)
%---------------------------------------------------

workspace

clc
clear 
close all

%%対象物体の定義 %should be updated
%対象物体の点座標
global pr;
pr_1 = [0.5 0.5 0.5]';
pr_2 = [-0.5 0.5 0.5]';
pr_3 = [-0.5 -0.5 0.5]';
pr_4 = [0.5 -0.5 0.5]';
pr = [pr_1 pr_2 pr_3 pr_4];

%反射率
global s d F0 rho;
s = 0.7;
d = 1 - s;
F0 = 0.3;
rho = 0.3;

%慣性モーメント
global Ix Iy Iz;
Ix = 60;
Iy = 40;
Iz = 80;

%%物体の初期情報
%角速度
wx = 0.4;
wy = 0.4;
wz = 0.4;
w = [wx wy wz]';

%外乱トルク
Tx = 0.0;
Ty = 0.0;
Tz = 0.0;
T = [Tx Ty Tz]';

%状態量の初期値 [phi,theta,psi]' %should be updated
phi = 10;
theta = 20;
psi = 30;
xEst = [phi,theta,psi]'; %[deg]
xEst = DegtoRad(xEst); %[rad]

%軌道の定義 %should be updated
global a Fw h_orbit;
global altitude;

r_earth = 6378.14; 
altitude = 600;
a = r_earth + altitude;    %軌道長半径
myu = 398600.5; %[km^3/s^2]
e = 0;                     %離心率
h_orbit = sqrt(myu*a*(1-e*e));   %角運動量
Fw = 0;                    %軌道面外方向外力
omega_earth = 7.292115 * 10.^(-5);

%トルクの係数
global C_tg C_td C_ts

%重力トルク
C_tg = 3 * myu / (a * a * a);

%大気抵抗トルク
Cd = 2.0;
rho_d = 1.137 * 10.^(-13);
C_td = -0.5 * rho_d * Cd;

%太陽輻射圧トルク
P = 4.5 * 10.^(-6);
r_sun = 1;   %[AU]
C_ts = -P / (r_sun * r_sun);

%太陽と観測者の位置関係
global u_obs u_sun u_h;  
U_obs = [0 0 1]';           %観測者ベクトル
U_sun = [0 0 1]';           %太陽光ベクトル
u_obs = U_obs/norm(U_obs);
u_sun = U_sun/norm(U_sun);

U_h = U_obs + U_sun;       %太陽光と観測者の二等分ベクトル
u_h = U_h/norm(U_h);

%タイムステップの定義
t = 0;
endtime = 10; %終端時間[秒] %should be updated
global dt;
dt = 0.1; %タイムステップ[秒]
nStep = (endtime - t)/dt;  %ステップ数

%配列の用意
result.magnitude = [];
result.C = [];
result.alpha = [];
result.w = [];
result.xEst = [];

for i = 1:nStep
    t = t + dt;
    w = fw_LungeKutta(t,w);
    xEst = f(t,xEst,w);
 
    [C,alpha,Sigma,A] = calc_plane(xEst,pr);
    magnitude = Magnitude(C);
    
    result.magnitude = [result.magnitude;magnitude'];
    result.C = [result.C;C'];
    result.alpha = [result.alpha;alpha'];
    result.w = [result.w;w'];
    result.xEst = [result.xEst;xEst'];
end

%%図を表示
%対象物体の角速度
figure
hold on;
plot(result.w(:,1),'-b'); 
plot(result.w(:,2),'-r'); 
plot(result.w(:,3),'-g'); 

%対象物体の姿勢
figure
hold on;
plot(result.xEst(:,1),'-b'); 
plot(result.xEst(:,2),'-r'); 
plot(result.xEst(:,3),'-g'); 

%ライトカーブ
figure
hold on;
plot(result.magnitude,'-b'); 
set(gca,'YDir','reverse');

figure
hold on;
plot(result.alpha,'-g'); 

%剛体のオイラー方程式
function w_dot = f_Eular(t,w)
wx = w(1);
wy = w(2);
wz = w(3);

global Ix Iy Iz;
global Tx Ty Tz;

w_dot = [((Iy-Iz)*wy*wz+Tx)/Ix
    ((Iz-Ix)*wz*wx+Ty)/Iy
    ((Ix-Iy)*wx*wy+Tz)/Iz];

end

function w = fw_LungeKutta(t,w)
%State model (Motion Equation)
global dt

k1 = f_Eular(t,w);
k2 = f_Eular(t+dt/2,w+k1/2);
k3 = f_Eular(t+dt/2,w+k2/2);
k4 = f_Eular(t+dt,w+k3);

w = w + dt * (k1 + 2*k2 + 2*k3 + k4)/6;

end

%姿勢の微分方程式を計算
function D = f_angle(t,x,w)
phi = x(1);
theta = x(2);
psi = x(3);

global a Fw h_orbit
omega = [a*Fw/h_orbit 0 h_orbit/(a*a)]';

R_mat = EularZYX(x);

Q_mat = [-R_mat(1,3) R_mat(1,1) -R_mat(1,2)
        -R_mat(2,3) R_mat(2,1) -R_mat(2,2)
        -R_mat(3,3) R_mat(3,1) -R_mat(3,2)];
    
A = [1 0 -sin(theta)
    0 cos(phi) sin(phi)*cos(theta)
    0 -sin(phi) cos(phi)*cos(theta)];

B = w - Q_mat * omega;
    
%姿勢角の微分方程式
D = inv(A) * B;
end

%姿勢角ルンゲクッタ法
function x = f(t,x,w)
global dt

k1 = f_angle(t,x,w);
k2 = f_angle(t+dt/2,x+k1/2,w);
k3 = f_angle(t+dt/2,x+k2/2,w);
k4 = f_angle(t+dt,x+k3,w);

x = x + dt * (k1 + 2*k2 + 2*k3 + k4)/6;

end

%トルク
%重力傾斜トルク
function Tg = Calc_Tg(x)
global C_tg 
global Ix Iy Iz

phi = x(1);
theta = x(2);
psi = x(3);

Tg = C_tg * [sin(phi)*cos(phi)*cos(theta)*cos(theta)*(Iz-Iy)
    -cos(phi)*sin(theta)*cos(theta)*(Ix-Iz)
    -sin(phi)*sin(theta)*cos(theta)*(Iy-Ix)];

end

%太陽輻射圧
function Fsrp = Calc_Fsrp()
global C_ts



end


%微小面の明るさを計算
function [C,alpha,Sigma,A] = calc_plane(x,pr) %should be updated
phi = x(1);
theta = x(2);
psi = x(3);

global u_obs u_sun u_h;
global altitude;
global s d F0 rho;

%定数
C_sunvis = 455;
n_u = 1000;
n_v = 1000;
n_gamma = 1000;
distance = altitude * 1000;

R_point_mat = EularPointZYX(x);

p = R_point_mat * pr;

u_u = p(:,4) - p(:,3);
u_v = p(:,2) - p(:,3);

Lx = norm(u_u);
Ly = norm(u_v);
A = Lx * Ly;

Un_out = cross(u_u,u_v);
un_out = Un_out/norm(Un_out);

sigma = dot(un_out,u_obs);
alpha = dot(u_h,un_out);

%BRDFの計算
%BRDFの反射項
k1 = ((n_u+1) * (n_v+1)).^(0.5) / (8*pi);
E = dot(u_h,u_sun) * max(dot(un_out,u_obs),dot(un_out,u_sun));
F_reflect = F0 + (1/s - F0) * (1 - dot(u_h,u_sun)).^5;

if (E == 0)
    Rs = inf;
else
    Rs = k1 * (dot(un_out,u_h).^n_gamma) * F_reflect / E;
end

%BRDFの拡散項
k2 = (28 * rho) * (1 - s * F0) / (23 * pi);
Rd = k2 * (1 - (1 - dot(un_out,u_obs)/2).^5) * (1 - (1 - dot(un_out,u_sun)/2).^5);
fr = s * Rs + d * Rd;

F_sun = C_sunvis * dot(un_out,u_sun);
F_obs = (F_sun * fr * A * dot(un_out,u_obs))/(distance.^2);

if (dot(un_out,u_sun) > 0) && (dot(un_out,u_obs) > 0)
    C = abs(F_obs/C_sunvis);
else
    C = 0;
end

if (sigma > 0)
    Sigma = A * sigma;
else
    Sigma = 0;
end
%disp(C)
%disp(alpha)
end

%等級
function magnitude = Magnitude(C)
if (C > 0)
    magnitude = -26.7 - 2.5 * log10(C);
else
    magnitude = inf;
end
end

%Deg-Rad変換
function alpha = RadtoDeg(alpha)
alpha = alpha*180/pi;
end

function alpha = DegtoRad(alpha)
alpha = alpha*pi/180;
end

%座標系回転座標系
function R_mat = EularZYX(x)
phi = x(1);
theta = x(2);
psi = x(3);

R_mat = [cos(theta)*cos(psi) cos(theta)*sin(psi) -sin(theta)
    sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi) sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) sin(phi)*cos(theta)
    cos(phi)*sin(theta)*cos(psi)+sin(phi)*cos(psi) cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi) cos(phi)*cos(theta)]; 

end

%座標点回転座標系
function R_point_mat = EularPointZYX(x)
phi = x(1);
theta = x(2);
psi = x(3);

R_point_mat = [cos(theta)*cos(psi) -cos(theta)*sin(psi) sin(theta)
    sin(phi)*sin(theta)*cos(psi)+cos(phi)*sin(psi) -sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) -sin(phi)*cos(theta)
    -cos(phi)*sin(theta)*cos(psi)+sin(phi)*cos(psi) cos(phi)*sin(theta)*sin(psi)+sin(phi)*cos(psi) cos(phi)*cos(theta)];

end

