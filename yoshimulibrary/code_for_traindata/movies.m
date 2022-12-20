% movies
clc
clear
close all

% add path for functions
addpath('../yoshimuLibrary-main/attitude');
addpath('../yoshimuLibrary-main/conversion');
addpath('../yoshimuLibrary-main/environment');
addpath('../yoshimuLibrary-main/lightcurves/');
addpath('../yoshimuLibrary-main/object/');
addpath('../yoshimuLibrary-main/orbit/');
addpath('../yoshimuLibrary-main/sphericalGaussian/');
addpath('../yoshimuLibrary-main/srp/');
addpath('../yoshimuLibrary-main/time/');
addpath('../yoshimuLibrary-main/ukf/');
addpath('../yoshimuLibrary-main/utility/');
addpath('../yoshimuLibrary-main/vectorMatrix/');
addpath('../WOBJ_toolbox_Version2b/');


%% parameters
% global MOI sat mass const n lam K mu
% %MOI = diag([32.625,73.95,79.935]);    % Moment of Inertia
% MOI = diag([1.416, 1.416, 2.0861]);     %　軸対称衛星
% sat = readSC('oneWebRaw.obj',3);
% mass = 150;     % total mass of sat. [kg]
% const = orbitConst;
% mu = const.GE; % 地心重力定数 [km^3/s^2]
% 
% lam = 1;    % positive parameter
% K = 0.00115497;    % gain, K < 0.5*(1+2sin(xi))*n
% 
% %% initial value
% oeIni = [const.RE+1200,0,deg2rad(0),0,0,0];
% % 初期軌道要素（semimajor axis[km], eccentricity[], inclination[rad],
% %           longitude of ascending node[rad], argument of perigee[rad], true anomaly[rad])
% [r0,v0] = oe2rv(oeIni, 1);     % 位置と速度に変換
% To = 2*pi*norm(r0)/norm(v0); % Orbital period, 6565[s]
% n = 2*pi / To; % Orbit rate
% 
% %w0 = [0;0;n];   %　軌道角速度[rad/sec]
% %w0 = [0;0;0.5235987756];      % z-axis spin(5rpm)
% 
% w0 = [0.0033299; 0.09432; 0.045375];   %　IJK系に対する機体固定座標系の角速度
% 
% q0 = [0;0;0;1];
% % h = cross(r0,v0); % orbital angular momentum w.r.t. inertial frame, nx3
% % N = cross(v0./norm(v0), h./norm(h))';
% % Roi = triad(N, v0'./norm(v0), [1;0;0], [0;1;0]); % DCM from inertial frame to orbital frame
% % qtemp = dcm2q(4,Roi);
% % q0 = qMult(4,1,q0',qtemp);
% 
% x0 = [r0'; v0'; w0; q0];

% %% Equation of Motion etc.
% tic
% 
%  for k = 1:1
%     tspan = 0:0.1:To*5;
% 
%     options = odeset('RelTol',1e-9,'AbsTol',1e-9);
%     %[t, y] = ode45(@EOM, tspan, x0, options);
%     %y = ode4(@EOM, tspan, x0);
%     [t, y] = ode45(@EOM_withcontrol, tspan, x0, options);
% 
%     r = y(:,1:3);
%     v = y(:,4:6);
%     w = y(:,7:9);
%     q = y(:,10:13);
%  
%     bface = zeros(length(y),1);
%     for j = 1:length(y)
% 
%         % 照射面選択
%         sat_.normal = [0 1 0;
%                       0 0 1;
%                       1 0 0;
%                       0 -1 0;
%                       0 0 -1;
%                       -1 0 0];
% 
%         Rib = inv(q2DCM(4,q(j,1:4)));  % Body to IJK
%         NTW2i(1:9) = rv2NTW([r(j,1:3) v(j,1:3)]);   % 位置速度からIJKで表したN,T,Wベクトル
%         Ti = NTW2i(4:6);    % IJKで表したTベクトル
%         Judge = zeros(6,1);
%         Ni = zeros(6,3);
%         for i = 1:6
%             Ni(i,1:3) = (Rib * sat_.normal(i,:)')' ;   % IJK系で表した法線ベクトル
%             Judge(i) = dot(-Ti(1:3),Ni(i,1:3)); % IJK系にて法線ベクトルと-Tベクトル(レーザ入射方向)の内積最小を取る
%         end
%         answer = min(Judge(:));
%         bface(j) = find(Judge == answer);  % 照射面によって機体系における推力およびトルク発生方向が決定
% 
%     end
% 
%     k
% 
% end
% 
% toc
X_test = readmatrix('./train_data/X_boxOneWing001.csv');
X_reg = readmatrix('./train_data/attiRegTimeHIstory.csv');
q_test = X_test(:,1:4); q_reg = X_reg(:,1:4);

% %% Show movie (inertial frame)
% % movie_path = fullfile('/Users/aihayashibara/Documents/MATLAB/movies');
% % str = horzcat(movie_path,'.mp4');
% curdir = pwd;
% savedir = fullfile(curdir, '../../../temporary');
% str = strcat(savedir, '/boxOneWing_test_movie.mp4');
% vid = VideoWriter(str,'MPEG-4');
% open(vid);
% 
% sat = readSC('boxWing.obj',2);
% sat.judge = zeros(length(sat.faces),1);
% 
% figure(1)
% axis equal
% grid on
% xlabel('x'), ylabel('y'), zlabel('z')
% xlim([-3 3]), ylim([-3 3]), zlim([-3 3])
% nFaces = length(sat.faces);
% satFig = patch('Faces',sat.faces,'Vertices',sat.vertices,'FaceVertexCData',zeros(nFaces,1),'FaceColor','flat');
% alpha(0.2);
% hold on
% view([34 13])
% % qui = quiver3(0,0,0,1,0,0,0,'color','r');
% 
% for i = 1:10:length(X_test)
%     qTmp = repmat(qInv(4, q_test(i,:)),size(sat.vertices,1),1);
%     newVertices = qRotation(4, sat.vertices, qTmp);
%     set(satFig, 'Vertices', newVertices);
% %     set(qui,'udata',v(i,1)*2,'vdata',v(i,2)*2,'wdata',v(i,3)*2);
% %    
% %     if bface(i) == 1
% %         tmp = zeros(nFaces,1);
% %         tmp(6) = 1;
% %         tmp(12) = 1;
% %         set(satFig, 'CData', tmp);
% %     elseif bface(i) == 2
% %         tmp = zeros(nFaces,1);
% %         tmp(1) = 1;
% %         tmp(7) = 1;
% %         set(satFig, 'CData', tmp);
% %     elseif bface(i) == 3
% %         tmp = zeros(nFaces,1);
% %         tmp(5) = 1;
% %         tmp(11) = 1;
% %         set(satFig, 'CData', tmp);
% %     elseif bface(i) == 4
% %         tmp = zeros(nFaces,1);
% %         tmp(2) = 1;
% %         tmp(8) = 1;
% %         set(satFig, 'CData', tmp);
% %     elseif bface(i) == 5
% %         tmp = zeros(nFaces,1);
% %         tmp(4) = 1;
% %         tmp(10) = 1;
% %         set(satFig, 'CData', tmp);
% %     else
% %         tmp = zeros(nFaces,1);
% %         tmp(3) = 1;
% %         tmp(9) = 1;CData
% %         set(satFig, 'CData', tmp);
% %     end
% 
%     drawnow
%     frame = getframe(gcf);
%  	writeVideo(vid,frame);
% end
% 
%  close(vid);
% 
 %% Show movie
% movie_path = fullfile('/Users/aihayashibara/Documents/MATLAB/movies/2windows');
% str = horzcat(movie_path,'.mp4');
curdir = pwd;
savedir = strcat(curdir, '/../../../temporary/');
str = strcat(savedir, 'boxOneWing_reg_movie.mp4');
vid = VideoWriter(str,'MPEG-4');
open(vid);

sat = readSC('boxWing.obj',2);
sat.judge = zeros(length(sat.faces),1);

figure(2)

subplot(1,2,1)      % Attitude(NTW) 
axis equal
grid on

xlabel('$x$, m','interpreter','latex')
ylabel('$y$, m','interpreter','latex')
zlabel('$z$, m','interpreter','latex')
xlim([-3 3]), ylim([-3 3]), zlim([-3 3])

nFaces = length(sat.faces);
satFig = patch('Faces',sat.faces,'Vertices',sat.vertices,'FaceVertexCData',zeros(nFaces,1),'FaceColor','flat');
% satFig2 = patch('Faces',sat.faces,'Vertices',sat.vertices,'FaceVertexCData',zeros(nFaces,1),'FaceColor','flat');
alpha(0.2);
hold on
view([35 -15])

% quiver3(0,2,0,0,-2,0,'LineWidth',1,'Color','r')

subplot(1,2,2)      % orbit
axis equal
grid on

xlabel('$X$, m','interpreter','latex')
ylabel('$Y$, m','interpreter','latex')
zlabel('$Z$, m','interpreter','latex')
xlim([-3 3]), ylim([-3 3]), zlim([-3 3])

nFaces = length(sat.faces);
satFig_reg = patch('Faces',sat.faces,'Vertices',sat.vertices,'FaceVertexCData',zeros(nFaces,1),'FaceColor','flat');
% satFig2 = patch('Faces',sat.faces,'Vertices',sat.vertices,'FaceVertexCData',zeros(nFaces,1),'FaceColor','flat');
alpha(0.2);
hold on
view([35 -15])


% drawEarth(0,0.5,const)
% plot3(r(:,1),r(:,2),r(:,3),'LineWidth',1,'Color','k')
% posi = plot3(r(i,1),r(i,2),r(i,3),'Marker','o','MarkerSize',8,'MarkerFaceColor','r');
%posi = plot3(r(i,1),r(i,2),r(i,3),'Marker',satFig2);

for i = 1:10:length(X_test)
    qTmp = repmat(qInv(4, q_test(i,:)),size(sat.vertices,1),1);
    newVertices = qRotation(4, sat.vertices, qTmp);
    set(satFig, 'Vertices', newVertices);

    qTmp_reg = repmat(qInv(4, q_reg(i,:)),size(sat.vertices,1),1);
    newVertices_reg = qRotation(4, sat.vertices, qTmp_reg);
    set(satFig_reg, 'Vertices', newVertices_reg);
%     Rbi = q2DCM(4, q(i,:)); % Direction Cosine Matrix from inertial frame to Body
%     h = cross(r,v); % orbital angular momentum w.r.t. inertial frame, nx3
%     N = cross(v(i,:)./norm(v(i,:)), h(i,:)./norm(h(i,:)))'; % N-axis of NTW frame at inertial frame
%     Roi = triad(N, v(i,:)'./norm(v(i,:)), [1;0;0], [0;1;0]); % DCM from inertial frame to orbital frame
%     Rio = Roi'; % DCM from orbital frame to inertial frame
%     Rbo = Rbi * Rio; % DCM from orbital frame to body-fixed frame
%     qbo = dcm2q(4, Rbo);
% 
%     qTmp = repmat(qInv(4, qbo),size(sat.vertices,1),1);
%     newVertices = qRotation(4, sat.vertices, qTmp);
%     set(satFig, 'Vertices', newVertices);

%     qTmp2 = repmat(qInv(4, q(i,:)),size(sat.vertices,1),1);
%     newVertices = qRotation(4, sat.vertices, qTmp2);
%     set(satFig2, 'Vertices', newVertices);
% 
%     if bface(i) == 1
%         tmp = zeros(nFaces,1);
%         tmp(6) = 1;
%         tmp(12) = 1;
%         set(satFig, 'CData', tmp);
% 
%     elseif bface(i) == 2
%         tmp = zeros(nFaces,1);
%         tmp(1) = 1;
%         tmp(7) = 1;
%         set(satFig, 'CData', tmp);
%         
%     elseif bface(i) == 3
%         tmp = zeros(nFaces,1);
%         tmp(5) = 1;
%         tmp(11) = 1;
%         set(satFig, 'CData', tmp);
%         
%     elseif bface(i) == 4
%         tmp = zeros(nFaces,1);
%         tmp(2) = 1;
%         tmp(8) = 1;
%         set(satFig, 'CData', tmp);
%         
%     elseif bface(i) == 5
%         tmp = zeros(nFaces,1);
%         tmp(4) = 1;
%         tmp(10) = 1;
%         set(satFig, 'CData', tmp);
%         
%     else
%         tmp = zeros(nFaces,1);
%         tmp(3) = 1;
%         tmp(9) = 1;
%         set(satFig, 'CData', tmp);
%         
%     end
% 
%     set(posi,'xdata',r(i,1),'ydata',r(i,2),'zdata',r(i,3));
    
    drawnow
    frame = getframe(gcf);
    writeVideo(vid,frame);
end

close(vid);


