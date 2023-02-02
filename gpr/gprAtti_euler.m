% 姿勢(Euler angle)，軌道を状態変数(変数6個)に持つGPR
clc
clear
close all
% -------------------------------------------------------------------------

curdir = pwd;
% addpath(strcat(dir, ''))

addpath('../hara_functions/');

% kernel parameters
tau = log(1);
sigma = log(0.3);
eta = log(0.1);
params = [tau sigma eta];


% 学習データ読み込み---------------------------------------------------------
Ntraindata = 26;
X = []; t_mApp = [];
for i = 1:1:Ntraindata
    % flat plate の学習データ
%     filename = strcat('train_data_using_yoshimulibrary/X_flatPlate', sprintf('%03d', i), '.csv');
%     df = readmatrix(filename);
%     X = [X; df]; % この場合の事前割り当てのやり方わかんない
%     filename = strcat('train_data_using_yoshimulibrary/t_mApp_flatPlate', sprintf('%03d', i), '.csv');
%     df = readmatrix(filename);
%     t_mApp = [t_mApp; df];
    % box wing の学習データ
    filename = strcat('train_data_using_yoshimulibrary/X_boxWing', sprintf('%03d', i), '.csv');
    df = readmatrix(filename);
    X = [X; df]; 
    filename = strcat('train_data_using_yoshimulibrary/t_mApp_boxWing', sprintf('%03d', i), '.csv');
    df = readmatrix(filename);
    t_mApp = [t_mApp; df];
end
% X = readmatrix('train_data_using_yoshimulibrary/X_boxWing002.csv'); 
% t_mApp = readmatrix('train_data_using_yoshimulibrary/t_mApp_boxWing002.csv');

% テストデータ読み込み
% X_test = readmatrix('train_data_using_yoshimulibrary/X_boxOneWing001.csv'); 
% t_mApp_test = readmatrix('train_data_using_yoshimulibrary/t_mApp_boxOneWing001.csv');
X_test = readmatrix('train_data_using_yoshimulibrary/X_boxWing029.csv'); 
t_mApp_test = readmatrix('train_data_using_yoshimulibrary/t_mApp_boxWing029.csv');
% X_test = readmatrix('train_data_using_yoshimulibrary/X_flatPlate001.csv'); 
% t_mApp_test = readmatrix('train_data_using_yoshimulibrary/t_mApp_flatPlate001.csv');


xtrain = [q2zyx_h(X(:,1:4)) X(:,5:7)]; % 学習データの入力 (Euler angleに変換)
xtest = [q2zyx_h(X_test(:,1:4)) X_test(:,5:7)]; % テストデータの入力 (Euler angleに変換)

% 学習データ，テストデータの出力（姿勢，ライトカーブ）
Lx = size(xtrain,2); Ly = Lx + 1; % 出力の次元はライトカーブで1増える
Ntrain = size(xtrain,1) - 1; Ntest = size(xtest,1) - 1; %今回扱う学習データの次元とデータ数(出力が差分だから-1)
ytrain = zeros(Ntrain, Ly); ytest = zeros(Ntest, Ly);
for i = 1:1:(Ntrain)
    ytrain(i,1:6) = xtrain(i+1,1:6) - xtrain(i,1:6);
%     ytrain(i,8) = t_mApp(i+1,2) - t_mApp(i,2);
    ytrain(i,7) = t_mApp(i,2); % ライトカーブは差分じゃなくそのままを学習するようにした
end
for i = 1:1:Ntest
    ytest(i,1:6) = xtest(i+1,1:6) - xtest(i,1:6);
%     ytest(i,8) = t_mApp_test(i+1,2) - t_mApp_test(i,2);
    ytest(i,7) = t_mApp_test(i,2);
end
% 学習データとテストデータの出力は作れたので，各データセットの入力と出力のサイズを統一する（出力の方が1小さいので）
xtrain = xtrain(1:Ntrain,:); xtest = xtest(1:Ntest,:);

% infの行を消し去りたい
Dp = [xtrain ytrain]; Dp_test = [xtest ytest]; % データセットごとinfの行を消すために一回入力と出力を結合する
Dp(any(isinf(Dp)'),:) = []; Dp_test(any(isinf(Dp_test)'),:) = [];% やっぱりinfのある行は取り除いた
xtrain = Dp(:,1:6); ytrain = Dp(:,7:13); xtest = Dp_test(:,1:6); ytest = Dp_test(:,7:13); 

Ntrain = size(xtrain,1); Ntest = size(xtest,1);% NtrainとNtestが変化したので再代入
t_mApp_test = t_mApp_test(1:Ntest,:); t_test = t_mApp_test(1:Ntest, 1); % プロットするためにt_testのサイズを調整

% 学習データとテストデータの出力の平均を0にする
mean_ytrain = zeros(1,Ly); mean_ytest = zeros(1,Ly);
for i = 1:1:Ly
    mean_ytrain(1,i) = mean(ytrain(:,i));
    mean_ytest(1,i) = mean(ytest(:,i));
    ytrain(:,i) = ytrain(:,i) - mean_ytrain(1,i);
    ytest(:,i) = ytest(:,i) - mean_ytest(1,i);
end

% 確め計算ゾーン ------------------------------------------------------------
% gaussian_kernel(xtrain(2,:), xtrain(3,:), params);
% kv(xtrain(2,:), xtrain, params);
% kernel_matrix()

% -------------------------------------------------------------------------

% カーネル行列のハイパーパラメータ推定
% params = optimize1(params, xtrain, ytrain);

tic
% 回帰の計算
% Ly = Ly;
xx = xtest;
yy_mu = zeros(Ntest, Ly); yy_var = zeros(Ntest, Ly);
% a = gpr(xx, xtrain, ytrain(:,8), params);
for i = 1:1:Ly
    regression = gpr(xx, xtrain, ytrain(:,i), params); % length(xx)行2列？
    yy_mu(:,i) = regression(:,1); yy_var(:,i) = regression(:,2); 
    i
end

two_sigma1 = yy_mu - 2 * sqrt(yy_var); two_sigma2 = yy_mu + 2 * sqrt(yy_var);
tEnd = toc; % gpr７回にかかる時間

% 平均0にしたのをもどす（とりあえず単純に）
mAppReg = yy_mu(:,Ly) + mean_ytrain(1,Ly);
ytest(:,Ly) = ytest(:,Ly) + mean_ytest(1,Ly);
for i = 1:1:Lx
    yy_mu(:,i) = yy_mu(:,i) + mean_ytrain(1,i);
    ytest(:,i) = ytest(:,i) + mean_ytest(1,i);
end
% 時系列順の姿勢履歴にorganize -----------------------------------------------

% eluler角で計算
% attiIni = xx(1,:); mAppIni = ytest(1,Ly);
% attiReg = zeros(Ntest, Lx); attiReg(1,:) = attiIni; 
% for i = 1:1:(Ntest-1)
%     % angle and anglar velocity
%     attiReg(i+1,:) = attiReg(i,:) + yy_mu(i,1:6);
% end
attiReg = zeros(Ntest, Lx); attiReg(1,:) = xx(1,:);
attiReg(2:end,:) = xtest(1:Ntest-1,:) + yy_mu(1:Ntest-1,1:6); % 瞬間瞬間の差分を計算してその妥当性を見たいので，xtestにattiRegを足す．

% 差分計算までeulerでやって，最後にquaternion変換を挟むことで2piの制限を考慮させる作戦
attiReg = [zyx2q_h(attiReg(:,1), attiReg(:,2), attiReg(:,3)), attiReg(:,4:6)];
attiReg = [q2zyx_h(attiReg(:,1:4)), attiReg(:,5:7)];

% 瞬間瞬間の回帰結果とテストデータの誤差
attiError = attiReg - xtest; mean_attiError = mean(attiError, 1);
mAppError = mAppReg - ytest(:,Ly); mean_mAppError = mean(mAppError, 1);


% % quaternionで差分計算して最後にeulerに戻す．
% yy_mu = [zyx2q_h(yy_mu(:,1), yy_mu(:,2), yy_mu(:,3)), yy_mu(:,4:7)]; % euler angleの差分をquaternion error に変換
% attiIni = [zyx2q_h(xx(1,1), xx(1,2), xx(1,3)), xx(1,4:6)];
% attiReg = zeros(Ntest, size(attiIni,2)); attiReg(1,:) = attiIni;
% for i = 1:1:Ntest-1
%     attiReg(i+1,1:4) = q_pro(attiReg(i,1:4)', yy_mu(i,1:4)')'; % quaternion 転置に注意
%     attiReg(i+1,5:7) = attiReg(i,5:7) + yy_mu(i,5:7); % anglar velocity 
% end
% attiReg = [q2zyx_h(attiReg(:,1:4)), attiReg(:,5:7)];


% csvファイルに書き出し ------------------------------------------------------
savedir = strcat(curdir, '/../../temporary/X_gpr/');
savename = strcat(curdir, 'test_x_t_mApp.csv'); writematrix([xtest, t_mApp_test], savename);
savename = strcat(savedir, 'attiRegTimeHIstory.csv'); writematrix(attiReg, savename);
savename = strcat(savedir, 'mAppReg.csv'); writematrix(mAppReg, savename);
savename = strcat(savedir, 'errors.csv'); writematrix([attiError, mAppError], savename);
savename = strcat(savedir, 'meanErrors.csv'); writematrix([mean_attiError, mean_mAppError], savename);
% savedir = strcat(curdir, '/results/');
% savename = strcat(savedir, 'attiRegTimeHIstory.csv');
% writematrix(attiReg, savename);

% plot --------------------------------------------------------------------
f1 = figure; f2 = figure; f3 = figure; f4 = figure; f5 = figure; f6 = figure; f7 = figure;
f8 = figure; f9 = figure; f10 = figure; f11 = figure; f12 = figure; f13 = figure; 
% f15 = figure;
savedir = strcat(curdir, '/../../temporary/X_gpr/');
figure(f1);
% patch([xx(:,1)', fliplr(xx(:,1)')], [two_sigma1', fliplr(two_sigma2')], 'c');
% hold on;
% plot(xtrain(:,1), ytrain(:,1), 'k.'); % 学習データ
% hold on;
plot(xtest(:,1), ytest(:,1), 'r.'); % 真値？
hold on;
plot(attiReg(:,1), yy_mu(:,1), 'b.'); % 回帰結果？
filename = "deltaPhi"; savename = strcat(savedir, filename, ".pdf");
title(filename);
exportgraphics(gcf, savename);

figure(f2);
% plot(xtrain(:,2), ytrain(:,2), 'k.'); % 学習データ
% hold on;
plot(xtest(:,2), ytest(:,2), 'r.'); % 真値？
hold on;
plot(attiReg(:,2), yy_mu(:,2), 'b.'); % 回帰結果？
filename = "deltaTheta"; savename = strcat(savedir, filename, ".pdf");
title(filename);
exportgraphics(gcf, savename);

figure(f3);
% plot(xtrain(:,3), ytrain(:,3), 'k.'); % 学習データ
% hold on;
plot(xtest(:,3), ytest(:,3), 'r.'); % 真値？
hold on;
plot(attiReg(:,3), yy_mu(:,3), 'b.'); % 回帰結果？
filename = "deltaPsi"; savename = strcat(savedir, filename, ".pdf");
title(filename);
exportgraphics(gcf, savename);

figure(f4);
% plot(xtrain(:,4), ytrain(:,4), 'k.'); % 学習データ
% hold on;
plot(xtest(:,4), ytest(:,4), 'r.'); % 真値？
hold on;
plot(attiReg(:,4), yy_mu(:,4), 'b.'); % 回帰結果？
filename = "deltaW1"; savename = strcat(savedir, filename, ".pdf");
title(filename);
exportgraphics(gcf, savename);

figure(f5);
% plot(xtrain(:,5), ytrain(:,5), 'k.'); % 学習データ
% hold on;
plot(xtest(:,5), ytest(:,5), 'r.'); % 真値？
hold on;
plot(attiReg(:,5), yy_mu(:,5), 'b.'); % 回帰結果？
filename = "deltaW2"; savename = strcat(savedir, filename, ".pdf");
title(filename);
exportgraphics(gcf, savename);

figure(f6);
% plot(xtrain(:,6), ytrain(:,6), 'k.'); % 学習データ
% hold on;
plot(xtest(:,6), ytest(:,6), 'r.'); % 真値？
hold on;
plot(attiReg(:,6), yy_mu(:,6), 'b.'); % 回帰結果？
filename = "deltaW3"; savename = strcat(savedir, filename, ".pdf");
title(filename);
exportgraphics(gcf, savename);

figure(f7);
plot(t_test, xtest(:,1), 'r.'); 
hold on;
plot(t_test, attiReg(:,1), 'b.');
hold on;
filename = "phiTimeHistory"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\phi [rad]'); 
exportgraphics(gcf, savename);

figure(f8);
plot(t_test, xtest(:,2), 'r.'); 
hold on;
plot(t_test, attiReg(:,2), 'b.');
hold on;
filename = "thetaTimeHistory"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\theta [rad]'); 
exportgraphics(gcf, savename);

figure(f9);
plot(t_test, xtest(:,3), 'r.'); 
hold on;
plot(t_test, attiReg(:,3), 'b.');
hold on;
filename = "psiTimeHistory"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\psi [rad]'); 
exportgraphics(gcf, savename);

figure(f10);
plot(t_test, xtest(:,4), 'r.'); 
hold on;
plot(t_test, attiReg(:,4), 'b.');
hold on;
filename = "omega1TimeHistory"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\omega_1 [rad/s]'); 
exportgraphics(gcf, savename);

figure(f11);
plot(t_test, xtest(:,5), 'r.'); 
hold on;
plot(t_test, attiReg(:,5), 'b.');
hold on;
filename = "omega2TimeHistory"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\omega_2 [rad/s]');
exportgraphics(gcf, savename);

figure(f12);
plot(t_test, xtest(:,6), 'r.'); 
hold on;
plot(t_test, attiReg(:,6), 'b.');
hold on;
filename = "omega3TimeHistory"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\omega_3 [rad/s]');
exportgraphics(gcf, savename);

figure(f13);
plot(t_test, ytest(:,7), 'r.'); 
hold on;
plot(t_test, mAppReg(:,1), 'b.');
hold on;
filename = "lightcurves"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('magnitude');
exportgraphics(gcf, savename);

f14 = figure; figure(f14);
plot(t_test, attiError(:,1), 'b.');
hold on;
filename = "phiError"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\phi [rad]'); 
exportgraphics(gcf, savename);

f15 = figure; figure(f15);
plot(t_test, attiError(:,2), 'b.');
hold on;
filename = "thetaError"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\theta [rad]'); 
exportgraphics(gcf, savename);

f16 = figure; figure(f16);
plot(t_test, attiError(:,3), 'b.');
hold on;
filename = "psiError"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\psi [rad]'); 
exportgraphics(gcf, savename);

f17 = figure; figure(f17);
plot(t_test, attiError(:,4), 'b.');
hold on;
filename = "omega1Error"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\omega_1 [rad/s]'); 
exportgraphics(gcf, savename);

f18 = figure; figure(f18);
plot(t_test, attiError(:,5), 'b.');
hold on;
filename = "omega2Error"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\omega_2 [rad/s]'); 
exportgraphics(gcf, savename);

f19 = figure; figure(f19);
plot(t_test, attiError(:,6), 'b.');
hold on;
filename = "omega3Error"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('\omega_3 [rad/s]'); 
exportgraphics(gcf, savename);

f20 = figure; figure(f20);
plot(t_test, mAppError(:,1), 'b.');
hold on;
filename = "lightcurvesError"; savename = strcat(savedir, filename, ".pdf");
title(filename);
xlabel('time [s]'); ylabel('magnitude'); 
exportgraphics(gcf, savename);

% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% 図をプロットする関数を作ろうとしたけどよくわからん買った．
% function fig = plot_drawline(f_num, xtrain, ytrain, xx, y)
%     fig = figure;
%     figure(f_num); 
% end

% ガウスカーネル(7次元の入力だけど，出力は1次元？)
function kernel = gaussian_kernel(x, y, params, train, diag)
    arguments
        x; y; params; train = true; diag = false;
    end
    tau = params(1,1); sigma = params(1,2); eta = params(1,3);
    % 無名関数
    kgauss = @(x, y) exp(tau) * exp(-norm(x - y) ^2 / (exp(sigma)));
    if train == true &&  diag == true
        kernel = kgauss(x, y) + exp(eta);
    else
        kernel = kgauss(x, y);
    end
end

% ある入力x(7次元)に対するk*を作る
% 第2引数xtrainは学習データDp(:, 1:7)のこと？
function kv = kv(x, xtrain, params)
    kv = zeros(size(xtrain,1), 1);
    for i = 1:1:size(xtrain,1)
        kv(i,1) = gaussian_kernel(x, xtrain(i,:), params, false);
    end
end

% カーネル行列Kを作る関数
function K = kernel_matrix(xx, params)
    N = size(xx,1);
    K = zeros(N, N);
    for i = 1:1:N
        for j = 1:1:N
            if i == j
                K(i,j) = gaussian_kernel(xx(i,:), xx(j,:), params, true, true);
            else
                K(i,j) = gaussian_kernel(xx(i,:), xx(j,:), params); % あらかじめtrain=true, diag=false
            end
        end
    end
end

% ガウス過程回帰を行う
% xxに何が入るかわからん → 学習データ以外の姿勢？
% xtrainは7次元の入力，ytrainは各姿勢データ1次元の出力(差分)
function y = gpr(xx, xtrain, ytrain, params)
    K = kernel_matrix(xtrain, params); % 学習データの入力Dp(:, 1:7)のカーネル行列K
    Kinv = inv(K);
%     N = length(xx) + length(xtrain); % xxは学習データ以外の姿勢? 学習データと完全に被らなければ足すだけでいい，のか？
    N = size(xx,1);
%     L = length(xtrain(1,:));
    mu = zeros(N, 1); var = zeros(N, 1);
    for i = 1:1:N
        s = gaussian_kernel(xx(i,:), xx(i,:), params, false, true); % カーネル行列k_**
        k = kv(xx(i,:), xtrain, params); % 縦ベクトル (回帰する状態のカーネル行列k_*)
        mu(i,1) = k' * Kinv * ytrain;
        var(i,1) = s - k' * Kinv * k;
    end
    y = [mu var]; % 入力された姿勢の次の差分の確率分布が回帰された？
end

% train dataのoutputの平均を0と仮定せずに考慮に入れたgpr
function y = gpr2(xx, xtrain, ytrain, params)
    K = kernel_matrix(xtrain, params); % 学習データの入力Dp(:, 1:7)のカーネル行列K
    N = size(xx,1);
    Kinv2 = inv(K + params(1,3)^2 * eye(N)); % alphaのための逆行列
    m = zeros(N, 1); var = zeros(N, 1);
    for i = 1:1:N
        s = gaussian_kernel(xx(i,:), xx(i,:), params, false, true); % カーネル行列k_**
        k = kv(xx(i,:), xtrain, params); % 縦ベクトル (回帰する状態のカーネル行列k_*)
        mu = mean(ytrain); % 学習データの出力の平均
        alpha = Kinv2 * (ytrain - mu); % kvの重み？
        m(i,1) = 0 + alpha' * k; % predictive mean
        var(i,1) = s + params(1,3)^2 - k' * Kinv2 * k; % predictive variance
    end
    y = [m var]; % 入力された姿勢の次の差分の確率分布が回帰された？
end


% 
% % ハイパーパラメータに対する，式(3.92)の勾配
% function kgrad = kgauss_grad(xi, xj, d, params)
%     if d == 1
%         kgrad = gaussian_kernel(xi, xj, params, false);
%     elseif d == 2
%         kgrad = gaussian_kernel(xi, xj, params, false) * (xi - xj) * (xi - xj) / exp(params(1,d));
%     elseif d == 3
%         if xi == xj
%             kgrad = exp(params(1,d));
%         else
%             kgrad = 0;
%         end
%     end
% end



















