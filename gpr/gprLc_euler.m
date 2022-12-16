% 回帰された姿勢(a priori state estimation)からライトカーブを回帰するモデル作り(one-step-ahead prediction)
clc
clear
close all

addpath('hara_functions/');
% -------------------------------------------------------------------------
% kernel parameters
tau = log(0.5);
sigma = log(1);
eta = log(0.1);
params = [tau sigma eta];

Ntraindata = 1; % 学習データを何個読み込むか

% 学習データ読み込み----------------------------------------------------------
X = []; t_mApp = [];
for i = 1:1:Ntraindata
    % flat plate の学習データ
    filename = strcat('train_data_using_yoshimulibrary/X_flatPlate', sprintf('%03d', i), '.csv');
    df = readmatrix(filename);
    X = [X; df]; % この場合の事前割り当てのやり方わかんない
    filename = strcat('train_data_using_yoshimulibrary/t_mApp_flatPlate', sprintf('%03d', i), '.csv');
    df = readmatrix(filename);
    t_mApp = [t_mApp; df];
    % box wing の学習データ
    filename = strcat('train_data_using_yoshimulibrary/X_boxWing', sprintf('%03d', i), '.csv');
    df = readmatrix(filename);
    X = [X; df]; 
    filename = strcat('train_data_using_yoshimulibrary/t_mApp_boxWing', sprintf('%03d', i), '.csv');
    df = readmatrix(filename);
    t_mApp = [t_mApp; df];
end
% テストデータ読み込み
X_test = readmatrix('train_data_using_yoshimulibrary/X_boxOneWing001.csv'); 
t_mApp_test = readmatrix('train_data_using_yoshimulibrary/t_mApp_boxOneWing001.csv');

% データ整理-----------------------------------------------------------------
xtrain = [q2zyx_h(X(:,1:4)) X(:,5:7)]; % 学習データの入力 (Euler angleに変換)
xtest = [q2zyx_h(X_test(:,1:4)) X_test(:,5:7)]; % テストデータの入力 (Euler angleに変換)

Lx = length(xtrain(1,:)); % 入力ベクトルの次元
Ly = 1; % 出力はライトカーブなので1次元
Ntrain = length(xtrain); Ntest = length(xtest); % 今回扱う学習，テストデータセットの行数

ytrain = t_mApp(1:Ntrain,2); % 学習データの出力
ytest = t_mApp_test(1:Ntest,2); % テストデータの出力
t_test = t_mApp_test(1:Ntest,1); % テストデータの時間（横軸）

% lightcurveがinfになってる行をデータセットから消す
Do = [xtrain ytrain]; Do_test = [xtest ytest t_test];
Do(any(isinf(Do)'),:) = []; Do_test(any(isinf(Do_test)'),:) = [];
xtrain = Do(:,1:6); ytrain = Do(:,7); 
xtest = Do_test(:,1:6); ytest = Do_test(:,7);
t_test = Do_test(:,8);
Ntrain = length(xtrain(:,1)); Ntest = length(xtest(:,1)); % NtrainとNtestが変化したので再代入

% 学習データとテストデータの出力の平均を0にする
for i = 1:1:Ly
    ytrain(:,i) = ytrain(:,i) - mean(ytrain(:,i));
    ytest(:,i) = ytest(:,i) - mean(ytest(:,i));
end

% 回帰の計算-----------------------------------------------------------------
tic;
xx = xtest;
yy_mu = zeros(Ntest, Ly); yy_var = zeros(Ntest, Ly);
for i = 1:1:Ly
    regression = gpr(xx, xtrain, ytrain(:,i), params);
    yy_mu(:,i) = regression(:,1); yy_var(:,i) = regression(:,2);
end
two_sigma1 = yy_mu - 2 * sqrt(yy_var); two_sigma2 = yy_mu + 2 * sqrt(yy_var);
tEnd = toc;

mAppReg = yy_mu(:,Ly);

% plot---------------------------------------------------------------------
f1 = figure;

figure(f1);
plot(t_test, ytest, 'r.'); % 真値
hold on;
plot(t_test, mAppReg, 'b.'); % 回帰結果
filename = "LightCurves"; savename = strcat("figures/Lc/", filename, ".png");
title(filename);
saveas(gcf, savename);

% figure(f1);
% plot(t_test, ytrain(1:Ntest,1), 'k.');
% hold on;
% plot(t_test, ytrain((Ntrain-1)/2 : ((Ntrain-1)/2 + Ntest -1), 1), 'g.');
% hold on;
% plot(t_test, ytest, 'r.'); % 真値
% hold on;
% plot(t_test, mAppReg, 'b.'); % 回帰結果
% filename = "LightCurves"; savename = strcat("figures/Lc/", filename, ".png");
% title(filename);
% saveas(gcf, savename);

% -------------------------------------------------------------------------
% ガウス過程回帰の関数たち
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

% ある入力xvectorに対するk*を作る
function kv = kv(x, xtrain, params)
    kv = zeros(length(xtrain), 1);
    for i = 1:1:length(xtrain)
        kv(i,1) = gaussian_kernel(x, xtrain(i,:), params, false);
    end
end

% カーネル行列Kを作る関数
function K = kernel_matrix(xx, params)
    N = length(xx);
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
% xxはテストデータの入力
function y = gpr(xx, xtrain, ytrain, params)
    K = kernel_matrix(xtrain, params); % 学習データの入力xで作るカーネル行列K
    Kinv = inv(K);
    N = length(xx);
    mu = zeros(N, 1); var = zeros(N, 1);
    for i = 1:1:N
        s = gaussian_kernel(xx(i,:), xx(i,:), params, false, true); % カーネル行列k_**
        k = kv(xx(i,:), xtrain, params); % 縦ベクトル (回帰する状態のカーネル行列k_*)
        mu(i,1) = k' * Kinv * ytrain;
        var(i,1) = s - k' * Kinv * k;
    end
    y = [mu var]; 
end