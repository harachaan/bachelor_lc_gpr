% 回帰された姿勢(a priori state estimation)からライトカーブを回帰するモデル作り(one-step-ahead prediction)
clc
clear
close all

addpath('hara_functions/');
% -------------------------------------------------------------------------
% kernel parameters
tau = log(1);
sigma = log(1);
eta = log(0.1);
params = [tau sigma eta];

Ntraindata = 1; % 学習データを何個読み込むか

% -------------------------------------------------------------------------
% 学習データ読み込み
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

% -------------------------------------------------------------------------
% データ整理
xtrain = [q2zyx_h(X(:,1:4)) X(:,5:7)]; % 学習データの入力 (Euler angleに変換)
xtest = [q2zyx_h(X_test(:,1:4)) X_test(:,5:7)]; % テストデータの入力 (Euler angleに変換)

Lx = length(xtrain(1,:)); % 入力ベクトルの次元
Ly = 1; % 出力はライトカーブなので1次元
Ntrain = length(xtrain); Ntest = length(xtest); % 今回扱う学習，テストデータセットの行数







% -------------------------------------------------------------------------
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