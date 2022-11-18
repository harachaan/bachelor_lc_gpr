% 姿勢，軌道を状態変数(変数７個？)に持つGPR
clc
clear
close all
dir = pwd;
% addpath(strcat(dir, ''))

% kernel parameters
tau = log(1);
sigma = log(1);
eta = log(1);
params = [tau sigma eta];

% 学習データ読み込み
% (from yoshimulibrary...)
D_p = readmatrix('train_data_using_yoshimulibrary/D_p001.csv');
t_mApp = readmatrix('train_data_using_yoshimulibrary/t_mApp001.csv');

q0_train = D_p(:,1,:); q1_train = D_p(:,2,:); q2_train = D_p(:,3,:); q3_train = D_p(:,4,:);
w1_train = D_p(:,5,:); w2_train = D_p(:,6,:); w3_train = D_p(:,7,:);
delta_q0_train = D_p(:,1,:); delta_q1_train = D_p(:,2,:); delta_q2_train = D_p(:,3,:); delta_q3_train = D_p(:,4,:);
delta_w1_train = D_p(:,5,:); delta_w2_train = D_p(:,6,:); delta_w3_train = D_p(:,7,:);

t_train = t_mApp(1:(length(t_mApp) - 1), 1);

% 確め計算ゾーン ------------------------------------------------------------

% -------------------------------------------------------------------------

% カーネル行列のハイパーパラメータ推定
% params = optimize1(params, xtrain, ytrain);

% 回帰の計算


% plot
f1 = figure; % f2 = figure;

figure(f1);
plot(t_train, q0_train, 'ko');
title("q0");

% figure(f2);
% for i = 1:1:length(t_train(1,1,:))
%     plot(t_train(:,:,i), q3_train(:,:,i), 'bo');
%     hold on;
% end
% title("q3");



% -------------------------------------------------------------------------
% 図をプロットする関数を作ろうとしたけどよくわからん買った．
% function fig = plot_drawline(f_num, xtrain, ytrain, xx, y)
%     fig = figure;
%     figure(f_num); 
% end

% ガウスカーネル(7次元)
function kernel = gaussian_kernel(x, y, params, train)
    arguments
        x; y; params; train = true; 
    end
    tau = params(1,1); sigma = params(1,2); eta = params(1,3);
    % 無名関数
    kgauss = @(x, y) exp(tau) * exp(-(x - y)^2 / (exp(sigma)));
    if train == true && x == y
        kernel = kgauss(x, y) + exp(eta);
    else
        kernel = kgauss(x, y);
    end
end

% ある入力x(7次元)に対するk*を作る
% 第2引数xtrainは学習データD_p(:, 1:7)のこと？
function kv = kv(x, xtrain, params)
    kv = zeros(length(xtrain), 1);
    for i = 1:1:length(xtrain)
        kv(i,1)
    end
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



















