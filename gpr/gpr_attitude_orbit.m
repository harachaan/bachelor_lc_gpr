% 姿勢，軌道を状態変数(変数７個？)に持つGPR
clc
clear
close all
dir = pwd;
addpath(strcat(dir, '/simple_plate'))

% kernel parameters
tau = log(1);
sigma = log(1);
eta = log(1);
params = [tau sigma eta];

% 学習データ読み込み
% (平板のモデルに対して，とりあえず乱数を足して作った１１個のデータ)
sample = readmatrix('simple_plate_0.csv', NumHeaderLines=1);
data = zeros(length(sample(:,1)), length(sample(1, :)), 11); % 事前割り当て
for i = 0:1:10
    filename = strcat('simple_plate_', num2str(i), '.csv');
    data(:,:,i+1) = readmatrix(filename, NumHeaderLines=1);
end
t_train = data(:,1,:); 
q0_train = data(:,2,:); q1_train = data(:,3,:); q2_train = data(:,4,:); q3_train = data(:,5,:);
x_cowell_train = data(:,6,:); y_cowell_train = data(:,7,:); z_cowell_train = data(:,8,:);
% ある時間に対する状態を縦ベクトルで表示
y_train = zeros(length(data(1,2:8,1)), length(data(:,1,1)), length(data(1,1,:))); % 転置を既に考慮していることに注意
for i = 1:1:length(data(1,1,:))
    y_train(:,:,i) = data(:,2:8,i)';
end
% 確め計算ゾーン ------------------------------------------------------------

% -------------------------------------------------------------------------

% カーネル行列のハイパーパラメータ推定
% params = optimize1(params, xtrain, ytrain);

% 回帰の計算


% plot
f1 = figure; f2 = figure;

figure(f1);
for i = 1:1:length(t_train(1,1,:))
    plot(t_train(:,:,i), q0_train(:,:,i), 'ko');
    hold on;
end
title("q0");

figure(f2);
for i = 1:1:length(t_train(1,1,:))
    plot(t_train(:,:,i), q3_train(:,:,i), 'bo');
    hold on;
end
title("q3");



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

% ある入力x(7次元)に対する
function kv = kv(x, xtrain, params)
    kv = zeros(length(xtrain), 1);
    for i = 1:1:length(xtrain)
        kv(i,1)
    end
end

% ハイパーパラメータに対する，式(3.92)の勾配
function kgrad = kgauss_grad(xi, xj, d, params)
    if d == 1
        kgrad = gaussian_kernel(xi, xj, params, false);
    elseif d == 2
        kgrad = gaussian_kernel(xi, xj, params, false) * (xi - xj) * (xi - xj) / exp(params(1,d));
    elseif d == 3
        if xi == xj
            kgrad = exp(params(1,d));
        else
            kgrad = 0;
        end
    end
end



















