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

% -------------------------------------------------------------------------
% 学習データ読み込み
% (from yoshimulibrary...)
Dp = readmatrix('train_data_using_yoshimulibrary/Dp_flatPlate002.csv');
t_mApp = readmatrix('train_data_using_yoshimulibrary/t_mApp_flatPlate002.csv');
% for i = 3:1:5
%     % flat plate の学習データ
%     filename = strcat('train_data_using_yoshimulibrary/Dp_flatPlate', sprintf('%03d', i), '.csv');
%     df = readmatrix(filename);
%     Dp = [Dp; df]; % この場合の事前割り当てのやり方わかんない
%     filename = strcat('train_data_using_yoshimulibrary/t_mApp_flatPlate', sprintf('%03d', i), '.csv');
%     df = readmatrix(filename);
%     t_mApp = [t_mApp; df];
%     % box wing の学習データ
%     filename = strcat('train_data_using_yoshimulibrary/Dp_boxWing', sprintf('%03d', i), '.csv');
%     df = readmatrix(filename);
%     Dp = [Dp; df]; 
%     filename = strcat('train_data_using_yoshimulibrary/t_mApp_boxWing', sprintf('%03d', i), '.csv');
%     df = readmatrix(filename);
%     t_mApp = [t_mApp; df];
% end
% 入力の学習データ
xtrain = Dp(:,1:7);
% q1_train = Dp(:,1,:); q2_train = Dp(:,2,:); q3_train = Dp(:,3,:); q4_train = Dp(:,4,:);
% w1_train = Dp(:,5,:); w2_train = Dp(:,6,:); w3_train = Dp(:,7,:);
% 出力の学習データ
ytrain = [Dp(:,8:14), t_mApp(:,3)];
% delta_q1_train = Dp(:,1,:); delta_q2_train = Dp(:,2,:); delta_q3_train = Dp(:,3,:); delta_q4_train = Dp(:,4,:);
% delta_w1_train = Dp(:,5,:); delta_w2_train = Dp(:,6,:); delta_w3_train = Dp(:,7,:);

t_train = t_mApp(1:(length(t_mApp)), 1);

% 確め計算ゾーン ------------------------------------------------------------
gaussian_kernel(xtrain(2,:), xtrain(3,:), params);
kv(xtrain(2,:), xtrain, params);
% kernel_matrix()

% -------------------------------------------------------------------------

% カーネル行列のハイパーパラメータ推定
% params = optimize1(params, xtrain, ytrain);

% 回帰の計算
% Dp_test = readmatrix('train_data_using_yoshimulibrary/Dp_flatPlate001.csv'); 
% t_mApp_test = readmatrix('train_data_using_yoshimulibrary/t_mApp_flatPlate001.csv');
Dp_test = readmatrix('train_data_using_yoshimulibrary/Dp_boxOneWing001.csv'); 
t_mApp_test = readmatrix('train_data_using_yoshimulibrary/t_mApp_boxOneWing001.csv');
t_test = t_mApp_test(1:length(t_mApp_test), 1);
xx = [Dp_test(:, 1:7)];
N = length(xx); % これがあってるかわからん
yy_mu = zeros(N, length(xx(1,:))); yy_var = zeros(N, length(xx(1,:)));
for i = 1:1:length(ytrain(1,:))
    regression = gpr(xx, xtrain, ytrain(:,i), params); % length(xx)行2列？
    yy_mu(:,i) = regression(:,1); yy_var(:,i) = regression(:,2); 
end
% クォータニオンの制約を満たすようにしたい
% とりあえず正規化              
for i = 1:1:length(yy_mu)
    yy_mu(i,1:4) = yy_mu(i,1:4) ./ norm(yy_mu(i,1:4));
end

two_sigma1 = yy_mu - 2 * sqrt(yy_var); two_sigma2 = yy_mu + 2 * sqrt(yy_var);

% 時系列順の姿勢履歴にorganize -----------------------------------------------
addpath('hara_functions/');
attiIni = xx(1, 1:7); mAppIni = t_mApp_test(1,2);
attiReg = zeros(N, length(xx(1,:))); attiReg(1,:) = attiIni;
attiReg_qe = zeros(N, 4); attiReg_qe(1,:) = q_pro(Dp_test(1,1:4)',attiReg(1,1:4)')'; % 誤差クォータニオンの初期値
mAppReg = zeros(N, 1); mAppReg(1,1) = mAppIni;
for i = 1:1:(N-1)
    % quaternions
    attiReg(i+1,1:4) = q_pro(yy_mu(i,1:4)', attiReg(i,1:4)')'; % 転置に注意
    % 真値と回帰結果の誤差クォータニオンを取る．
    attiReg_qe(i+1,1:4) = q_pro(Dp_test(i+1,1:4)', attiReg(i+1,1:4)')'; % 転置に注意
    % anglar velocity
    attiReg(i+1,5:7) = attiReg(i,5:7) + yy_mu(i,5:7);
    % Light Curves
    mAppReg(i+1,1) = mAppReg(i,1) + yy_mu(i,8);
end


% plot --------------------------------------------------------------------
f1 = figure; f2 = figure; f3 = figure; f4 = figure; f5 = figure; f6 = figure; f7 = figure;
f8 = figure; f9 = figure; f10 = figure; f11 = figure; f12 = figure; f13 = figure; f14 = figure;
f15 = figure;
figure(f1);
% patch([xx(:,1)', fliplr(xx(:,1)')], [two_sigma1', fliplr(two_sigma2')], 'c');
% hold on;
plot(xtrain(:,1), ytrain(:,1), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,1), Dp_test(:,8), 'r.'); % 真値？
hold on;
plot(xx(:,1), yy_mu(:,1), 'b.', LineWidth=2); % 回帰結果？
title("q1");

figure(f2);
plot(xtrain(:,2), ytrain(:,2), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,2), Dp_test(:,9), 'r.'); % 真値？
hold on;
plot(xx(:,2), yy_mu(:,2), 'b.'); % 回帰結果？
title("q2");

figure(f3);
plot(xtrain(:,3), ytrain(:,3), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,3), Dp_test(:,10), 'r.'); % 真値？
hold on;
plot(xx(:,3), yy_mu(:,3), 'b.'); % 回帰結果？
title("q3");

figure(f4);
plot(xtrain(:,4), ytrain(:,4), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,4), Dp_test(:,11), 'r.'); % 真値？
hold on;
plot(xx(:,4), yy_mu(:,4), 'b.'); % 回帰結果？
title("q4");

figure(f5);
plot(xtrain(:,5), ytrain(:,5), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,5), Dp_test(:,12), 'r.'); % 真値？
hold on;
plot(xx(:,5), yy_mu(:,5), 'b.'); % 回帰結果？
title("w1");

figure(f6);
plot(xtrain(:,6), ytrain(:,6), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,6), Dp_test(:,13), 'r.'); % 真値？
hold on;
plot(xx(:,6), yy_mu(:,6), 'b.'); % 回帰結果？
title("w2");

figure(f7);
plot(xtrain(:,7), ytrain(:,7), 'k.', MarkerSize=10); % 学習データ
hold on;
plot(xx(:,7), Dp_test(:,14), 'r.'); % 真値？
hold on;
plot(xx(:,7), yy_mu(:,7), 'b.'); % 回帰結果？
title("w3");

figure(f8);
% plot(t_test, Dp_test(:,1), 'r.'); 
% hold on;
% plot(t_test, attiReg(:,1), 'b.');
% hold on;
% 元々真値と回帰結果を同時にプロットしていたが，真値と回帰結果の誤差クォータニオンのプロットに変更
plot(t_test, attiReg_qe(:,1), 'g.');
filename = "q1errorTimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f9);
% plot(t_test, Dp_test(:,2), 'r.'); 
% hold on;
% plot(t_test, attiReg(:,2), 'b.');
% hold on;
plot(t_test, attiReg_qe(:,2), 'g.');
filename = "q2errorTimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f10);
% plot(t_test, Dp_test(:,3), 'r.'); 
% hold on;
% plot(t_test, attiReg(:,3), 'b.');
% hold on;
plot(t_test, attiReg_qe(:,3), 'g.');
filename = "q3errorTimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f11);
% plot(t_test, Dp_test(:,4), 'r.'); 
% hold on;
% plot(t_test, attiReg(:,4), 'b.');
% hold on;
plot(t_test, attiReg_qe(:,4), 'g.');
filename = "q4errorTimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f12);
plot(t_test, Dp_test(:,5), 'r.'); 
hold on;
plot(t_test, attiReg(:,5), 'b.');
hold on;
filename = "w1TimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f13);
plot(t_test, Dp_test(:,6), 'r.'); 
hold on;
plot(t_test, attiReg(:,6), 'b.');
hold on;
filename = "w2TimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f14);
plot(t_test, Dp_test(:,7), 'r.'); 
hold on;
plot(t_test, attiReg(:,7), 'b.');
hold on;
filename = "w3TimeHistory"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

figure(f15);
plot(t_test, t_mApp_test(:,2), 'r.'); % 真値
hold on;
plot(t_test, mAppReg(:,1), 'b.'); % 回帰結果
hold on;
filename = "LightCurves"; savename = strcat("figures/", filename, ".png");
title(filename);
saveas(gcf, savename);

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
% xxに何が入るかわからん → 学習データ以外の姿勢？
% xtrainは7次元の入力，ytrainは各姿勢データ1次元の出力(差分)
function y = gpr(xx, xtrain, ytrain, params)
    K = kernel_matrix(xtrain, params); % 学習データの入力Dp(:, 1:7)のカーネル行列K
    Kinv = inv(K);
%     N = length(xx) + length(xtrain); % xxは学習データ以外の姿勢? 学習データと完全に被らなければ足すだけでいい，のか？
    N = length(xx);
    L = length(xtrain(1,:));
    mu = zeros(N, L); var = zeros(N, L);
    for i = 1:1:N
        s = gaussian_kernel(xx(i,:), xx(i,:), params, false, true); % カーネル行列k_**
        k = kv(xx(i,:), xtrain, params); % 縦ベクトル (回帰する状態のカーネル行列k_*)
        mu(i,1) = k' * Kinv * ytrain;
        var(i,1) = s - k' * Kinv * k;
    end
    y = [mu var]; % 入力された姿勢の次の差分の確率分布が回帰された？
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



















