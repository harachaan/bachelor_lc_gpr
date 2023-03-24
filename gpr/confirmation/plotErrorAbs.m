function errorAbs = plotErrorAbs(filename)
% 誤差の絶対値をとってプロットする関数
% attiError.csvをtemporaryフォルダに入れておく

curdir = pwd;
filepath = strcat(curdir, '/../../../temporary/', filename);
savedir = strcat(curdir, '/../../../temporary/X_gpr/');

errors = readmatrix(filepath);
attiError = abs(errors(:,1:6)); 
mAppError = abs(errors(:,7));
errorAbs = mean([attiError, mAppError], 1);

t_test = 0:3:(size(errors, 1) - 1) * 3;


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
end
