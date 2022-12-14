% ニュートン法的な考え方が通じるかの確認
truemApp(1,1) = t_mApp_test(1,2);
truew(1,1:3) = ytest(1,5:7);
for i = 1:1:(Ntest-1)
    truemApp(i+1,1) = truemApp(i,1) + ytest(i,8);
    truew(i+1,1:3) = truew(i,1:3) + ytest(i,5:7);
end
% 通じたっぽい．．そりゃそうやけども