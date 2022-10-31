% qとpのクォータニオン積
% クォータニオンpで表される姿勢をクォータニオンqで回転させる

function q_product = q_pro(q, p)

q1 = q(1, 1); q2 = q(2, 1); q3 = q(3, 1);
qbar = [q1; q2; q3]; q4 = q(4, 1);

p1 = p(1, 1); p2 = p(2, 1); p3 = p(3, 1);
pbar = [p1; p2; p3]; p4 = p(4, 1);

q_product = [q4*pbar + p4*qbar - cross(qbar, pbar);
             q4*p4 - qbar.'*pbar];
end


%-------------------------------------------------
% 最初に作ったやつ
% 
% function q_product = q_pro(q, p)
% q1 = q(1,1);
% q2 = q(2,1);
% q3 = q(3,1);
% qbar = [q1
%     q2
%     q3];
% q4 = q(4,1);
% qbar_cross_mat = [0 -q3 q2
%                   q3 0 -q1
%                   -q2 q1 0];
% 
% q_product = [q4*eye(3)-qbar_cross_mat qbar;
%             -qbar' q4] * p; % この組み方賢い．
% end