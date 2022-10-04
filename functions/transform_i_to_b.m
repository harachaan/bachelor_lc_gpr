% 慣性系で表されているベクトルr_iを機体固定座標系へ変換
% ここでクォータニオンq(from ode45)は慣性系から機体固定座標系への回転を表す
function r_b = transform_i_to_b(q, q_inv, r_i)
r_b_q = zeros(4, length(q));
for i = 1:1:length(q)
    r_i_q = [r_i(:,i)
              0]; % クォータニオン積のために４行ベクトル化
    q_bi = q(:,i);
    q_bi_inv = q_inv(:,i);
    r_b_q(:,i) = q_pro(q_pro(q_bi, r_i_q), q_bi_inv);
end
r_b = r_b_q(1:3,:);

end