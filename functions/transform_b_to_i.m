% 機体固定座標系で表されているベクトルr_bを慣性系へ変換
% ここでクォータニオンq(from ode45)は慣性系から機体固定座標系への回転を表す
function r_i = transform_b_to_i(q, q_inv, r_b)
r_i_q = zeros(4, length(q));
for i = 1:1:length(q)
    r_b_q = [r_b(:,i)
              0]; % クォータニオン積のために４行ベクトル化
    q_bi = q(:,i);
    q_bi_inv = q_inv(:,i);
    r_i_q(:,i) = q_pro(q_pro(q_bi_inv, r_b_q), q_bi);
end
r_i = r_i_q(1:3,:);

end
