% とりあえずの学習データをランダムに作りたい
function samples = make_samples(filename, numheader)

    data = readmatrix(filename, "NumHeaderLines", numheader);
    % 区間(0, 1)の一様分布からの乱数をもとの平板データに足す
    x = 1e-3 * (0.5 - rand(length(data), 7, 10));
    samples = zeros(length(data), 7, length(x(1,1,:)));
    for i = 1:1:length(x(1,1,:))
        samples(:,:,i) = data + x(:,:,i);
        sample = array2table(samples(:,:,i));
        sample.Properties.VariableNames(1:7) = {'q0', 'q1', 'q2', 'q3', 'x_cowell', 'y_cowell', 'z_cowell'};
        savename = strcat('simple_plate_', num2str(i), '.csv');
        writetable(sample, savename);
    end

end