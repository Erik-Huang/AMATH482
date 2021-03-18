close all; clear all; clc;
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')

%% Test videos
implay(vidFrames1_3);
implay(vidFrames2_3)
implay(vidFrames3_3)

%% Calculate size of frames
numFrames1_3 = size(vidFrames1_3,4);
numFrames2_3 = size(vidFrames2_3,4);
numFrames3_3 = size(vidFrames3_3,4);

%% Cam 1_2
filter = zeros(480,640);
filter(100:450, 300:450) = 1;
cam_1_3_res = zeros(numFrames1_3, 2);

for i = 1:numFrames1_3 
    frame = vidFrames1_3(:,:,:,i);
    frame = rgb2gray(frame);
    frame = double(frame);
    frame = frame .* filter;
    
    pink = frame > 250;
    Index = find(pink);
    [y, x] = ind2sub(size(pink), Index);
    cam_1_3_res(i,:) = [mean(x), mean(y)];
    imshow(pink); drawnow
end

%% Cam 2_2
% filter = zeros(480,640);
% filter(100:450, 250:350) = 1;
cam_2_3_res = zeros(numFrames2_3, 2);

for i = 1:numFrames2_3 
    frame = vidFrames2_3(:,:,:,i);
    frame = rgb2gray(frame);
    frame = double(frame);
    % frame = frame .* filter;
    
    pink = frame > 245;
    Index = find(pink);
    [y, x] = ind2sub(size(pink), Index);
    cam_2_3_res(i,:) = [mean(x), mean(y)];
    imshow(pink); drawnow
end

%% Cam 3_3
filter = zeros(480,640);
filter(180:340, 220:520) = 1;
cam_3_3_res = zeros(numFrames3_3, 2);

for i = 1:numFrames3_3 
    frame = vidFrames3_3(:,:,:,i);
    frame = rgb2gray(frame);
    frame = double(frame);
    frame = frame .* filter;
    
    pink = frame > 245;
    Index = find(pink);
    [y, x] = ind2sub(size(pink), Index);
    cam_3_3_res(i,:) = [mean(x), mean(y)];
    imshow(pink); drawnow
end

%% RESULT: Sync all video frames

[minimum, i] = min(cam_1_3_res(1:20, 2));
cam_1_3_trim = cam_1_3_res(1:end, :);
[minimum, i] = min(cam_2_3_res(1:30, 2));
cam_2_3_trim = cam_2_3_res(1:end, :);
[minimum, i] = min(cam_3_3_res(1:30, 1));
cam_3_3_trim = cam_3_3_res(1:end, :);


max_size = min([length(cam_1_3_trim(:,1)) length(cam_2_3_trim(:,1)) length(cam_3_3_trim(:,1))]);
max_frame = 1:max_size;
cam_1_3_trim = cam_1_3_trim(1:max_size, :);
cam_2_3_trim = cam_2_3_trim(1:max_size, :);
cam_3_3_trim = cam_3_3_trim(1:max_size, :);

%%


X = [cam_1_3_trim'; cam_2_3_trim'; cam_3_3_trim'];
pos_mean = mean(X, 2);
X_centered = X - pos_mean;

[u,s,v] = svd(X_centered ./ sqrt(max_size - 1), 'econ');
lambda = diag(s).^2;
Y = u' * X_centered; 

save('case_3_energy.mat', 'lambda');

%PCA Plot
figure(1)
subplot(2,1,1)
plot(max_frame, X_centered(2,:), max_frame, X_centered(4,:), max_frame, X_centered(5,:), 'LineWidth', 1.2)
hold on
plot(max_frame, Y(1,:), 'k','LineWidth', 1.5)
axis([0 max_size -180 180])
ylabel("Displacement (pixels)"); xlabel("Frames"); 
title("Case 3 - Displacement in Z-axis (Vertical)");
legend('Cam1', 'Cam2', 'Cam3', 'First Principal', 'Fontsize', 5)
subplot(2,1,2)
plot(max_frame, X_centered(1,:), max_frame, X_centered(3,:), max_frame, X_centered(6,:), 'LineWidth', 1.5)
axis([0 max_size -100 100])
ylabel("Displacement (pixels)"); xlabel("Frames"); 
title("Case 3 - Displacement in XY-plane (Horizontal)");
legend('Cam1', 'Cam2', 'Cam3', 'Fontsize', 5)
%Energy Plot
figure(2)
plot(1:1:6,lambda/sum(lambda),'r:o')
title('Energy of each dimension in Case 3','Fontsize',16)
ylabel('Energy','Fontsize',16)
xlabel('Dimensions','Fontsize',16)