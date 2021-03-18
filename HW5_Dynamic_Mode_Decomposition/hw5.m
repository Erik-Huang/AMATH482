clear all; close all; clc;

%% Input data
videoInput = VideoReader('ski_drop_low.mp4');
dt = 1/videoInput.Framerate;
t = 0:dt:videoInput.Duration;
vidFrames = read(videoInput);
numFrames = get(videoInput,'NumFrames');
frame = vidFrames(:,:,:,1);
V = zeros(size(frame,1)*size(frame,2), numFrames);
for iter = 1:numFrames
    frame = vidFrames(:,:,:,iter);
    frame = im2double(frame);
    frame = rgb2gray(frame);
    frame = reshape(frame,[],1);
    V(:,iter) = frame;
end

%% SVD
X1 = V(:,1:end-1);
X2 = V(:,2:end);

[U2, S2, V2] = svd(X1, 'econ');
lambda = diag(S2).^2;
rank = size(U2,2);

%% Plot SVD Results
all_ranks = lambda/sum(lambda);
plot(1:10, all_ranks(1:10), 'bo--', 'Linewidth', 2);
title("Energy capture of first 10 modes of Monte Carlo Video");
xlabel("Modes"); ylabel("Energy");

%% DMD
rank = 10; % Use the first 40 modes only
rang = 1:rank;
S2 = S2(rang,rang);
U2 = U2(:,rang);
V2 = V2(:,rang);
Stilde = U2' * X2 * V2 * diag(1./diag(S2));

[eV, D] = eig(Stilde);
Phi = X2 * V2 / S2 * eV;

y0 = Phi \ V(:,1);
u_modes = zeros(rank, length(X1(1,:)));
omega = log(diag(D));
for iter = 1:length(X1(1,:))
    u_modes(:,iter) = y0.*exp(omega*iter);
end
u_dmd = Phi*u_modes;
u_dmd = abs(u_dmd);
u_sparse = X1 - u_dmd; % Get foreground

%% Plot DMD Results
h = 540;
w = 960;
f = 200;
colormap(gray);

subplot(1,3,1)
original = reshape(V(:,f), h, w);
imagesc(original);
xlabel('Original');

subplot(1,3,2)
background = reshape(u_dmd(:,f), h, w);
imagesc(background);
xlabel('Background');

subplot(1,3,3)
foreground = reshape(u_sparse(:,f), h, w);
imagesc(foreground);
xlabel('Foreground');