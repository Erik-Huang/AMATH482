% Clean workspace
clear all; close all; clc 

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata 5 

%% FFT
L = 10; % spatial domain
n = 64; % Fourier modes

x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

sum = zeros(n,n,n);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Utn = fftn(Un);
    sum = sum + Utn;
end


%% Display data before Averaging
isosurface(X, Y, Z, fftshift(abs(Utn)) / max(abs(Utn),[],'all'), 0.5) 
axis([-20 20 -20 20 -20 20]), grid on
drawnow

%% Average the spectrum to cancel out white noise and find the center frequency
avg = abs(fftshift(sum))/49;
[M,I] = max(avg(:));
cf = [Kx(I),Ky(I),Kz(I)]; % Location of the center frequency
isosurface(Kx, Ky, Kz, avg / max(avg(:)), 0.5)
axis([-20 20 -20 20 -20 20]), grid on
drawnow

%% filter function
tau = 1;
filter = exp(-tau*((Kx-cf(1)).^2+(Ky-cf(2)).^2+(Kz-cf(3)).^2));

traj_x = zeros(1,49);
traj_y = zeros(1,49);
traj_z = zeros(1,49);

for k = 1:49
    Un(:,:,:) = reshape(subdata(:,k),n,n,n);
    filt = fftn(Un).* fftshift(filter);
    filt = fftshift(filt);
    filt = ifftn(filt);
    filt = filt / max(filt(:));
    
    [M,I] = max(filt(:));
    cf = [X(I),Y(I),Z(I)];
    traj_x(1,k) = cf(1);
    traj_y(1,k) = cf(2);
    traj_z(1,k) = cf(3);
end

plot3(traj_x, traj_y, traj_z, 'b', 'LineWidth', 1)
hold on
plot3(traj_x(49), traj_y(49), traj_z(49), 'rx', 'Markersize', 5)
xlabel('X')
ylabel('Y')
zlabel('Z')
title("Trajectory of the submarine")


