% Clean workspace
clear all; close all; clc
%% Figure of sampling versus time for GNR
figure(1)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
 
t = (1:length(y))/Fs;
plot(t,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O'' Mine');
% p8 = audioplayer(y,Fs); playblocking(p8); % Sound player
 
%% Gabor Transform for GNR
n = length(y);
L = tr_gnr;
k = (1/L)*[0:(n/2 - 1) -n/2:-1];
ks = fftshift(k);
% create signal
S = y';
tau = 0:0.1:L;
a = 5000;
Sft = zeros(length(y), length(tau));
 
for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2); % Gaussian
    Sft = fft(g.*S);
    [M,I] = max(abs(Sft(:)));
    Sft(:,i) = fftshift(abs(Sft));
end
 
 
figure(2)
pcolor(tau,ks,Sft)
shading interp
set(gca,'ylim',[0 1000],'Fontsize',12)
colormap(hot)
yticks([277, 367, 415, 554, 698, 740])
yticklabels({'#C', '#F', '#G', '#C', 'F', '#F'})
yyaxis right
set(gca,'ylim',[0 1000],'Fontsize',12)
xlabel('Time (t)'), ylabel('Frequency (k)');
title('Sweet Child O'' Mine');

%% Clear space
clear all; close all; clc

%% Gabor Transform for Floyd
[y, Fs] = audioread('Floyd.m4a');
n = length(y); % Fourier modes
tr_gnr = n / Fs; % record time in seconds t = (1:n) / Fs;
					
k = (1 / tr_gnr) * [0:(n/2 - 1) (-n/2):-1]; 
% frequency component ks = fftshift(k);
					
tau = 0:1:tr_gnr;
a = 6000;
S = y';
Sgt_spec = zeros(n - 1, length(tau));
					
for j = 1:length(tau)
    g = exp(-a * (t - tau(j)).^2);
    Sg = g .* S;
    Sft = fft(Sg);
					
    Sft = Sft(1:n-1);
    [maximum, index] = max(abs(Sft));
    filter = exp(-0.01 * (k - k(index)).^2);
    Sft = Sft .* filter;
    Sft(k > 250) = 0;
    Sft(k < 60) = 0;
    Sgt_spec(:, j) = fftshift(abs(Sft));				
end 

%%
figure(2)
pcolor(tau,ks,Sft)
shading interp
set(gca,'ylim',[0 1000],'Fontsize',12)
colormap(hot)
yticks([87.307, 110.00, 123.47, 185.00, 246.94])
yticklabels({'F','A', 'B', '#F', 'B'})
yyaxis right
set(gca,'ylim',[0 1000],'Fontsize',12)
xlabel('Time (t)'), ylabel('Frequency (k)');
title('Comfortably Numb');

 
