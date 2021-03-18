close all; clear all; clc;

load("case_1_energy.mat")
case1_lambda = lambda;
load("case_2_energy.mat")
case2_lambda = lambda;
load("case_3_energy.mat")
case3_lambda = lambda;
load("case_4_energy.mat")
case4_lambda = lambda;

figure(1)
subplot(2,2,1)
plot(1:1:6,case1_lambda/sum(case1_lambda),'r:o')
title('Energy of each dimension in Case 1','Fontsize',10)
ylabel('Energy','Fontsize',12)
xlabel('Dimensions','Fontsize',12)

subplot(2,2,2)
plot(1:1:6,case2_lambda/sum(case2_lambda),'r:o')
title('Energy of each dimension in Case 2','Fontsize',10)
ylabel('Energy','Fontsize',12)
xlabel('Dimensions','Fontsize',12)

subplot(2,2,3)
plot(1:1:6,case3_lambda/sum(case3_lambda),'r:o')
title('Energy of each dimension in Case 3','Fontsize',10)
ylabel('Energy','Fontsize',12)
xlabel('Dimensions','Fontsize',12)

subplot(2,2,4)
plot(1:1:6,case4_lambda/sum(case4_lambda),'r:o')
title('Energy of each dimension in Case 4','Fontsize',10)
ylabel('Energy','Fontsize',12)
xlabel('Dimensions','Fontsize',12)