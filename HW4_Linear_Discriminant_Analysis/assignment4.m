clear all; close all; clc;

[train_image, train_label] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[test_image,  test_label] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
train_image_reshaped = reshape(train_image, size(train_image,1)*size(train_image,2), []).';
test_image_reshaped = reshape(test_image, size(test_image,1)*size(test_image,2), []).';
train_image = im2double(train_image_reshaped)';
test_image = im2double(test_image_reshaped)';
train_label = im2double(train_label);
test_label = im2double(test_label);

row_mean = mean(train_image,2);
train_image = double(train_image)-repmat(row_mean, 1, length(train_image));

% Singular Value Decomposition
[U, S, V ] = svd(train_image, 'econ');

energy = 0;
r = 0;
while energy < 0.95
    r = r + 1;
    energy = energy + S(r,r)/sum(diag(S));
end
train_image = (U(:, 1:r))'*train_image;
test_image = (U(:, 1:r))'*test_image;
%% PCA PLOT on digit 0, 9
for i = [0,9]
    disp(i);
    modes = train_image(:, find(train_label == i));
    plot3(modes(2,:), modes(3,:), modes(5,:),'o'); 
    hold on
end
xlabel('Mode 2')
ylabel('Mode 3')
zlabel('Mode 5')
legend('0', '9')

%% LDA classifier

X = train_image;
T = test_image;

Sb = zeros(row_size);
Sw = zeros(row_size);
train_size = size(train_image, 2);
test_size = size(test_image, 2);
row_size = size(train_image,1);
Mu = mean(train_image, 2);

for i = [0,7]
    mask = (train_label ==  i);
    x = X(:, mask);
    ni = size(x, 2);
    pi = ni / train_size;
    mu_i = mean(x, 2);
    Si = (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';
    Sw = Sw + Si ;
    Sb = Sb + (mu_i - Mu) * (mu_i - Mu)';
end
M = pinv(Sw) * Sb;
[U, D, V] = svd(M);

%% PLOT CLASSIFIER

G2 = U(:,1:size(M,1));
Y2 = G2' * X;
mask = (train_label == 0);
a = Y2(1,mask);
b = Y2(2,mask);
d = [a'; b'];
plot(d,0*ones(size(d)),'.b','Linewidth',2)
hold on
mask = (train_label == 4);
a = Y2(1,mask);
b = Y2(2,mask);
d = [a'; b'];
plot(d,4*ones(size(d)),'.g','Linewidth',2)
hold on 
mask = (train_label == 9);
a = Y2(1,mask);
b = Y2(2,mask);
d = [a'; b'];
plot(d,9*ones(size(d)),'.r','Linewidth',2)
ylim([-1 10])
title(['LDA classifier on digit 0, 4 and 9']);
 
%% Accuracy Test for LDA
digit1 = 4;
digit2 = 9;

Y = G2' * X;
Y_t = G2'* T;

train_n = Y(:,find(train_label == digit1|train_label ==digit2));
test_n = Y_t(:,find(test_label == digit1|test_label ==digit2)); 
accuracy = accurCal(test_n, train_n,...
    test_label(find(test_label == digit1 |test_label ==digit2)), ...
    train_label(find(train_label == digit1 |train_label ==digit2)));
disp(accuracy);


%% SVM on two digit

train_trimed = train_image(:,find(train_label == digit1|train_label == digit2));
label_trimed = train_label(find(train_label == digit1|train_label == digit2));
test_trimed = test_image(:,find(test_label == digit1|test_label == digit2));
test_label_trimed = test_label(find(test_label == digit1|test_label == digit2));
svm_best = fitcsvm(train_trimed',label_trimed);
svm_loss = loss(svm_best, test_trimed.', test_label_trimed);


%% Build Decision Tree
tree = fitctree(train_image',train_label, 'MaxNumSplits',10,'CrossVal','on');
rfL = kfoldLoss(rfMdl, 'LossFun','ClassifErr');
view(tree.Trained{1},'Mode','graph');


%%
function [accuracy] = accurCal(test_data, train_data, test_label, train_label)
test_size = size(test_data, 2);
cnt = zeros(test_size, 1);
for test_digit = 1:1:test_size
    test_mat = repmat(test_data(:, test_digit), [1,size(train_data, 2)]);
    dist = sum(abs(test_mat - train_data).^2);
    [M, I] = min(dist);
    if train_label(I) == test_label(test_digit)
        cnt(test_digit) = cnt(test_digit) + 1;
    end
end
accuracy = double(sum(cnt)) / test_size;
end