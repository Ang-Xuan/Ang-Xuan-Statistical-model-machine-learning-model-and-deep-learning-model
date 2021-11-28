clear;  clc;
% data=xlsread('dataset.xlsx',1,'B1:B144');
data=xlsread('dataset.xlsx',4,'B1:B3652');
% data=xlsread('exchange.xlsx','C2:C1094');
% data=xlsread('SML2010.xlsx','B1:B2764');
% data=xlsread('PV.xlsx','B2:B105121');
% data=xlsread('electricity.xlsx','B1:B140256');
%% 构造样本集
num_whole=length(data);
train_ratio=0.9;                             %训练集数据比例
num_train=round(num_whole*train_ratio);
num_test=num_whole-num_train;

xxxxx = 24;
data_n = zeros(xxxxx+1, num_whole-xxxxx);
for i=1:num_whole-xxxxx
data_n(:,i) = data(i:i+xxxxx);
end
%% 划分训练、测试样本
trainx = data_n(1:xxxxx, 1:num_train);
trainy = data_n(xxxxx+1, 1:num_train);

testx = data_n(1:xxxxx, num_train-xxxxx+1:end);
real = data(num_train+1:end);
%% 创建RNN神经网络
net=elmannet(1:16,32,'traingdx');       % 包含   个神经元，训练函数为traingdx
net.trainParam.show=1;           % 设置显示级别
net.trainParam.epochs=20;           % 最大迭代次数为2000次
net.trainParam.goal=0.00001;       % 误差容限，达到此误差就可以停止训练
net.trainParam.max_fail=5;       % 最多验证失败次数
net=init(net);                     % 对网络进行初始化
%% 网络训练
[trainx1, st1] = mapminmax(trainx);
[trainy1, st2] = mapminmax(trainy);
testx1 = mapminmax('apply',testx,st1);

[net,per] = train(net,trainx1,trainy1);
%% 测试
test_ty1 = sim(net, testx1);       % 将测试数据输入网络进行测试
prediction = mapminmax('reverse', test_ty1, st2);
%% 
sse = sum((real - prediction).^2);
mse = sse/length(real);
rmse = sqrt(mse);
mae = mean(abs(real-prediction));
mape = mean(abs((real-prediction)./real));
r = sum( (real -mean(real)).*( prediction - mean(prediction)) ) / ...
    sqrt( sumsqr(real -mean(real)).* sumsqr(prediction - mean(prediction)) );
[sse;mse;rmse;mae;mape;r];