clear;clc;
data=xlsread('dataset.xlsx',1,'B1:B144');
data=xlsread('dataset.xlsx',4,'B1:B3652');
% data=xlsread('bicycle.xlsx','B1:B833');
% data=xlsread('exchange.xlsx','C2:C1094');
% data=xlsread('SML2010.xlsx','B1:B2764');
% data=xlsread('PV.xlsx','B2:B105121');
% data=xlsread('electricity.xlsx','B1:B140256');
num_whole=length(data);         
train_ratio=0.9;                             %训练集数据比例
num_train=round(num_whole*train_ratio);
traindata= data(1:num_train);
num_test=num_whole-num_train;
%% 训练数据
xxxxx=24;
utrain=num_train-xxxxx;
input_train=zeros(xxxxx,utrain);
for i=1:utrain 
    input_train(:,i)=traindata(i:i+xxxxx-1)';     %训练数据自身划出一部分变成列
end

output_train=zeros(1,utrain);
for i=1:utrain    
   output_train(1,i)=traindata(i+xxxxx)';        %训练数据摘出后几个
end
%% 测试数据
ntest =num_test+xxxxx;
testdata=data(end-ntest+1:end);                  %测试数据取尾部余部分＋XXXX

input_test=zeros(xxxxx,num_test);
for i=1:num_test 
    input_test(:,i)=testdata(i+1:i+xxxxx)';       %侵占了前面XXXX
end

real=testdata(end-num_test+1:end)';
%% 归一化
[input_train_guiyi,PS1]=mapminmax(input_train);
[output_train_guiyi,PS2]=mapminmax(output_train);

[input_test_guiyi,PS3]=mapminmax(input_test);
[real_guiyi,PS4]=mapminmax(real);
%% 训练
trainD=reshape(input_train_guiyi, [size(input_train_guiyi,1),1,1,size(input_train_guiyi,2)]);
testD =reshape(input_test_guiyi,  [size(input_test_guiyi,1),1,1,size(input_test_guiyi,2)]);                                 

layers = [imageInputLayer([xxxxx 1 1] )                         % 输入层，1个通道
    convolution2dLayer(3,16,'Padding','same')                  % 卷积层：卷积核大小为3×3，卷积核的个数为16 
    reluLayer                                                  % ReLU非线性激活函数
    averagePooling2dLayer(1,'Stride',2)                        % 池化层：池化方式：平均池化；池化区域为2×2，步长为2  
    dropoutLayer(0.2)                                          % dropout层，随机将20%的输入置零   
    fullyConnectedLayer(200)  
    fullyConnectedLayer(200) 
    fullyConnectedLayer(1)                                    % 全连接层,全连接层的输出为1
    regressionLayer ];                                        % 回归层，用于预测结果计算损失  
    
options = trainingOptions('adam', ...                         % 设置训练方法
    'MiniBatchSize',8, ...                                   % 设置最小样本训练数量，
    'MaxEpochs',1, ...                                      % 设置最大训练轮数
    'InitialLearnRate',0.005, ...                             % 设置初始学习率为0.001
    'LearnRateSchedule','piecewise', ...                      % 设置初始的学习率是变化的
    'LearnRateDropFactor',0.01, ...                           % 设置学习率衰减因子为0.1
    'LearnRateDropPeriod',20, ...                             % 设置学习率衰减周期为20轮，即：每20轮，在之前的学习率基础上，乘以学习率的衰减因子0.1
    'Shuffle','every-epoch', ...                              % 设置每一轮都打乱数据
    'ValidationData',{testD,real_guiyi'}, ...                 % 设置验证数据
    'ValidationFrequency',1000, ...                           % 设置验证频率
    'Plots','training-progress', ...                          % 设置打开训练进度图
    'Verbose',true);                                          % 设置关闭命令窗口的输出

net = trainNetwork(trainD,output_train_guiyi',layers,options);
prediction = predict(net,testD);       
%% 输出反归一化
CNNprediction= mapminmax('reverse',prediction,PS4);
%% 指标
sse = sum((real' - CNNprediction).^2);
mse = sse/length(real);
rmse = sqrt(mse);
mae = mean(abs(real'-CNNprediction));
mape = mean(abs((real'-CNNprediction)./real'));
r = sum( (real' -mean(real')).*( CNNprediction - mean(CNNprediction)) ) / ...
    sqrt( sumsqr(real' -mean(real')).* sumsqr(CNNprediction - mean(CNNprediction)) );
[sse;mse;rmse;mae;mape;r];