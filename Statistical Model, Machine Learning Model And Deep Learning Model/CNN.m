clear;clc;
data=xlsread('dataset.xlsx',1,'B1:B144');
data=xlsread('dataset.xlsx',4,'B1:B3652');
% data=xlsread('bicycle.xlsx','B1:B833');
% data=xlsread('exchange.xlsx','C2:C1094');
% data=xlsread('SML2010.xlsx','B1:B2764');
% data=xlsread('PV.xlsx','B2:B105121');
% data=xlsread('electricity.xlsx','B1:B140256');
num_whole=length(data);         
train_ratio=0.9;                             %ѵ�������ݱ���
num_train=round(num_whole*train_ratio);
traindata= data(1:num_train);
num_test=num_whole-num_train;
%% ѵ������
xxxxx=24;
utrain=num_train-xxxxx;
input_train=zeros(xxxxx,utrain);
for i=1:utrain 
    input_train(:,i)=traindata(i:i+xxxxx-1)';     %ѵ������������һ���ֱ����
end

output_train=zeros(1,utrain);
for i=1:utrain    
   output_train(1,i)=traindata(i+xxxxx)';        %ѵ������ժ���󼸸�
end
%% ��������
ntest =num_test+xxxxx;
testdata=data(end-ntest+1:end);                  %��������ȡβ���ಿ�֣�XXXX

input_test=zeros(xxxxx,num_test);
for i=1:num_test 
    input_test(:,i)=testdata(i+1:i+xxxxx)';       %��ռ��ǰ��XXXX
end

real=testdata(end-num_test+1:end)';
%% ��һ��
[input_train_guiyi,PS1]=mapminmax(input_train);
[output_train_guiyi,PS2]=mapminmax(output_train);

[input_test_guiyi,PS3]=mapminmax(input_test);
[real_guiyi,PS4]=mapminmax(real);
%% ѵ��
trainD=reshape(input_train_guiyi, [size(input_train_guiyi,1),1,1,size(input_train_guiyi,2)]);
testD =reshape(input_test_guiyi,  [size(input_test_guiyi,1),1,1,size(input_test_guiyi,2)]);                                 

layers = [imageInputLayer([xxxxx 1 1] )                         % ����㣬1��ͨ��
    convolution2dLayer(3,16,'Padding','same')                  % ����㣺����˴�СΪ3��3������˵ĸ���Ϊ16 
    reluLayer                                                  % ReLU�����Լ����
    averagePooling2dLayer(1,'Stride',2)                        % �ػ��㣺�ػ���ʽ��ƽ���ػ����ػ�����Ϊ2��2������Ϊ2  
    dropoutLayer(0.2)                                          % dropout�㣬�����20%����������   
    fullyConnectedLayer(200)  
    fullyConnectedLayer(200) 
    fullyConnectedLayer(1)                                    % ȫ���Ӳ�,ȫ���Ӳ�����Ϊ1
    regressionLayer ];                                        % �ع�㣬����Ԥ����������ʧ  
    
options = trainingOptions('adam', ...                         % ����ѵ������
    'MiniBatchSize',8, ...                                   % ������С����ѵ��������
    'MaxEpochs',1, ...                                      % �������ѵ������
    'InitialLearnRate',0.005, ...                             % ���ó�ʼѧϰ��Ϊ0.001
    'LearnRateSchedule','piecewise', ...                      % ���ó�ʼ��ѧϰ���Ǳ仯��
    'LearnRateDropFactor',0.01, ...                           % ����ѧϰ��˥������Ϊ0.1
    'LearnRateDropPeriod',20, ...                             % ����ѧϰ��˥������Ϊ20�֣�����ÿ20�֣���֮ǰ��ѧϰ�ʻ����ϣ�����ѧϰ�ʵ�˥������0.1
    'Shuffle','every-epoch', ...                              % ����ÿһ�ֶ���������
    'ValidationData',{testD,real_guiyi'}, ...                 % ������֤����
    'ValidationFrequency',1000, ...                           % ������֤Ƶ��
    'Plots','training-progress', ...                          % ���ô�ѵ������ͼ
    'Verbose',true);                                          % ���ùر�����ڵ����

net = trainNetwork(trainD,output_train_guiyi',layers,options);
prediction = predict(net,testD);       
%% �������һ��
CNNprediction= mapminmax('reverse',prediction,PS4);
%% ָ��
sse = sum((real' - CNNprediction).^2);
mse = sse/length(real);
rmse = sqrt(mse);
mae = mean(abs(real'-CNNprediction));
mape = mean(abs((real'-CNNprediction)./real'));
r = sum( (real' -mean(real')).*( CNNprediction - mean(CNNprediction)) ) / ...
    sqrt( sumsqr(real' -mean(real')).* sumsqr(CNNprediction - mean(CNNprediction)) );
[sse;mse;rmse;mae;mape;r];