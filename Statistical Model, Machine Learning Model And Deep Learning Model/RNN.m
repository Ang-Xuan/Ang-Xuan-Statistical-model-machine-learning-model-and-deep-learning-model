clear;  clc;
% data=xlsread('dataset.xlsx',1,'B1:B144');
data=xlsread('dataset.xlsx',4,'B1:B3652');
% data=xlsread('exchange.xlsx','C2:C1094');
% data=xlsread('SML2010.xlsx','B1:B2764');
% data=xlsread('PV.xlsx','B2:B105121');
% data=xlsread('electricity.xlsx','B1:B140256');
%% ����������
num_whole=length(data);
train_ratio=0.9;                             %ѵ�������ݱ���
num_train=round(num_whole*train_ratio);
num_test=num_whole-num_train;

xxxxx = 24;
data_n = zeros(xxxxx+1, num_whole-xxxxx);
for i=1:num_whole-xxxxx
data_n(:,i) = data(i:i+xxxxx);
end
%% ����ѵ������������
trainx = data_n(1:xxxxx, 1:num_train);
trainy = data_n(xxxxx+1, 1:num_train);

testx = data_n(1:xxxxx, num_train-xxxxx+1:end);
real = data(num_train+1:end);
%% ����RNN������
net=elmannet(1:16,32,'traingdx');       % ����   ����Ԫ��ѵ������Ϊtraingdx
net.trainParam.show=1;           % ������ʾ����
net.trainParam.epochs=20;           % ����������Ϊ2000��
net.trainParam.goal=0.00001;       % ������ޣ��ﵽ�����Ϳ���ֹͣѵ��
net.trainParam.max_fail=5;       % �����֤ʧ�ܴ���
net=init(net);                     % ��������г�ʼ��
%% ����ѵ��
[trainx1, st1] = mapminmax(trainx);
[trainy1, st2] = mapminmax(trainy);
testx1 = mapminmax('apply',testx,st1);

[net,per] = train(net,trainx1,trainy1);
%% ����
test_ty1 = sim(net, testx1);       % ��������������������в���
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