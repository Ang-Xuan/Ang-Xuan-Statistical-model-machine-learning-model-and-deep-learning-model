clc,clear;warning off;
% data=xlsread('dataset.xlsx',1,'B1:B144');
data=xlsread('dataset.xlsx',4,'B1:B3652');
% data=xlsread('exchange.xlsx','C2:C1094');
% data=xlsread('SML2010.xlsx','B1:B2764');
% data=xlsread('PV.xlsx','B2:B105121');
% data=xlsread('electricity.xlsx','B1:B140256');
data=data';
numTimeStepsTrain = round(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);
%% 
layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(200)
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.0005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% 
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

net = trainNetwork(XTrain,YTrain,layers,options);

net = resetState(net);
net = predictAndUpdateState(net,XTrain);
YPred = [];
for i = 1:numel(dataTest)
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i));
end
%% 
YPred = sig*YPred + mu;
sse = sum((dataTest - YPred).^2);
mse = sse/length(dataTest);
rmse = sqrt(mse);
mae = mean(abs(dataTest-YPred));
mape = mean(abs((dataTest-YPred)./dataTest));
r = sum( (dataTest -mean(dataTest)).*( YPred - mean(YPred)) ) / ...
    sqrt( sumsqr(dataTest -mean(dataTest)).* sumsqr(YPred - mean(YPred)) );
[sse;mse;rmse;mae;mape;r];