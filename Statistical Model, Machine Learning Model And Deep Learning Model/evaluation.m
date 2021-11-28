% ����real��prediction������ʵ���ݣ�Ԥ�����ݣ���Ϊn*1������ʽ
% ������ϵ�������Ϊsse��mse��rmse��mae��mape��R��POCID
function index=evaluation(real,prediction)
error= prediction-real;

sse = sum(error.^2);
mse = sse/length(real);
rmse = sqrt(mse);
mae = mean(abs(error));
mape = mean(abs((error)./real));
R = corr(real,prediction);

POCID=0;
for i=2:length(real)
   if  (real(i)-real(i-1))* (prediction(i)-prediction(i-1)) > 0 
       POCID=POCID+1;
   else
       POCID=POCID+0;
   end
end
POCID=POCID/(length(real)-1);

index=[rmse;R;POCID];
end