% 输入real，X，Y 代表真实数据，预测数据1，预测数据2，皆为n*1纵列形式
% 输出结果表示X，Y预测误差相比，小于1则后者更好，大于1则前者更好
function TU=comparison(real,X,Y)
error_1= X-real;
error_2= Y-real;
TU = sum(error_2.^2)/sum(error_1.^2);
end