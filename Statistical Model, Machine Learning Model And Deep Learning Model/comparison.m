% ����real��X��Y ������ʵ���ݣ�Ԥ������1��Ԥ������2����Ϊn*1������ʽ
% ��������ʾX��YԤ�������ȣ�С��1����߸��ã�����1��ǰ�߸���
function TU=comparison(real,X,Y)
error_1= X-real;
error_2= Y-real;
TU = sum(error_2.^2)/sum(error_1.^2);
end