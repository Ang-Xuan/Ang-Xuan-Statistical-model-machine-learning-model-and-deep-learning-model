# -*- coding: utf-8 -*-
# 用 ARIMA 进行时间序列预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.graphics.api import qqplot
import csv

# 1.创建数据
with open('bicycle.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data_raw = []
    for i in reader:
        data_raw.append(float(i[1]))

data = data_raw[:750]
data = pd.Series(data)
data.index = pd.Index(pd.date_range('2011-05-01', '2013-05-19', freq='D'))

data_raw = pd.Series(data_raw)
data_raw.index = pd.Index(pd.date_range('2011-05-01', '2013-08-10', freq='D'))
#绘制时序的数据图
'''data_raw.plot(figsize=(12, 8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("bicycle")
plt.show()'''

#2.下面我们先对非平稳时间序列进行时间序列的差分，找出适合的差分次数d的值：
'''fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
diff1 = data.diff(1)
diff1.plot(ax=ax1)
#这里是做了1阶差分，可以看出时间序列的均值和方差基本平稳，不过还是可以比较一下二阶差分的效果：

#这里进行二阶差分
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
diff2 = diff1.diff()
diff2.plot(ax=ax2)'''
#由下图可以看出来一阶跟二阶的差分差别不是很大，所以可以把差分次数d设置为1，上面的一阶和二阶程序我们注释掉
#plt.show()

#这里我们使用一阶差分的时间序列
#3.接下来我们要找到ARIMA模型中合适的p和q值：
diff1 = data.diff(1)
diff1.dropna(inplace=True)
diff2 = diff1.diff(1)
diff2.dropna(inplace=True)
#加上这一步，不然后面画出的acf和pacf图会是一条直线

#第一步：先检查平稳序列的自相关图和偏自相关图
'''fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff2, lags=40, ax=ax1)
#lags 表示滞后的阶数
#第二步：下面分别得到acf 图和pacf 图
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff2, lags=40, ax=ax2)
plt.show()'''

'''import itertools
import numpy as np
import seaborn as sns

p_min = 0
d_min = 2
q_min = 0
p_max = 5
d_max = 2
q_max = 6
# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue

    try:
        model = sm.tsa.ARIMA(data, order=(p, d, q),
                             # enforce_stationarity=False,
                             # enforce_invertibility=False,
                             )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.show()'''
#第三步:找出最佳模型ARMA
arma_mod = sm.tsa.ARMA(diff1, (2, 0)).fit()
#print(arma_mod.aic, arma_mod.bic, arma_mod.hqic)

#第五步：平稳模型预测,对未来进行预测
predict = arma_mod.predict('2013-05-20', '2013-08-10', dynamic=True)
print(predict)
fig, ax = plt.subplots(figsize=(12, 8))
ax = diff1.loc['2011-05-01':].plot(ax=ax)
predict.plot(ax=ax)
plt.title("predict_diff")

#还原到原始序列
ts_restored = data
#第六步：使用ARIMA模型进行预测
model = ARIMA(ts_restored, order=(2, 1, 0)) #导入ARIMA模型
result = model.fit()
predict = result.predict('2013-05-20', '2013-08-10', typ='levels')
print(predict)
#print(result.summary())

predict.to_csv("data.csv", header='true', encoding='utf-8')

result.conf_int()#模型诊断，可以发现所有的系数置信区间都不为0；即在5%的置信水平下，所有的系数都是显著的，即模型通过检验。

#最后画出时序图
#fig, ax = plt.subplots(figsize=(12, 8))
#ax = data_raw.loc['2011-01-01':'2012-12-31'].plot(ax=ax)
#plt.title("1")
fig = result.plot_predict(2, 833)  #因为前面是731个数，所以加上预测的15个就是746
plt.plot(data_raw)
plt.title("Predict")
plt.show()   #数据预测并画图
