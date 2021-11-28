# -*- coding: utf-8 -*-
# 用 SARIMA 进行时间序列预测

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv

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
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

'''print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))'''

'''warnings.filterwarnings("ignore")  # specify to ignore warning messages
a = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            a.append(results.aic)
        except:
            continue
print(a)'''

mod = sm.tsa.statespace.SARIMAX(data,
                                order=(2, 0, 2),
                                seasonal_order=(1, 1, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

#results.plot_diagnostics(figsize=(15, 12))
#plt.show()

predict = results.get_prediction('2013-05-20', '2013-08-10', dynamic=False)#预测值
predict_ci = predict.conf_int()

predict_dynamic = results.get_prediction('2013-05-20', '2013-08-10', dynamic=True, full_results=True)#动态预测值
predict_dynamic_ci = predict_dynamic.conf_int()
predict = predict_dynamic.predicted_mean
print(predict_dynamic.predicted_mean)
predict.to_csv("data.csv", header='true', encoding='utf-8')
ax = data['2011-05-01':].plot(label='data', figsize=(20, 15))
predict_dynamic.predicted_mean.plot(label='Forecast', ax=ax)
plt.plot(data_raw)
plt.show()


