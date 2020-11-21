import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np


# dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
# data = pd.read_csv('trade/csv/2.csv',sep=',').fillna(0)
data = pd.read_csv('trade/csv/1.csv',sep=',',parse_dates=['Date'], date_parser=dateparse).fillna(0)
# data.set_index('date_time')

plt.figure(figsize=(10,6))
plt.grid(True)
#plt.xlabel('Дата')
plt.ylabel('Цена закрытия')
plt.plot(data['Close'])
plt.title('Цена закрытия')
plt.show()
plt.savefig('trade/png/b1.png', bbox_inches = 'tight')

df_close = data['Close']
df_close.plot(style='k.')
plt.title('Точечный график цены закрытия')
plt.show()
plt.savefig('trade/png/b2.png', bbox_inches = 'tight')
#Distribution of the dataset
# df_close.plot(kind='kde')
#We can observe a near-normal distribution(bell-curve) over sales values.


def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    plt.savefig('trade/png/с1.png')
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df_close)


result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)


from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Скользящая средняя')
plt.plot(std_dev, color ="black", label = "Среднеквадратичное отклонение")
plt.plot(moving_avg, color="red", label = "Среднее зн.")
plt.legend()
plt.show()
plt.savefig('trade/png/b3.png', bbox_inches = 'tight')


train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
#plt.xlabel('Дата')
plt.ylabel('Цена закрытия')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()



#Model
model = ARIMA(train_data, order=(3, 1, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())


fc, se, conf = fitted.forecast(124, alpha=0.05)  # 95% conf

#pandas series
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='Тренировочная выборка')
plt.plot(test_data, color = 'blue', label='Рельная цена')
plt.plot(fc_series, color = 'orange',label='Предсказанная цена')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Предсказание цены')
#plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend(loc='upper left', fontsize=8)
plt.show()
plt.savefig('trade/png/b4.png', bbox_inches = 'tight')
print("предсказанные зн"+str(fc_series))
print("реальные зн"+str(test_data))

#report
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

#Around 3.5% MAPE implies the model is about 96.5% accurate in predicting the next 15 observations.


# Auto arima p,q,d as 3,1,2
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()
plt.savefig('trade/png/b5.png', bbox_inches = 'tight')


fc, se, conf = fitted.forecast(124, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='Тренировочная выборка')
plt.plot(test_data, color = 'blue', label='Рельная цена')
plt.plot(fc_series, color = 'orange',label='Предсказанная цена')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Предсказание цены')
#plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend(loc='upper left', fontsize=8)
plt.savefig('trade/png/books1.png', bbox_inches = 'tight')
plt.show()
plt.savefig('trade/png/b6.png', bbox_inches = 'tight')
