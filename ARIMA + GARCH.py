import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

# 读取CSV文件
# 假设数据文件名为 'data.csv'，确保日期和数字格式正确
df = pd.read_csv('/Users/qunxing/Downloads/CFA/ETH.csv', parse_dates=['日期'], dayfirst=True, thousands=',', decimal='.')
df.set_index('日期', inplace=True)

# 转换涨跌幅，去掉百分号并转换为浮点数
df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype('float') / 100.0

# 以收盘价为分析对象
close_prices = df['收盘']

# 计算一阶差分
diff_series = close_prices.diff().dropna()

# ADF测试
result = adfuller(diff_series, autolag='AIC')  # 使用AIC自动选择滞后长度
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 如果ADF统计值大于任何置信度的临界值，我们不能拒绝序列是非平稳的假设

# 绘制ACF图
plt.figure(figsize=(12, 6))
plot_acf(diff_series, lags=40)  # lags是滞后数
plt.title('Autocorrelation Function')
plt.show()

# 绘制PACF图
plt.figure(figsize=(12, 6))
plot_pacf(diff_series, lags=40)  # lags是滞后数
plt.title('Partial Autocorrelation Function')
plt.show()

# 检查数据
print(close_prices.head())

from statsmodels.tsa.arima.model import ARIMA

# 建立ARIMA模型，参数假设为(1, 0, 0)
arima_model = ARIMA(diff_series, order=(1, 0, 0))
arima_result = arima_model.fit()

# 打印摘要
print(arima_result.summary())

# 检查残差的自相关性
residuals = arima_result.resid
plt.figure()
plot_acf(residuals, lags=20)
plt.title('Residuals ACF')
plt.show()

from arch import arch_model

# 使用ARIMA模型的残差
residuals = arima_result.resid

# 建立GARCH(1, 1)模型
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(update_freq=5)

# 打印GARCH模型结果
print(garch_result.summary())

plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('ARIMA Model Residuals')
plt.show()

# 预测未来5个时间点
forecast = arima_result.get_forecast(steps=5)
forecast_ci = forecast.conf_int()

print("Forecast:")
print(forecast.summary_frame())

# 绘制预测结果及其置信区间
plt.figure(figsize=(10, 5))
plt.plot(diff_series, label='Observed')
forecast.predicted_mean.plot(label='Forecast')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast and Confidence Interval')
plt.legend()
plt.show()

