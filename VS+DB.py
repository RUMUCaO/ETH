import pandas as pd
import numpy as np
import pymc as pm
import theano.tensor as tt

# 加载数据
data = pd.read_csv('/Users/qunxing/Downloads/CFA/ETH.csv')
data['日期'] = pd.to_datetime(data['日期'])
data.set_index('日期', inplace=True)
returns = data['收盘'].pct_change().dropna()

# 建立贝叶斯ARIMA(1,0,0) + GARCH(1,1)模型
with pm.Model() as model:
    # ARIMA部分参数的先验分布
    ar = pm.Normal('ar', mu=0, sigma=1)
    ma = pm.Normal('ma', mu=0, sigma=0)
    sigma_ar = pm.HalfNormal('sigma_ar', sigma=0)

    # GARCH部分参数的先验分布
    omega = pm.HalfNormal('omega', sigma=1)
    alpha = pm.Beta('alpha', alpha=2, beta=2)
    beta = pm.Beta('beta', alpha=2, beta=2)

    # 计算残差
    epsilon = returns - ar * returns.shift(1) - ma * returns.shift(1).fillna(0)
    epsilon = epsilon.dropna()

    # 波动性方程
    h = tt.zeros_like(epsilon)
    h = tt.set_subtensor(h[0], omega / (1 - alpha - beta))
    for t in range(1, len(epsilon)):
        h = tt.set_subtensor(h[t], omega + alpha * epsilon[t - 1] ** 2 + beta * h[t - 1])

    # 似然函数
    likelihood = pm.Normal('likelihood', mu=0, sigma=tt.sqrt(h), observed=epsilon)

    # 变分推断优化
    approx = pm.fit(method='advi')
    trace = approx.sample(draws=5000)
