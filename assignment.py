import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv('car_prices.csv')
# 读取数据集

# 选择要分析的特征和目标变量
X = data['odometer'].values.reshape(-1, 1)  # 里程数作为特征
y = data['sellingprice'].values  # 售价作为目标变量

# 使用 scikit-learn 中的线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取模型参数
slope = model.coef_[0]
intercept = model.intercept_

# 构建预测模型
def linear_regression_predict(x):
    """使用拟合的线性回归模型进行预测"""
    return slope * x + intercept

# 计算预测值
predicted_sellingprice = linear_regression_predict(X)

# 计算 R 平方值
mean_y = np.mean(y)
total_sum_of_squares = np.sum((y - mean_y)**2)
residual_sum_of_squares = np.sum((y - predicted_sellingprice)**2)
r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

print("R 平方值:", r_squared)