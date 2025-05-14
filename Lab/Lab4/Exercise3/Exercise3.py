from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


houses = fetch_california_housing()
df = pd.DataFrame(houses.data, columns=houses.feature_names)
target = pd.DataFrame(houses.target)

x = np.array(df)
y = np.array(target).flatten()

print(x.shape, y.shape)

linear_model = LinearRegression()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
reg = linear_model.fit(x_scaled, y)
print(f'Coefficients: {reg.coef_}')

max_coef = 0
max_coef_name = ''
for i in range(len(reg.coef_)):
    if np.abs(reg.coef_[i]) > np.abs(max_coef):
        max_coef = reg.coef_[i]
        max_coef_name = df.columns[i]
print(f'Coefficient with most weight: {max_coef_name}, {max_coef}')
