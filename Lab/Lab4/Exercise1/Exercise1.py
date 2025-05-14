from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


houses = fetch_california_housing()
df = pd.DataFrame(houses.data, columns=houses.feature_names)
target = pd.DataFrame(houses.target)

x = np.array(df.iloc[::10, :])
y = np.array(target.iloc[::10])

linear_model = LinearRegression()
reg = linear_model.fit(x, y)

predict = reg.predict([[8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]])

print(f'Coefficients: {reg.coef_}')
print(f'Y-intercept: {reg.intercept_}')

print(f'Predicted Median House Value {predict}')