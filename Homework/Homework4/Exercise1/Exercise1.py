import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('datasets/avgHigh_jan_1895-2018.csv')

x = np.array(df.iloc[:, 0]).reshape(-1, 1) # reshape to 2D array
y = np.array(df.iloc[:, 1]).reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

slope = model.coef_
y_intercept = model.intercept_

reg = (slope * x) + y_intercept

x_predict = np.array([201901, 202301, 202401]).reshape(-1, 1)
predict = model.predict(x_predict)

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, reg, color='red', label='Model')
plt.scatter(x_predict, predict, color='green', label='Predicted')

plt.legend(loc='lower right')
plt.title(f'January Average High Temperatures. Slope: {slope}, y-intercept: {y_intercept}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()