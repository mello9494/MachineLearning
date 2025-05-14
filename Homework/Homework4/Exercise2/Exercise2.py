import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True)

df = pd.read_csv('datasets/avgHigh_jan_1895-2018.csv')

x = np.array(df.iloc[:, 0]).reshape(-1, 1) # reshape to 2D array
y = np.array(df.iloc[:, 1]).reshape(-1, 1)

test_size = float(input('Enter a test size: '))
int_test_size = int(len(x) - (len(x) * test_size))

x_train = x[:int_test_size]
x_test = x[int_test_size:]
y_train = y[:int_test_size]
y_test = y[int_test_size:]

model = LinearRegression()
model.fit(x_train, y_train)

slope = model.coef_
y_intercept = model.intercept_

reg = (slope * x_train) + y_intercept

predict = model.predict(x_test)
rmse = 0

for i in range(len(y_test)):
    print(f'Actual: {y_test[i]}\tPredicted: {predict[i]}')
    rmse += (y_test[i] - predict[i]) ** 2

rmse /= len(y_test)
rmse **= 0.5

plt.scatter(x_train, y_train, color='blue', label='Data Points')
plt.plot(x_train, reg, color='red', label='Model')
plt.scatter(x_test, y_test, color='green', label='Predicted')

plt.legend(loc='lower right')
plt.title(f'January Average High Temperatures. Slope: {slope}, y-intercept: {y_intercept}, RMSE: {rmse}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()