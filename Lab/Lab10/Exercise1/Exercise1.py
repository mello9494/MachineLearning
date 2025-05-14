import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes(as_frame=True)
x = np.array(diabetes.data)
y = np.array(diabetes.target)

scaler = StandardScaler().fit(x, y)
x_scaled = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

activations = ['relu', 'logistic', 'tanh']
max_iters = [1000, 2000, 3000]

best_arch = (float('inf'), float('inf'))
best_out = (float('inf'), float('inf'))

for act in activations:
  for iter in max_iters:
    mlp = MLPRegressor(activation=act, max_iter=iter, random_state=0)
    mlp.fit(x_train, y_train)
    pred = mlp.predict(x_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f'MSE: {mse}')
    print(f'R^2: {r2}\n')

    if mse < best_out[0] and r2 < best_out[1]:
      best_out = (mse, r2)
      best_arch = (act, iter)


print(f'Best architecture: {best_arch}')
print(f'Best output: {best_out}')