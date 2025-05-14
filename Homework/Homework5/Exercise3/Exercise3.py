import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly
import plotly.graph_objs as go

df = pd.read_csv('datasets/vehicles.csv')
names = []
# get the names from the name file
with open('datasets/vehicles.names', 'r') as file:
    for line in file:
        sub = line.strip().split(':')
        names.append(sub[0])

x = np.array(df.loc[:, 'cyl':])
y = np.array(df.loc[:, 'mpg'])
names = names[2:]  # remove the first two values (make and mpg)

reg = LinearRegression()
reg.fit(x, y)
predicted = reg.predict(np.array([6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4]).reshape(1, -1))

print(f'Predicted value: {predicted[0]} mpg')
