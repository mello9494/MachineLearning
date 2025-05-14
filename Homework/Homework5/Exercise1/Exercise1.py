import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('datasets/vehicles.csv')
names = []
# get the names from the name file
with open('datasets/vehicles.names', 'r') as file:
    for line in file:
        sub = line.strip().split(':')
        names.append(sub[0])

x = np.array(df.loc[:, 'mpg':])
y = np.array(df.loc[:, 'mpg'])
names = names[2:]  # remove the first two values (make and mpg)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
reg = LinearRegression()
reg.fit(x_scaled, y)

names_vals = []
for name, val in zip(names, reg.coef_):
    names_vals.append((name, val))
    print(name, val)

# sort the values by absolute value
for i in range(len(names_vals)-1):
    for j in range(i+1, len(names_vals)):
        if abs(names_vals[j][1]) > abs(names_vals[i][1]):
            names_vals[j], names_vals[i] = names_vals[i], names_vals[j]

# print the most important coefficient names and values
print('5 Most important values')
for i in range(5):
    print(f'Name: {names_vals[i][0]}\tValue: {names_vals[i][1]}')

    