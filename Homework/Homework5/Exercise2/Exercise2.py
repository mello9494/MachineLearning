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

names_vals = []
for name, val in zip(names, reg.coef_):
    names_vals.append((name, val))

# sort the values by absolute value
for i in range(len(names_vals)-1):
    for j in range(i+1, len(names_vals)):
        if abs(names_vals[j][1]) > abs(names_vals[i][1]):
            names_vals[j], names_vals[i] = names_vals[i], names_vals[j]

# print the most important coefficient names and values
print('5 Most important values')
for i in range(5):
    print(f'Name: {names_vals[i][0]}\tValue: {names_vals[i][1]}')


X = df.loc[:, names_vals[0][0]]
Y = df.loc[:, names_vals[1][0]]
Z = df.loc[:, names_vals[2][0]]
markercolor = df.loc[:, names_vals[3][0]]
markershape = df.loc[:, names_vals[4][0]].replace(3, 'square').replace(4, 'circle').replace(5, 'cross')
markersize = df.loc[:, 'mpg']

fig = go.Scatter3d(x=X, y=Y, z=Z,
                    marker=dict(size=markersize,
                               color=markercolor,
                               symbol=markershape,
                               opacity=0.9,
                               reversescale=True,
                               colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

mylayout = go.Layout(scene=dict(xaxis=dict(title=names_vals[0][0]),
                                yaxis=dict(title=names_vals[1][0]),
                                zaxis=dict(title=names_vals[2][0])))

plotly.offline.plot({"data": [fig], "layout": mylayout}, auto_open=True, filename=("6DPlot.html"))