from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


houses = fetch_california_housing()
df = pd.DataFrame(houses.data, columns=houses.feature_names)
target = pd.DataFrame(houses.target)

x = np.array(df.iloc[::10, :2])
y = np.array(target.iloc[::10]).flatten()

print(x.shape, y.shape)

linear_model = LinearRegression()
reg = linear_model.fit(x, y)
print(reg.coef_)

slope1, intercept1, r1, p1, std_error1 = stats.linregress(x[:,0], y)
slope2, intercept2, r2, p2, std_error2 = stats.linregress(x[:,1], y)

print(f'Slope1: {slope1}\ty-intercept1: {intercept1}\t')
print(f'Slope2: {slope2}\ty-intercept2: {intercept2}\t')

x1, x2 = np.meshgrid(x[:,0], x[:,1])
b1, b2 = reg.coef_[0], reg.coef_[1]
z = reg.intercept_ + b1*x1 + b2*x2

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(x1, x2, z, color = 'blue')
ax.scatter3D(x[:,0], x[:,1], y, c=y, cmap='Greens')
ax.set_title('3D Graph')
plt.show()
