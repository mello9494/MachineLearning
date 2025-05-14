import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

np.set_printoptions(suppress=True)

df = pd.read_csv('datasets/materials.csv')

x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])


# get correlation coefficients and see which ones are the best
mean_x = [0 for _ in range(x.shape[1])]  # means for x
mean_y = 0  # mean for y

# mean for x
for i in range(len(x[0])):
    column = x[:, i]
    for j in column:
        mean_x[i] += j
    mean_x[i] /= len(column)

# mean for y
for i in range(len(y)):
    mean_y += y[i]

mean_y /= len(y)

cor_coeff = {}

for i in range(len(x[0])):
    column = x[:, i]
    column_mean = mean_x[i]
    numerator = 0
    denominator = 0

    # calculate numerator
    for j in range(len(column)):
        numerator += ((column[j] - column_mean) * (y[j] - mean_y))

    # calculate denominator
    left = 0
    right = 0
    for j in range(len(column)):
        left += (column[j] - column_mean) ** 2
        right += (y[j] - mean_y) ** 2
        
    denominator = (left * right) ** 0.5

    cor_coeff[df.columns[i+1]] = (numerator / denominator)

for i in range(len(cor_coeff)):
    print(f'r for {df.columns[i+1]}: {cor_coeff[df.columns[i+1]]}')

# get the two most correlated predictor values
in_order_coeffs = [i for i in cor_coeff.values()]
temp_max = 0
for i in range(len(in_order_coeffs)):
    for j in range(i, len(in_order_coeffs)):
        if np.abs(in_order_coeffs[j]) < np.abs(in_order_coeffs[i]):
            in_order_coeffs[i], in_order_coeffs[j] = in_order_coeffs[j], in_order_coeffs[i]


in_order_coeffs = in_order_coeffs[-2:]

x1_label, x2_label = '', ''
for i in cor_coeff:
    if cor_coeff[i] == in_order_coeffs[0]:
        x1_label = i
    elif cor_coeff[i] == in_order_coeffs[1]:
        x2_label = i


linear_model = LinearRegression()
reg = linear_model.fit(df.loc[:, [x1_label, x2_label]], y)

x1, x2 = np.meshgrid(df.loc[:, x1_label], df.loc[:, x2_label])
b1, b2 = reg.coef_[0], reg.coef_[1]
z = reg.intercept_ + b1*x1 + b2*x2

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(x1, x2, z, color = 'blue')
ax.scatter3D(df.loc[:, x1_label], df.loc[:, x2_label], y, c=y, cmap='Greens')
ax.set_title('3D Graph')
ax.set_xlabel('Pressure')
ax.set_ylabel('Temperature')
ax.set_zlabel('Strength')
plt.show()
