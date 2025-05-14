import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

np.set_printoptions(suppress=True)

df = pd.read_csv('datasets/materials.csv')

x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

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

cor_coeff = []

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

    cor_coeff.append(numerator / denominator)

for i in range(len(cor_coeff)):
    print(f'r for {df.columns[i+1]}: {cor_coeff[i]}')



#######################
new_x = np.ones((df.shape))
for i in range(len(new_x)):
    for j in range(1, len(new_x[i])):
        new_x[i][j] = x[i][j - 1]

x_trans = np.transpose(new_x)
A = np.zeros((new_x.shape[1], new_x.shape[1]))

# A matrix
for i in range(len(A)):
    for j in range(len(A[i])):
        for k in range(len(x)):
            A[i][j] += x_trans[i][k] * new_x[k][j]

# B matrix
temp = 0
B = np.zeros(new_x.shape[1])
for i in range(len(x_trans)):
    for j in range(len(x_trans[i])):
        temp += x_trans[i][j] * y[j]
    B[i] = temp
    temp = 0

A = np.linalg.inv(A)

coefs = np.zeros(new_x.shape[1])
for i in range(len(A)):
    for j in range(len(B)):
        coefs[i] += A[i][j] * B[j]

print(f'Coefficients: {coefs}')

vals_to_predict = np.array([[32.1, 37.5, 128.9], [36.9, 35.37, 130.03]])
predicted = np.zeros(vals_to_predict.shape[0])

for i in range(len(vals_to_predict)):
    predicted[i] = coefs[0]
    for j in range(1, len(coefs)):
        predicted[i] += coefs[j] * vals_to_predict[i][j - 1]

print(f'Predicted: {predicted}')
