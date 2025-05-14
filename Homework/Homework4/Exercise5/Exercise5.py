import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv('datasets/materialsOutliers.csv')

x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

ransac = linear_model.RANSACRegressor(residual_threshold=15, stop_probability=1.00)
reg = linear_model.LinearRegression()

ransac.fit(x, y)
inlierMask = ransac.inlier_mask_
outlierMask = np.logical_not(inlierMask)
x_inliers = x[inlierMask]
y_inliers = y[inlierMask]

reg.fit(x_inliers, y_inliers)

print(f'Coefficients: {reg.coef_}')
print(f'y-intercept: {reg.intercept_}')




