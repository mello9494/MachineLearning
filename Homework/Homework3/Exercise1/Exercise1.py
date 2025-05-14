import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

sqFeet = np.array([[100], [150], [185], [235], [310], [370], [420], [430], [440], [530], [600], [634], [718], [750], [850], [903], [978], [1010], [1050], [1990]])
price = np.array([12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989])

model = linear_model.LinearRegression()
model.fit(sqFeet, price)

predict = model.predict(sqFeet)

ransac = linear_model.RANSACRegressor()
ransac.fit(sqFeet, price)
inlierMask = ransac.inlier_mask_
outlierMask = np.logical_not(inlierMask)

lineX = np.arange(sqFeet.min(), sqFeet.max())[:, np.newaxis]
lineY = model.predict(lineX)
lineYRansac = ransac.predict(lineX)

print(f'Before RANSAC: \n\tSlope: {(predict[0] - predict[1])/(sqFeet[0] - sqFeet[1])}\n\ty-intercept: {model.intercept_}\n')
print(f'After RANSAC: \n\tSlope: {(lineYRansac[0] - lineYRansac[1])/(lineX[0] - lineX[1])}\n\ty-intercept: {ransac.predict([[0]])}\n')

plt.scatter(sqFeet[inlierMask], price[inlierMask], color='yellowgreen', marker='*', label='Inliers')
plt.scatter(sqFeet[outlierMask], price[outlierMask], color='red', marker='o', label='Outliers')
plt.plot(sqFeet, model.predict(sqFeet), color='blue', label='Before RANSAC')
plt.plot(lineX, lineYRansac, color='orange', label='After RANSAC')

plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend(loc='lower right')
plt.show()


