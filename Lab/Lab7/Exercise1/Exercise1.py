import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('datasets/wdbc.data.csv')

x = df.iloc[:, 2:]
y = df.iloc[:, 1]

scaler = StandardScaler()
log = LogisticRegression()
pca = PCA(n_components=2)

scaled_x = scaler.fit_transform(x)
pca_x = pca.fit_transform(scaled_x)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

data_point = np.array([7.76, 24.54, 47.92, 181, 0.05263, 0.04362, 0, 0, 0.1587, 0.05884, 0.3857, 1.428, 2.548, 19.15, 0.007189,
0.00466, 0, 0, 0.02676, 0.002783, 9.456, 30.37, 59.16, 268.6, 0.08996, 0.06444, 0, 0, 0.2871, 0.07039]).reshape(1, -1)

scaled_data_point = scaler.transform(data_point)
pca_data_point = pca.transform(scaled_data_point)
log.fit(pca_x, y)
coefs = log.coef_.flatten()
print(f'Coefs: {coefs}')
sorted_pca_x = np.sort(pca_x[:, 0])
x2 = [-((log.intercept_ / coefs[1]) + ((coefs[0] * sorted_pca_x[i]) / coefs[1])) for i in range(len(sorted_pca_x))]

pred = log.predict(pca_data_point)
print(f'Prediction: {pred}')

cdict={0: 'red', 1: 'green'}
labl={'M': 0, 'B': 1}
labels = [labl[i] for i in y]
fig,ax = plt.subplots(figsize=(7,5))
for l in np.unique(labels):
    ix = np.where(labels == l)
    ax.scatter(pca_x[:, 0][ix], pca_x[:, 1][ix], c=cdict[l], label=list(set(y))[l])

ax.scatter(pca_data_point[:, 0], pca_data_point[:, 1], label='Data point', marker='+', s=300, color='b')
ax.plot(sorted_pca_x, x2, color='orange', label='Decision boundary (logistic regression)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA = 2. Classification: Benign')
plt.legend(loc='upper left')
plt.show()
