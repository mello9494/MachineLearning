import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

names = np.array(['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitosis', 'Class'])
df = pd.read_csv('datasets/breast-cancer-wisconsin.data.csv', names=names)
df = df[df['Bare Nuclei'] != '?']

x = np.array(df.drop(['id', 'Class'], axis=1))
y = np.array(df['Class'])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=42)

svc = SVC(kernel='linear').fit(x_train, y_train)
y_pred = svc.predict(x_test)

w0, w1 = svc.coef_[0]
intercept = svc.intercept_[0]

x_range = np.linspace(0, 2, 100)
boundary_y = (-intercept - w0 * x_range) / w1

acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy score: {acc_score}')
print(f'Confusion matrix: \n{conf_matrix}\n')

unique = list(set(y))
colors = {unique[0]: 'purple', unique[1]: 'yellow'}
for i in range(len(unique)):
    ix = np.where(y == unique[i])
    plt.scatter(x_pca[:, 0][ix], x_pca[:, 1][ix], c=colors[unique[i]], label=unique[i])

plt.plot(x_range, boundary_y, color='green')
plt.legend(loc='lower left')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('SVC with PCA')
plt.show()



