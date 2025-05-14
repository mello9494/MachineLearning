import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from matplotlib import pyplot as plt

np.set_printoptions(precision=2, suppress=True)
# 11 columns inlcuding ID as first
names = np.array(['id', 'Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli', 'Mitoses', 'Class'])
df = pd.read_csv('datasets/breast-cancer-wisconsin.data.csv', names=names)   
df = df[df['Bare Nuclei'] != "?"]
# df['Bare Nuclei'] = df['Bare Nuclei'].astype(int)
x = np.array(df.loc[:, 'Clump Thickness':'Class'])
y = np.array(df.loc[:, 'Class'])

for i in x:
    print(i)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)
print(f'Pred shape{pred.shape}')
print('Model accuracy score: ', accuracy_score(Y_test, pred))

conf_matrix = confusion_matrix(Y_test, pred)
print(f'\nConfusion Matrix: \n{conf_matrix}')

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
xticklabels = knn.classes_, yticklabels = knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()