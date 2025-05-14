import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from matplotlib import pyplot as plt

np.set_printoptions(precision=2, suppress=True)

names = np.array(['class', 'Alcohol','Malic Acid','Ash','Acadlinity','Magnisium','Total Phenols','Flavanoids','NonFlavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline'])
df = pd.read_csv('datasets/wine.data.csv', names=names)
x = np.array(df.loc[:, 'Alcohol':])
y = np.array(df.loc[:, 'class'])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
accuracies = []

for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, pred)
    print(f'Model accuracy score for k = {i}: {accuracy}')
    accuracies.append(accuracy)

plt.scatter([i+1 for i in range(len(accuracies))], accuracies, color="red")
plt.plot([i+1 for i in range(len(accuracies))], accuracies, color="blue")

plt.ylabel("Accuracy")
plt.xlabel("K-values")
plt.show()