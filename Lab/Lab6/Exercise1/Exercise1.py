import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

df_train = pd.read_csv('datasets/fashion-mnist_train.csv')
df_test = pd.read_csv('datasets/fashion-mnist_test.csv')

x_train = np.array(df_train.iloc[:, 1:])
y_train = np.array(df_train.iloc[:, 0])
x_test = np.array(df_test.iloc[:, 1:])
y_test = np.array(df_test.iloc[:, 0])

print(df_train.shape, df_test.shape)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = LogisticRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
classification = classification_report(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)

print(f'Model score: {model.score(x_train, y_train)}')
print(f'Accuracy: {accuracy}')
print(f'Classification report: \n{classification}')
print(f'Confusion matrix: \n{conf_matrix}')

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(set(y_train)))
disp.plot()
plt.show()
