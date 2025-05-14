import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

df_train = pd.read_csv('datasets/fashion-mnist_train.csv')
df_test = pd.read_csv('datasets/fashion-mnist_test.csv')
labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

x_train = np.array(df_train.iloc[:, 1:])
y_train = np.array(df_train.iloc[:, 0])
x_test = np.array(df_test.iloc[:, 1:])
y_test = np.array(df_test.iloc[:, 0])
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

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

print('\nPredictions')
# bag
item = cv2.cvtColor(cv2.imread('Lab/Lab6/bag.jpg'), cv2.COLOR_BGR2GRAY)
item = cv2.resize(item, (28, 28))
item = item.reshape(1, 28 * 28)

print(f'Actual: Bag (8)')
print(f'Prediction: {labels[model.predict(item)[0]]} ({model.predict(item)[0]})\n')

# trousers
item = cv2.cvtColor(cv2.imread('Lab/Lab6/trousers.bmp'), cv2.COLOR_BGR2GRAY)
item = cv2.resize(item, (28, 28))
item = item.reshape(1, 28 * 28)

print(f'Actual: Trouser (1)')
print(f'Prediction: {labels[model.predict(item)[0]]} ({model.predict(item)[0]})\n')



