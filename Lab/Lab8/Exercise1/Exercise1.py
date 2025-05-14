import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('datasets/speedLimits.csv')
speed = np.array(df.loc[:, 'Speed'])
ticket = np.array(df.loc[:, 'Ticket'])

x_train, x_test, y_train, y_test = train_test_split(speed, ticket, test_size=0.1, random_state=0)
x_train = np.array(x_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best_score = ('', 0)
for i in kernels:
    model = SVC(kernel=i).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(f'Accuracy score ({i}): {acc_score}')
    if acc_score > best_score[1]:
        best_score = (i, acc_score)

print(f'\nBest score: {best_score[0]}, {best_score[1]}')

for i in range(len(ticket)):
    if ticket[i] == 'NT':
        plt.scatter(speed[i], ticket[i], color='green', label='NT')
    else:
        plt.scatter(speed[i], ticket[i], color='red', label='T')

plt.ylabel('Ticket?')
plt.xlabel('Speed')
plt.title('Speed vs Ticket')
plt.show()