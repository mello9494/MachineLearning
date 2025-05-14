import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.set_printoptions(suppress=True)

df = pd.read_csv('datasets/Student-Pass-Fail.csv')

x = np.array(df.drop('Pass_Or_Fail', axis=1))
y = np.array(df.loc[:, 'Pass_Or_Fail'])

# split the data
test_size = float(input('Enter test size: '))
int_test_size = int(len(x) - (len(x) * test_size))

x_train = x[:int_test_size]
x_test = x[int_test_size:]
y_train = y[:int_test_size]
y_test = y[int_test_size:]

# fit the data
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

# compute accuracy score
acc_score = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        acc_score += 1

acc_score /= len(y_test) 
print(f'Accuracy score: {acc_score}')

# compute confusion matrix
dims = len(set(y))  # get number of different outputs possible
conf_matrix = np.zeros((dims, dims))
for i in range(len(y_test)):
    conf_matrix[y_test[i]][y_pred[i]] += 1
print(f'Confusion matrix:\n{conf_matrix}')

# Built in
# acc_score = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)