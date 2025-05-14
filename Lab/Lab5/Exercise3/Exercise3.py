import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.set_printoptions(suppress=True)

df = pd.read_csv('datasets/Bank-data.csv')

x = np.array(df.iloc[:, 1:-1])
y = np.array(df.iloc[:, -1])

log_reg = LogisticRegression()
log_reg.fit(x, y)

log_odds = np.exp(log_reg.coef_)
print(f'Odds: {log_odds}')

data_points = np.array([[1.335, 0, 1, 0, 0, 109], [1.25, 0, 0, 1, 0, 279]])

y_prob = log_reg.predict_proba(data_points)[:, 1]

for i in range(len(y_prob)):
    if y_prob[i] > 0.5:
        print(f'Client {i+1} subscribes. Probability: {y_prob[i]}')
    else:
        print(f'Client {i+1} does not subscribe. Probability: {y_prob[i]}')