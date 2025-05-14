import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

np.set_printoptions(suppress=True)

df = pd.read_csv('datasets/Student-Pass-Fail.csv')

x = np.array(df.drop('Pass_Or_Fail', axis=1))
y = np.array(df.loc[:, 'Pass_Or_Fail'])

log_reg = LogisticRegression()
log_reg.fit(x, y)

log_odds = np.exp(log_reg.coef_)
print(f'Odds: {log_odds}')

data_points = np.array([[7, 28], [10, 34], [2, 39]])

y_pred = log_reg.predict(data_points)
y_prob = log_reg.predict_proba(data_points)[:, 1]

for i in range(len(y_prob)):
    if y_prob[i] > 0.5:
        print(f'Student {i+1} passes. Probability: {y_prob[i]}')
    else:
        print(f'Student {i+1} does not pass. Probability: {y_prob[i]}')

