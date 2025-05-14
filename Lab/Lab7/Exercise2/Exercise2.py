import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

orig_df = pd.read_csv('datasets/golf.csv')
df = orig_df.copy()

le = preprocessing.LabelEncoder()
for i in orig_df.columns:
    df[i] = np.array(le.fit_transform(df[i]))

print(df.shape)

x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

model = GaussianNB()
model.fit(x, y)

data_points = pd.DataFrame([['Rainy', 'Hot', 'High', 'TRUE'], ['Sunny', 'Mild', 'Normal', 'FALSE'], ['Sunny', 'Cool', 'High', 'FALSE']])
for i in range(len(orig_df.columns[:-1])):
    le.fit(orig_df.iloc[:, i])
    data_points.iloc[:, i] = le.transform(data_points.iloc[:, i])

print(f'Data points after encoding:\n{data_points}')

new_x = np.array(data_points)
pred = model.predict(new_x)

for i in pred:
    if i == 0:
        print(f'{i}: No golf tomorrow')
    elif i == 1:
        print(f'{i}: Golf tomorrow')


