from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

orig_df = pd.read_csv('datasets/balloons_2features.csv')
df = orig_df.copy()

le = LabelEncoder()
for i in df.columns[:-1]:
    df[i] = np.array(le.fit_transform(df[i]))

print(df.head())

x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

class Tree:
    def __init__(self, val=0, threshold=None, feature=None, left=None, right=None):
        self.val = val
        self.threshold = threshold
        self.feature = feature
        self.left = left
        self.right = right


tot_length = len(df)
poss = np.unique(df['Inflated'])
root_entropy = 0
for i in poss:
    tot_pos = len(df[df['Inflated'] == i])
    root_entropy -= (tot_pos / tot_length) * np.log2(tot_pos / tot_length)

print(f'Root entropy: {root_entropy}')

def ig(df, feature):
    tot_length = len(df)
    poss_feat = np.unique(df[feature])
    poss_res = np.unique(df.iloc[:, -1])
    entropies = []
    for i in poss_feat:
        tot_pos = df[df[feature] == i]
        entropy = 0
        for j in poss_res:
            temp = tot_pos[tot_pos.iloc[:, -1] == j]
            entropy -= (len(temp) / len(tot_pos)) * np.log2(len(temp) / len(tot_pos))
        entropy = (len(tot_pos) / tot_length) * entropy
        entropies.append(entropy)

    ig = root_entropy - (np.sum(entropies))
    return ig

print(f'Act IG: {ig(df, 'Act')}')
print(f'Age IG: {ig(df, 'Age')}')

# Built in
# print('Built in:')
# model = DecisionTreeClassifier()
# model.fit(x, y)

# data_points = pd.DataFrame([['Stretch', 'Adult']])
# for i in range(len(orig_df.columns[:-1])):
#     le.fit(orig_df.iloc[:, i])
#     data_points.iloc[:, i] = le.transform(data_points.iloc[:, i])

# data_points = np.array(data_points)
# y_pred = model.predict(data_points)
# print(f'Prediction: {y_pred}')