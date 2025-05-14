import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = pd.read_csv('datasets/balloons_extended.csv')

le = LabelEncoder()
for i in df.columns:
    df[i] = np.array(le.fit_transform(df[i]))

x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, pred)
acc_score = accuracy_score(y_test, pred)

print(f'Accuracy score: {acc_score}')
print(f'Confusion matrix: \n{conf_matrix}\n')

for imp, feat in zip(model.feature_importances_, df.columns[:-1]):
    print(f'{feat} importance: {imp}')

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
xticklabels = model.classes_, yticklabels = model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plot_tree(model, feature_names=[x for x in range(10)], filled=True, rounded=True)
plt.show()

text_rep = export_text(model)
print(text_rep)