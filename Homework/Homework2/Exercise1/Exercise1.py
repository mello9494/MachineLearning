import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

features = ['id','gender','ses','schtyp','prog','read','write','math','science','socst','honors','awards','cid']
df = pd.read_csv('datasets/hsbdemo.csv', names=features)


# function to convert non-int values to integers
def convert(feature):
    vals = {}
    index = 0
    # loop to associate non-int values with a numerical value
    for i in df.loc[1:, feature]:
        if i not in vals:
            vals[i] = index
            index += 1

    # convert values to their associated ints
    for i in range(1, len(df)):
        value = df.loc[i, feature]
        df.loc[i, feature] = str(vals[value])


def main():
    # test to see if value needs to be converted to numerical value
    for i in df:
        try:
            int(df.loc[1, i])
            str(df.loc[1, i])
        except ValueError:
            convert(i)

    K = 5
    x = np.array(df.loc[1:, 'gender':'awards'])
    y = np.array(df.loc[1:, 'prog'])
    x = np.delete(x, 3, axis=1) # remove the 'prog' column from x
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=3)

    knn = KNeighborsClassifier(n_neighbors = K)
    knn.fit(X_train, Y_train)
    pred = knn.predict(X_test)
    print(pred)
    print(Y_test)
    print(f'Pred shape{pred.shape}')
    print('Model accuracy score: ', accuracy_score(Y_test, pred))

    print('Incorrect predictions:')
    print('X\t\t\t\t\t\tY Predicted')
    for i in range(len(pred)):
        if pred[i] != Y_test[i]:
            print(f'{X_test[i]}  {Y_test[i]} {pred[i]}')


    conf_matrix = confusion_matrix(Y_test, pred)
    print(f'\nConfusion Matrix: \n{conf_matrix}')

    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
    xticklabels = knn.classes_, yticklabels = knn.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

main()
# print(df)