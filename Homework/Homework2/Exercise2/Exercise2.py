import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    x = StandardScaler().fit_transform(x)

    explained_variance = None
    for i in range(1, 11):
        pca = PCA(n_components=i)
        principalComponents = pca.fit_transform(x)
        explained_variance = pca.explained_variance_ratio_

    print(f'Variance: {explained_variance}')
    print(f'Cum sum: {np.cumsum(explained_variance)}')
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Principle Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.show()


main()