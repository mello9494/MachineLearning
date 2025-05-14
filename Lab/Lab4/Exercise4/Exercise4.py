from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


houses = fetch_california_housing()
df = pd.DataFrame(houses.data, columns=houses.feature_names)
target = pd.DataFrame(houses.target)

x = np.array(df.iloc[:, :-2])
y = np.array(target).flatten()

print(x.shape, y.shape)

sns.set_theme(font_scale=0.5)
sns.pairplot(data=df.iloc[:-2], vars=df.columns[:-2], diag_kind='hist', height=1.5)
plt.show()
