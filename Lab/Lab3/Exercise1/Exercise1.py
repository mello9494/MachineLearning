import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

np.set_printoptions(suppress=True)

features = ['type', 'flour', 'milk', 'sugar', 'butter', 'egg', 'baking powder', 'vanila', 'salt']
df = pd.read_csv('datasets/recipes_muffins_cupcakes_scones.csv', names=features)

x = np.array(df.loc[1:, 'flour':'salt'])
y = np.array(df.loc[1:, 'type'])

pca = PCA(n_components=3)
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# variance ratio
explained_variance = None
for i in range(1, 9):
    pca = PCA(n_components=i)
    principalComponents = pca.fit_transform(x_scaled)
    explained_variance = pca.explained_variance_ratio_

print(f'Explained variance: {explained_variance}\n')
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Principle Components')
plt.ylabel('Cumulative Variance Ratio')
plt.show()

# PC1 and PC2 plots
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
x_variance = np.var(x_pca, axis=0)
x_variance_ratio = x_variance / np.sum(x_variance)
print(f'X variance ratio {x_variance_ratio}\n')

# heatmap
plt.matshow(pca.components_[:2], cmap='viridis')
plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(df.columns[1:])),df.columns[1:],rotation=65,ha='left')
plt.tight_layout()
plt.show()

# scatterplot
x_axis = x_pca[:, 0]
y_axis = x_pca[:, 1]

cdict = {0:'red', 1:'green', 2:'blue'}
labl = {0:'Muffin', 1:'Cupcake', 2:'Scone'}
alpha = {0:.3, 1:.5, 2:.7}
labels = []
for i in y:
    if i == 'Muffin':
        labels.append(0)
    elif i == 'Cupcake':
        labels.append(1)
    elif i == 'Scone':
        labels.append(2)

for i in np.unique(labels):
    ix = np.where(labels==i)
    plt.scatter(x_axis[ix], y_axis[ix], c=cdict[i], label=labl[i], alpha=alpha[i])

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.title('PCA = 2')
plt.show()

# histogram
muffin = np.array([[j for j in x[i]] for i in range(len(x)) if y[i] == 'Muffin'])
cupcake = np.array([[j for j in x[i]] for i in range(len(x)) if y[i] == 'Cupcake'])
scone = np.array([[j for j in x[i]] for i in range(len(x)) if y[i] == 'Scone'])
fig,axes =plt.subplots(2, 4, figsize=(12, 9))
ax=axes.ravel()
for i in range(8):
    _,bins=np.histogram([int(j) for j in x[:,i]], bins=25)
    ax[i].hist([int(j) for j in muffin[:,i]],bins=bins,color='r',alpha=.5)
    ax[i].hist([int(j) for j in cupcake[:,i]],bins=bins,color='g',alpha=0.4)
    ax[i].hist([int(j) for j in scone[:,i]],bins=bins,color='b',alpha=0.3)
    ax[i].set_title(y[i],fontsize=9)
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].set_yticks(())
ax[0].legend(['muffin','cupcake', 'scone'],loc='best',fontsize=8)
plt.tight_layout()
plt.show()

# correlation heatmap
s=sns.heatmap(df.iloc[1:, 1:].corr(),cmap='coolwarm') 
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()

# min and max variation
pca1 = pca.components_[0]
pca2 = pca.components_[1]

pca1_pos = np.sort([i for i in pca1 if i > 0])
pca1_neg = np.sort([i for i in pca1 if i < 0])

pca2_pos = np.sort([i for i in pca2 if i > 0])
pca2_neg = np.sort([i for i in pca2 if i < 0])

pca1_min_var = 0
pca2_min_var = 0

if pca1_pos[0] > 0 - pca1_neg[-1]:
    pca1_min_var = pca1_neg[-1]
else:
    pca1_min_var = pca1_pos[0]

if pca2_pos[0] > 0 - pca2_neg[-1]:
    pca2_min_var = pca2_neg[-1]
else:
    pca2_min_var = pca2_pos[0]

print(f'Max positive pca1 variance: {pca1_pos[-1]}')
print(f'Max negative pca1 variance: {pca1_neg[0]}')
print(f'Min pca1 variance: {pca1_min_var}')

print(f'Max positive pca2 variance: {pca1_pos[-1]}')
print(f'Max negative pca2 variance: {pca1_neg[0]}')
print(f'Min pca2 variance: {pca2_min_var}')
