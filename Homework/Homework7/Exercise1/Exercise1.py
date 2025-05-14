import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

normalEmails = ['Homework/Homework7/HW7_files/train_N_I.txt', 'Homework/Homework7/HW7_files/train_N_II.txt', 'Homework/Homework7/HW7_files/train_N_III.txt']
spamEmails = ['Homework/Homework7/HW7_files/train_S_I.txt', 'Homework/Homework7/HW7_files/train_S_II.txt', 'Homework/Homework7/HW7_files/train_S_III.txt']
testEmails = ['Homework/Homework7/HW7_files/testEmail_I.txt', 'Homework/Homework7/HW7_files/testEmail_II.txt']

fileWordsCount = []
wordsN = {}
wordsS = {}

# read the normal emails
for i in range(len(normalEmails)):
    words = []
    with open (normalEmails[i], 'r') as file:
        words = file.read().split()
        file.close()

    wordsCount = Counter(words)
    fileWordsCount.append(wordsCount)
    for i in wordsCount:
        if i in wordsN:
            wordsN[i] += wordsCount[i]
        else:
            wordsN[i] = wordsCount[i]

# read the spam emails
for i in range(len(spamEmails)):
    words = []
    with open (spamEmails[i], 'r') as file:
        words = file.read().split()
        file.close()

    wordsCount = Counter(words)
    fileWordsCount.append(wordsCount)
    for i in wordsCount:
        if i in wordsS:
            wordsS[i] += wordsCount[i]
        else:
            wordsS[i] = wordsCount[i]

# convert list to dataframe
df = pd.DataFrame(fileWordsCount).fillna(0)
# add label to rows
df['norm_or_spam'] = ['Normal', 'Normal', 'Normal', 'Spam', 'Spam', 'Spam']

x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

model = GaussianNB(priors=[0.73, 0.27])
model.fit(x, y)

# reset the wordCounts array for test emails
fileWordsCount = []

# read the test emails
for i in range(len(testEmails)):
    words = []
    with open (testEmails[i], 'r') as file:
        words = file.read().split()
        file.close()

    fileWordsCount.append(Counter(words))

temp_df = pd.DataFrame(fileWordsCount, columns=df.columns).fillna(0)
new_x = np.array(temp_df.iloc[:, :-1])

pred = model.predict(new_x)
print(f'Predicted values: {pred}')

# graphs
plt.bar(wordsN.keys(), wordsN.values())
plt.title('Normal Words Frequencies')
plt.show()

plt.bar(wordsS.keys(), wordsS.values())
plt.title('Spam Words Frequencies')
plt.show()

