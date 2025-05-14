import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = np.array([[1, 5], [3, 2], [8, 4], [7, 14]])

def mean(arr):
    feature_avg = []
    length = len(arr)
    temp_avg = 0
    for i in range(len(arr[0])):
        for j in range(len(arr)):
            temp_avg += arr[j][i]
        feature_avg.append(temp_avg / length)
        temp_avg = 0

    print(f'Averages: {feature_avg}')
    return np.array(feature_avg)


def std(arr, avg):
    temp_sum = 0
    standards = []
    for i in range(len(arr[0])):
        for j in range(len(arr)):
            temp_sum += (arr[j][i] - avg[i]) ** 2
        temp_sum /= len(arr)
        temp_sum = np.sqrt(temp_sum)
        standards.append(temp_sum)
        temp_sum = 0

    print(f'Standards: {standards}')
    return np.array(standards)


def scale(arr, avg, standard):
    new_values = [[arr[i][j] for j in range(len(arr[i]))] for i in range(len(arr))]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            new_values[i][j] = (arr[i][j] - avg[j]) / standard[j]

    return np.array(new_values)


def inverse_scale(arr, avg, standard):
    new_values = [[arr[i][j] for j in range(len(arr[i]))] for i in range(len(arr))]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            new_values[i][j] = (arr[i][j] * standard[j]) + avg[j]
        
    return np.array(new_values)


avg = mean(data)
standard = std(data, avg)
s = scale(data, avg, standard)
i_s = inverse_scale(s, avg, standard)
print(f'Scaled data: \n{s}')
print(f'Inverse-scaled data: \n{i_s}')

scaler = StandardScaler()
built_in_standardization = scaler.fit_transform(data)
built_in_inv_standardization = scaler.inverse_transform(built_in_standardization)
print(f'Built in standardization: \n{built_in_standardization}')
print(f'Built in inverse standardization: \n{built_in_inv_standardization}')
