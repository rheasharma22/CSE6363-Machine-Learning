from math import sqrt
import numpy as np
import pandas as pd

data = pd.read_csv('iris.data')
values_col = data.columns[data.columns.str.startswith('I')]
classlabel = data[values_col]
flat_list = [item for sublist in classlabel.values for item in sublist]

lst = []
for i in flat_list:
    if i =='Iris-setosa':
        lst.append(0)
    elif i =='Iris-versicolor':
        lst.append(1)
    elif i =='Iris-virginica':
        lst.append(2)
s = np.array(lst)
features = data.drop(data.columns[4], axis=1)
train_data = features.values

def centroid_init(k = 3):
    np.random.seed(0)
    cen = np.random.random((k, 4))
    return cen

initialcen = centroid_init() * 5

def paired_distance(datapoint, cent):
    return sqrt(np.sum((datapoint - cent) * (datapoint - cent)))

#Function for calculating K-means along with training data and iterations
def K_Means_clustering(X_train, iterations):
    size = X_train.shape[0]
    centroid = centroid_init(3) * 5
    kvalue = centroid.shape[0]
    distance = np.zeros([size, kvalue])
    class_assign = np.zeros([size, ])
    cenn = centroid
    for t in range(iterations):
        for r in range(0, size):
            for c in range(0, kvalue):
                distance[r][c] = paired_distance(X_train[r], centroid[c])
        class_assign = (np.argmin(distance, axis=1)).reshape((-1,))
        cenn = np.concatenate((cenn, centroid))
        print("The centroid values for iteration",t+1,"are:")
        print(centroid)
        print("##################################################################################")
        for c in range(0, kvalue):
            temp = np.zeros([1, 4])
            count = 0
            for r in range(0, size):
                temp = temp + 0.98 * (class_assign[r] == c) * X_train[r] + 0.02 * (class_assign[r] != c) * X_train[r]
                count = count + 0.98 * (class_assign[r] == c) + 0.02 * (class_assign[r] != c)
            centroid[c] = (temp.reshape((-1,))) / count
    return centroid, class_assign, cenn

centroid, class_assign, cenn = K_Means_clustering(train_data, iterations = 10)
print("\nPredicted results for K means (labels):")
print(class_assign)
err = np.count_nonzero(class_assign - s)/len(flat_list)
print("The error is:", err*100, "%")
acc = 1 - err
print("The accuracy is:", acc*100, "%")