import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


def plot(dataset, history_centroids, belongs_to):
    colors = ['r', 'k']

    fig, ax = plt.subplots()

    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        print(index, centroids)
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(1)
    plt.show()


def euclidean(p,q):
    # dist = 0.0
    # for i in range(len(p)):
    #     dist += (p[i]-q[i])**2
    # return np.sqrt(dist)
    return np.linalg.norm(p-q)


def kmeans(k, datapoint, epsilon = 0):
    history_centroids = []
    num_data, num_feature = datapoint.shape
    centroids = datapoint[np.random.randint(0, num_data - 1, size=k)]
    history_centroids.append(centroids)
    old_centroids = np.zeros(centroids.shape)
    label = np.zeros((num_data,1))
    norm = euclidean(centroids, old_centroids)
    iteration = 0

    while norm > epsilon:
        iteration += 1
        norm = euclidean(centroids, old_centroids)
        old_centroids = centroids
        for index_data, data_value in enumerate(datapoint):
            dist_vector = np.zeros((k, 1))
            for index_centroid, centroid in enumerate(centroids):
                dist_vector[index_centroid] = euclidean(centroid, data_value)
            label[index_data] = np.argmin(dist_vector)

        tmp_centroids = np.zeros((k, num_feature))

        for index in range(k):
            data_map = [x for x in range(len(label)) if label[x] == index]
            centroid = np.mean(datapoint[data_map], axis=0)
            tmp_centroids[index, :] = centroid

        centroids = tmp_centroids

        history_centroids.append(tmp_centroids)

    return centroids, history_centroids, label

def meanErrors(centroids, datapoint):
    errors = 0.0
    for data in datapoint:
        for centroid in centroids:
            errors += euclidean(centroid, data)
    return errors / len(data)


def ElbowMeth(kmax, datapoint, k = 1):
    sse_col = []
    while k <= kmax:
        sse = 0
        centroids, history_centroids, label = kmeans(k, datapoint)
        clustermean = meanErrors(centroids, datapoint)
        for data in datapoint:
            data_dist = euclidean(centroids, data)
            sse += math.pow(data_dist - clustermean, 2)
        sse_col.append(sse)
        k = k + 1
    return sse_col


data = pd.read_csv('Absenteeism_at_work.csv')
x = data.iloc[:, 5]
y = data.iloc[:, 6]
datapoint = []
for i in range(len(data)):
    datapoint.append([x[i], y[i]])

# datapoint = [[3,2],[2,2],[1,2],[0,1],[1,0],[1,1],[5,6],[7,7],[9,10],[11,13],[12,12],[12,13],[13,13]]

dataset = np.reshape(datapoint,(len(datapoint),2))

# dataset = np.loadtxt('durudataset.txt')
centroids, history_centroids, label = kmeans(2,dataset)
plot(dataset, history_centroids, label)


result = ElbowMeth(5, dataset)
k_point = [i for i in range(1,6)]
print(k_point)
plt.scatter(k_point, result)
plt.plot(k_point, result)
plt.show()
print(result)