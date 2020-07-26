import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:128]
    x_data = StandardScaler().fit_transform(x_data)  # 把数据归一化
    y_data = data[:, 128]
    return x_data, y_data


def pca(x_data, y_data):
    pca = PCA(n_components=2)
    pca.fit(x_data)
    X_reduction = pca.transform(x_data)
    return X_reduction


def cluster(x):
    result = DBSCAN(eps=2, min_samples=10).fit_predict(x)
    # result = KMeans(n_clusters=5, random_state=9).fit_predict(x)
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], c=result)
    plt.show()


if __name__ == '__main__':
    cluster(pca(*load_data('F:/CZY/data.csv')))