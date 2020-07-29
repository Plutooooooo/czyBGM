import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:128]
    x_data = StandardScaler().fit_transform(x_data)  # 把数据归一化
    y_data = data[:, 128]
    return x_data,

def pca(x_data,y_data):
    pca = PCA(n_components=3)
    pca.fit(x_data)
    X_reduction = pca.transform(x_data)
    return X_reduction

def dbscan(data):
    rs = []  # 存放各个参数的组合计算出来的模型评估得分和噪声比
    eps = np.arange(0.2, 4, 0.2)  # eps参数从0.2开始到4，每隔0.2进行一次
    min_samples = np.arange(2, 20, 1)  # min_samples参数从2开始到20

    best_score = 0
    best_score_eps = 0
    best_score_min_samples = 0

    for i in eps:
        for j in min_samples:
            try:  # 因为不同的参数组合，有可能导致计算得分出错，所以用try
                db = DBSCAN(eps=i, min_samples=j).fit(data)
                labels = db.labels_  # 得到DBSCAN预测的分类便签
                k = metrics.silhouette_score(data, labels)  # 轮廓系数评价聚类的好坏，值越大越好
                ratio = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
                rs.append([i, j, k, ratio, n_clusters_])
                if k > best_score:
                    best_score = k
                    best_score_eps = i
                    best_score_min_samples = j
            except:
                db = ''  # 这里用try就是遍历i，j 计算轮廓系数会出错的，出错的就跳过

    rs = pd.DataFrame(rs)
    rs.columns = ['eps', 'min_samples', 'score', 'ratio', 'n_clusters']
    sns.relplot(x="eps", y="min_samples", size='score', data=rs)
    sns.relplot(x="eps", y="min_samples", size='ratio', data=rs)
    plt.show()
    # plt.figure(figsize=(5, 5))
    # plt.scatter(data[:, 0], data[:, 1], c=result)

if __name__ == '__main__':
    dbscan(*load_data('F:/CZY/data.csv'))