import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:128]
    y_data = data[:, 128]
    x_data = StandardScaler().fit_transform(x_data)  # 把数据归一化
    return x_data, y_data



def draw_graph(x_data,y_data):
    # 绘制主成分增加对原数据的保留信息影响
    # pca = PCA(n_components=X_train.shape[1])
    # pca.fit(X_train)
    # plt.plot([i for i in range(X_train.shape[1])],
    #          [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(X_train.shape[1])])
    # plt.show()
    # 降维成2维，进行数据可视化
    pca = PCA(n_components=2)
    pca.fit(x_data)
    X_reduction = pca.transform(x_data)
    for i in range(2):
        plt.scatter(X_reduction[y_data == i, 0], X_reduction[y_data == i, 1], alpha=0.8, label='%s' % i)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_graph(*load_data('F:/CZY/datasets/all/128data.csv'))
