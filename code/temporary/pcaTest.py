from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
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
    # 把64维降维2维，进行数据可视化
    pca = PCA(n_components=2)
    pca.fit(x_data)
    X_reduction = pca.transform(x_data)
    for i in range(10):
        plt.scatter(X_reduction[y_data == i, 0], X_reduction[y_data == i, 1], alpha=0.8, label='%s' % i)
    plt.legend()
    plt.show()


    # pca = PCA(n_components=2)
    # pca.fit(x_data)
    # reduced_X = pca.transform(x_data)
    # red_x, red_y = [], []
    # blue_x, blue_y = [], []
    # green_x, green_y = [], []
    # yellow_x, yellow_y = [], []
    #
    #
    # for i in range(len(reduced_X)):
    #     if y[i] == 0:
    #         red_x.append(reduced_X[i][0])
    #         red_y.append(reduced_X[i][1])
    #     elif y[i] == 1:
    #         blue_x.append(reduced_X[i][0])
    #         blue_y.append(reduced_X[i][1])
    #     else:
    #         green_x.append(reduced_X[i][0])
    #         green_y.append(reduced_X[i][1])
    #
    # plt.scatter(red_x, red_y, c='r', marker='x')
    # plt.scatter(blue_x, blue_y, c='b', marker='D')
    # plt.scatter(green_x, green_y, c='g', marker='.')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    draw_graph(*load_data('F:/CZY/data.csv'))
