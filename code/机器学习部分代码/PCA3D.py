from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:1024]
    y_data = data[:, 1024]
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
    pca = PCA(n_components=3)
    pca.fit(x_data)
    X_reduction = pca.transform(x_data)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(2):
        ax.scatter(X_reduction[y_data == i, 0], X_reduction[y_data == i, 1], X_reduction[y_data == i, 2],alpha=0.8, label='%s' % i)
    plt.legend()
    plt.savefig('F:/CZY/results/mensclothing/PCA3D.png')
    plt.show()




if __name__ == '__main__':
    draw_graph(*load_data('F:/CZY/datasets/mensclothing/1024data.csv'))
