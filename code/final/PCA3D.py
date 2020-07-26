import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:128]
    y_data = data[:, 128]
    x_data = StandardScaler().fit_transform(x_data)  # 把数据归一化
    return x_data, y_data



def draw_graph(x_data,y_data):
    pca = PCA(n_components=3)
    pca.fit(x_data)
    X_reduction = pca.transform(x_data)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(2):
        ax.scatter(X_reduction[y_data == i, 0], X_reduction[y_data == i, 1], X_reduction[y_data == i, 2],alpha=0.8, label='%s' % i)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_graph(*load_data('F:/CZY/datasets/all/128data.csv'))
