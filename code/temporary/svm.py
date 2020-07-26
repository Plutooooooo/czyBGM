from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,roc_curve


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:128]
    y_data = data[:, 128]
    x_data = StandardScaler().fit_transform(x_data)  # 把数据归一化
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.3)
    return x_train, x_test, y_train, y_test


def gridSearch(x_train, y_train):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced', )
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    grid_result = grid.fit(x_train, y_train)
    # 计算精度
    print("Best: %f using %s" % (grid_result.best_score_, grid.best_params_))
    return grid_result.best_params_
    # means = grid_result.cv_results_['mean_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("%f  with:   %r" % (mean, param))

def svm_c(x_train, x_test, y_train, y_test):
    best_params = gridSearch(x_train,y_train)
    model = SVC(kernel='rbf', class_weight='balanced',C=float(best_params['C']),gamma=float(best_params['gamma']),probability=True)
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    y_predictH = model.predict_proba(x_test)
    # 输出测试数据集的精确率、召回率、F1值、AUC值，画出ROC曲线
    print('精准率:', precision_score(y_test, y_predict,average='micro'))
    print('召回率:', recall_score(y_test, y_predict,average='micro'))
    print('F1率:', f1_score(y_test, y_predict,average='micro'))
    print('AUC:', roc_auc_score(y_test, y_predictH[:, -1]))

    fpr, tpr, theta = roc_curve(y_test, y_predictH[:, -1])
    print('fpr=n', fpr)
    print('tpr=n', tpr)
    print('theta=n', theta)
    # 画出ROC曲线
    plt.plot(fpr, tpr)
    plt.show()


if __name__ == '__main__':
    svm_c(*load_data('F:/CZY/newData.csv'))
