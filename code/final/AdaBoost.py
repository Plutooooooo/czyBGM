import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,roc_curve
from matplotlib import colors

# ddl_heat=['#E3F1FF','#C7E3FF','#ABD5FF','#8FC8FF','#72BAFF','#56ACFF','#3A9EFF','#1E90FF']
ddl_heat = ['#F0F8FF', '#ADD8E6', '#87CEFA', '#00BFFF', '#1E90FF']
# ddl_heat = ['#DBDBDB','#DCD5CC','#DCCEBE','#DDC8AF','#DEC2A0','#DEBB91',\
#             '#DFB583','#DFAE74','#E0A865','#E1A256','#E19B48','#E29539']
ddlheatmap = colors.ListedColormap(ddl_heat)

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x_data = data[:, 0:128]
    y_data = data[:, 128]
    x_data = StandardScaler().fit_transform(x_data)  # 把数据归一化
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.3)
    return x_train, x_test, y_train, y_test


def gridSearch(x_train, y_train):
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'n_estimators':range(10,310,10)}]
    grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=3, n_jobs=-1,scoring='roc_auc',iid=False)
    # 训练模型
    grid_result = grid.fit(x_train, y_train)
    # 计算测试集精度
    print("Best: %f using %s" % (grid_result.best_score_, grid.best_params_))
    return grid_result.best_params_

def AdaBoost(x_train, x_test, y_train, y_test):
    params = gridSearch(x_train,y_train)
    model = AdaBoostClassifier(n_estimators=params['n_estimators'])
    plot_learning_curve(model, "Learning Curves(AdaBoost)", x_train, y_train)
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    cr = classification_report(y_test, y_predict)
    plot_classification_report(cr)

def plot_learning_curve(estimator, title, X, y,
                        ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('F:/CZY/results/all/AdaBoost_learning_curve.png')
    plt.show()
    return plt

def plot_classification_report(cr, title=None, cmap=ddlheatmap):
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []
    for line in lines[2:(len(lines) - 5)]:
        s = line.split()
        if (len(s) > 0):
            classes.append(s[0])
            value = [float(x) for x in s[1: len(s) - 1]]
            matrix.append(value)
    fig, ax = plt.subplots(1)
    for column in range(len(matrix[0])):
        for row in range(len(classes)):
            #             txt = matrix[row][column]
            ax.text(column, row, matrix[row][column], va='center', ha='center')
    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes) + 1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.savefig('F:/CZY/results/all/AdaBoost_report.png')
    plt.show()

if __name__ == '__main__':
    AdaBoost(*load_data('F:/CZY/data.csv'))