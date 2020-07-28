from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer #标签二值化处理

def load_data(file_name):
    data=np.genfromtxt(file_name,delimiter=',')
    np.random.shuffle(data)
    x_data=data[:,0:128]
    y_data=data[:,128]
    x_train=x_data[0:1000,:]
    x_test=x_data[1000:,:]
    y_train=y_data[0:1000]
    y_test=y_data[1000:]
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test) = load_data('C:\\Users\\micha\\PycharmProjects\\Keras\\128data.csv')

print(x_train.shape)
print(y_train.shape)

x_train.reshape(1000,128)
x_test.reshape(-1,128)
y_train=LabelBinarizer().fit_transform(y_train)
y_test=LabelBinarizer().fit_transform(y_test)

#数据拆分
#x_train,x_test,y_train,y_test = train_test_split(x_data,y_data)

#构建模型，训练500周期
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100
                                        ), max_iter=500,activation='tanh')
mlp.fit(x_train,y_train)

#测试集准确率的评估
predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))
