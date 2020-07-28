from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.recurrent import SimpleRNN,LSTM
from keras.optimizers import Adam
import numpy as np


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

print(f'x_train_shape:{x_train.shape}')#需要(60000,28,28)
print(f'y_train_shape:{y_train.shape}')
print(f'归一化系数：{np.max(x_train)}')

#x_train=x_train/255 #归一化
#x_test=x_test/255   #归一化

x_train-=np.mean(x_train,axis=0)
x_train/=np.std(x_train,axis=0)
x_test-=np.mean(x_test,axis=0)
x_test/=np.std(x_test,axis=0)

x_train=x_train.reshape(1000,8,16)
x_test=x_test.reshape(-1,8,16)

#one hot处理y_data
y_train=np_utils.to_categorical(y_train,num_classes=2)
y_test=np_utils.to_categorical(y_test,num_classes=2)

#数据长度
input_size=8
#序列长度
time_steps=16
#隐藏层神经元个数
cell_size=100

#创建RNN
model=Sequential()
#循环神经网络隐藏层 输入128个神经元 输出100个神经元    SimpleRNN可以改为LSTM
model.add(SimpleRNN(
    units=cell_size,
    input_shape=(input_size,time_steps)
))
#隐藏层-输出层
model.add(Dense(units=100,activation='tanh'))
model.add(Dense(units=100,activation='tanh'))
model.add(Dense(units=2,activation='softmax'))

#定义优化器
adam=Adam(lr=0.0001)

#编译模型
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=adam,loss='mse',metrics=['accuracy'])

#训练模型
#每次传入32个  迭代周期10  全部训练一次为一个周期
model.fit(x_train,y_train,batch_size=8,epochs=10)

#评估模型
loss,accuracy=model.evaluate(x_test,y_test)
print(f'loss:{loss}')
print(f'accuracy:{accuracy}')

from keras.utils.vis_utils import plot_model
model.save('RNN_Model_抖音音乐分类')
plot_model(model,to_file='CNN_MNIST_model.png')
