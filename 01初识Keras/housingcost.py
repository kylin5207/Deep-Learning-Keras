from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
""" 回归预测波士顿房价"""

def build_model():
    """构建模型"""
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #mse均方误差，mae平均绝对误差
    model.compile(optimizer='rmsprop', loss = 'mse', metrics=['mae'])
    return model

#1. 获取训练数据集和测试数据集
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#2. 标准化数据集
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
#使用训练集上得到的均值和标准差对测试集进行标准化
test_data -= mean
test_data /= std

#3. 构建网络
#使用两个隐藏层，每层有64单元的神经网络
model = build_model()

#4. 模型拟合,使用k折交叉验证来验证模型的拟合效果
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("当前第{:}折".format(i))
    #准备K折交叉验证的验证集
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i+1) * num_val_samples]
    #准备K折交叉验证的训练集, 利用concatenate((a1,a2,..),axis=)按照axis轴连接数组，组成一个新的array
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],
                                         train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i+1) * num_val_samples:]],
                                        axis=0)

    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
