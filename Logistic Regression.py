import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Data Generating
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

X_train = X[:500, :]
y_train = y[:500, :]
X_test = X[500:, :]
y_test = y[500:, :]

training_X = pd.DataFrame(X_train)
training_y = pd.DataFrame(y_train)
test_X = pd.DataFrame(X_test)
test_y = pd.DataFrame(y_test)

w = Variable(torch.FloatTensor(np.zeros((30,1))), requires_grad = True)
b = Variable(torch.FloatTensor(np.array([[0]])), requires_grad = True)
print('Initial value of w =', w.T)
print('===================================================')
print('Initial value of b =', b)

# Vectorization
X_vec = Variable(torch.FloatTensor(np.array(training_X)))  # N * 30
y_vec = Variable(torch.LongTensor(np.array(training_y))).reshape(-1,1)  # N*1

Iter_times = 100000
alpha = 0.000015
loss_list = []

for i in range(Iter_times):

    # torch.mm是矩阵乘法
    z = torch.mm(X_vec, w) + b  # 500*1
    # torch.sigmoid 激活函数
    y_hat = torch.sigmoid(z)  # 500*1
    # 计算交叉熵损失函数
    loss_vec = -(y_vec * torch.log(y_hat) + (1.0 - y_vec) * torch.log(1.0 - y_hat))
    loss = torch.mean(loss_vec)

    # 获取梯度
    loss.backward()
    grad_w = w.grad
    grad_b = b.grad

    # 学习率随梯度下降次数增加而减少，避免震荡
    alpha_temp = alpha / (1 + 0.001 * i)

    # 梯度下降，更新参数
    w.data = w.data - alpha_temp * grad_w
    b.data = b.data - alpha_temp * grad_b

    # 清除原有梯度，不清零的话，下次计算的梯度会和当前计算的梯度叠加
    # 导致模型不收敛
    w.grad.data.zero_()
    b.grad.data.zero_()

    print(i+1, 'iterations have been completed!')
    print('     -> Now w1 =', w[0, 0])
    print('     -> Now w2 =', w[1, 0])
    print('     -> Now b =', b[0, 0])
    print('     -> Loss', loss)
    print('=================================================')

    loss_list.append(loss)
    length = loss_list.__len__()
    if torch.abs(loss_list[length - 1] - loss_list[length - 2]) < 10 ** (-5) and length >= 2:
        break


# Visualization of the Cross Entropy Loss Function
plt.figure(figsize = (16,8))
length = loss_list.__len__()
print('The length of loss_list is', length)
plt.plot(np.arange(1,201,1), loss_list[0:200], 'black')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# Prediction on the Test Set and Model Evaluation
X_vec_test = Variable(torch.FloatTensor(np.array(test_X)))
y_vec_test = Variable(torch.FloatTensor((np.array(test_y)))).reshape(-1,1)
z_test = torch.mm(X_vec_test, w) + b
y_pred = torch.sigmoid(z_test)

# torch.relu
y_pred = torch.relu(y_pred - 0.50).T
y_pred[y_pred > 0.0] = 1.0
print(y_pred)

# 降维， accuracy_score为sklearn库里的不识别tensor变量，转成numpy
# 求过梯度的用.detach.numpy()转为numpy
# 没求过梯度的用.cpu.numpy()转为numpy
y_pred_np = y_pred.detach().numpy()
y_pred_np = np.squeeze(y_pred_np)
print('Shape of y_pred_np', y_pred_np.shape)

accuracy = accuracy_score(y_pred_np, test_y)
print('The accuracy score is:', accuracy)






