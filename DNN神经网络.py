import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Data Generating
x1_Pos = []
x2_Pos = []
y_Pos = []

for i in range(1000):
    temp = 4.0 * np.random.rand() - 2.0
    y_Pos.append(1)
    x1_Pos.append(temp)
    if i % 2 == 0:
        x2_Pos.append(np.sqrt(4.0 - temp ** 2) + 0.3 * np.random.randn())
    elif i % 2 == 1:
        x2_Pos.append(-np.sqrt(4.0 - temp ** 2) + 0.3 * np.random.randn())

# =============================================================
x1_Neg = []
x2_Neg = []
y_Neg = []

for i in range(1000):
    temp = 10.0 * np.random.rand() - 5.0
    y_Neg.append(0)
    x1_Neg.append(temp)
    if i % 2 == 0:
        x2_Neg.append(np.sqrt(25.0 - temp ** 2) + 0.3 * np.random.randn())
    elif i % 2 == 1:
        x2_Neg.append(-np.sqrt(25.0 - temp ** 2) + 0.3 * np.random.randn())

# ==============================================================
plt.figure(figsize = (12,5))
plt.scatter(x1_Pos, x2_Pos, color = 'black', label = 'Class 1', alpha = 0.5)
plt.scatter(x1_Neg, x2_Neg, color = 'red', label = 'Class 2', alpha = 0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid()
plt.show()

Dict = {'x1': x1_Pos + x1_Neg, 'x2': x2_Pos + x2_Neg, 'y': y_Pos + y_Neg}
DataTrain = pd.DataFrame(Dict)
print(DataTrain)

# ==================================================================
x1_Pos = []
x2_Pos = []
y_Pos = []

for i in range(250):
    temp = 4.0 * np.random.rand() - 2.0
    y_Pos.append(1)
    x1_Pos.append(temp)
    if i % 2 == 0:
        x2_Pos.append(np.sqrt(4.0 - temp ** 2) + 0.3 * np.random.randn())
    elif i % 2 == 1:
        x2_Pos.append(-np.sqrt(4.0 - temp ** 2) + 0.3 * np.random.randn())

# =============================================================
x1_Neg = []
x2_Neg = []
y_Neg = []

for i in range(250):
    temp = 10.0 * np.random.rand() - 5.0
    y_Neg.append(0)
    x1_Neg.append(temp)
    if i % 2 == 0:
        x2_Neg.append(np.sqrt(25.0 - temp ** 2) + 0.3 * np.random.randn())
    elif i % 2 == 1:
        x2_Neg.append(-np.sqrt(25.0 - temp ** 2) + 0.3 * np.random.randn())

Dict = {'x1': x1_Pos + x1_Neg, 'x2': x2_Pos + x2_Neg, 'y': y_Pos + y_Neg}
DataTest = pd.DataFrame(Dict)
print(DataTest)

# ===================================================================

# Model training
# 2-layer DNN
# x_bar 1*2
# w1 2 * 10, b1 1* 10
# w2 10*2, b2 1*2
# z1 = sigmoid(xw1 + b1), z2 = sigmoid(sigmoid(xw1 + b1) + b2)
# y_hat = softmax(z2)
# Loss(w1,w2,b1,b2) = 1/N * omega_{1-N)(yi*log(y_hat) + (1 - yi) * log(1 - y_hat))
# k = 0,1,2,...
# w1_k+1 = w1_k - alpha_k * dloss/dw1
# w2_k+1 = w2_k - alpha_k * dloss/dw2
# b1_k+1 = b1_k - alpha_k * dloss/db1
# b2_k+1 = b2_k - alpha_k * dloss/db2

class DeepNeuralNetworkModel(nn.Module):
    # Constructor of the class, input_dim1, output_dim2指输入特征和输出特征的维度
    def __init__(self, input_dim1, output_dim1, input_dim2, output_dim2):
        # output_dim1 = input_dim2
        # super这一行必须要写，官方语言
        super(DeepNeuralNetworkModel, self).__init__()

        # Fully connected layer 1 全联接层
        self.FC_layer1 = nn.Linear(input_dim1, output_dim1)
        # nn.init.constant_(self.FC_layer1.weight, 0.1)
        # nn.init.constant_(self.FC_layer1.bias, -0.1)

        # Fully connected layer 2
        self.FC_layer2 = nn.Linear(input_dim2, output_dim2)
        # nn.init.constant_(self.FC_layer2.weight, 0.1)
        # nn.init.constant_(self.FC_layer2.bias, -0.1)

        # Activation function - sigmoid()
        self.act_sig = nn.Sigmoid()

    # Forward propagation function
    def forward(self, x):
        z1_ = self.FC_layer1(x)
        z1 = self.act_sig(z1_)

        z2_ = self.FC_layer2(z1)
        z2 = self.act_sig(z2_)

        return z2

X_vec = Variable(torch.FloatTensor(np.array(DataTrain[['x1', 'x2']])))
y_vec = Variable(torch.LongTensor(np.array(DataTrain['y']))).reshape(-1, 1)  # N*1

alpha = 0.2
DNN_Model = DeepNeuralNetworkModel(2, 10, 10, 2)
# SGD优化器为随机梯度下降优化器，原理和笔记一致
optimizer = torch.optim.SGD(DNN_Model.parameters(), lr = alpha)
loss_function = nn.CrossEntropyLoss()

# Dynamically change the learning rate
def adjust_learning_rate(optimizer, epoch):
    lr = alpha / (1 + 0.00001 * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# =================================================================
# Training
Iter_times = 200000
loss_list = []
for i in range(Iter_times):
    # forward propagation
    outputs = DNN_Model.forward(X_vec)

    #compute loss
    loss = loss_function(outputs, torch.squeeze(y_vec))

    # backward propagation
    loss.backward()

    # update parameters
    optimizer.step()

    # Rest grad to 0
    optimizer.zero_grad()

    if (i + 1) % 500 == 0:
        print(i + 1, 'Iteration have been completed')
        print('     -> Now loss', loss)
        print('======================================')

    # update learning rate
    adjust_learning_rate(optimizer, i)

    # update loss
    loss_list.append(loss)

    length = loss_list.__len__()
    if (torch.abs(loss_list[length - 1] - loss_list[length -2]) < 10 ** (-15) and length >= 2):
        break

# ===========================================================================
# Visualization of cross entropy loss function
plt.figure(figsize=(14, 6))
length = loss_list.__len__()
print('The length of loss_list is', length)
plt.plot(np.arange(1, 20001), loss_list[:20000], 'black')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# ==========================================================================
# Prediction on the Test Set and Model Evaluation
X_vec_test = Variable(torch.FloatTensor(np.array(DataTest[['x1','x2']])))
y_vec_test = Variable(torch.LongTensor(np.array(DataTest['y']))).reshape(-1,1)
# 走一次前向传播
pred = DNN_Model.forward(X_vec_test)
# sigmoid输出的一定是0-1之间，不用走softmax了
pred_vec = pred[:, 1]
pred_vec[pred_vec > 0.5] = 1
pred_vec[pred_vec <= 0.5] = 0


y_pred_np = y_vec_test.detach().numpy()
y_pred_np = np.squeeze(y_pred_np)
print('shape of y_pred_np:', y_pred_np.shape)

pred_vec_np = pred_vec.detach().numpy()
pred_vec_np = np.squeeze(pred_vec_np)
print('shape of pred_vec_np:', pred_vec_np.shape)

accuracy = accuracy_score(pred_vec_np, y_pred_np)
print('The accuracy score is:', accuracy)
