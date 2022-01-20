from re import A
from time import time
from turtle import forward
import numpy as np

#单个神经元
class neuron():
    def __init__(self, input_dim, g = 'relu'):
        #输入的维数
        self.input_dim = input_dim
        #权值矩阵初始化, n * 1的列向量
        tmp = np.ones([input_dim, 1])
        self.weight_arr = tmp
        #激活函数选择
        g_list = ['relu', 'sigmoid']
        self.g = 0
        for i in range(0, len(g_list)):
            if i == g_list[i]:
                self.g == i

    #计算经过权值矩阵加权后的变量,设输入为n维行向量
    #若有m组数据，则输出应为m * 1的列向量
    def get_z(self, a_input):
        #print(a_input)
        #print(self.weight_arr)
        return a_input @ self.weight_arr

    def sigmoid(self, z_input):
        return 1 / (1 + np.exp(- z_input))

    def dy_sigmoid(self, z_input):
        tmp = np.exp(- z_input)
        return tmp / np.power(tmp, 2)

    def relu(self, z_input):
        m = z_input.shape[0]
        for i in range(0, m):
            if z_input[i, 0] < 0:
                z_input[i, 0] = 0
        return z_input

    def dy_relu(self, z_input):
        m = z_input.shape[0]
        for i in range(0, m):
            if z_input[i, 0] > 0:
                z_input[i, 0] = 1
            else:
                z_input[i, 0] = 0
        return z_input

    def g_fun(self, z_input):
        if self.g == 0:
            return self.relu(z_input)
        elif self.g == 1:
            return self.sigmoid(z_input)
    
    def dg_fun(self, z_input):
        if self.g == 0:
            return self.dy_relu(z_input)
        else:
            return self.dy_sigmoid(z_input)

class network():
    def __init__(self, input_dim, layers, neuron_num, lr = 0.001, g = 'relu'):
        self.layers = layers
        self.lr = lr
        self.g = g
        self.neuron_num = []
        self.a_output = []
        self.z_input = []
        self.dz = []
        self.w_arr = []
        self.neuron_net = []
        self.bias = []
        #各层的神经元数量，从1开始计
        self.neuron_num.append(-1)
        for i in range(0, layers):
            self.neuron_num.append(neuron_num[i])
        tmp = [-1]
        self.neuron_net.append(tmp)
        tmp = np.zeros([1, 1])
        self.bias.append(tmp)
        self.w_arr.append(tmp)
        #输入层
        tmp = []
        for i in range(0, self.neuron_num[1]):
            tmp.append(neuron(input_dim, g))
        self.neuron_net.append(tmp)
        self.w_arr.append(np.ones([input_dim, self.neuron_num[1]]))
        self.net_build()
    
    def net_build(self):
        #初始化各层的连接，各层权值矩阵, 每个神经元的权值对应一个列向量
        for i in range(2, self.layers + 1):
            tmp = []
            w_init = np.ones([self.neuron_num[i - 1], self.neuron_num[i]])
            for j in range(0, self.neuron_num[i]):
                tmp.append(neuron(self.neuron_num[i - 1], self.g))
            self.neuron_net.append(tmp)
            self.w_arr.append(w_init)
        #初始化偏置层,每一层的偏置为n * 1的列向量
        for i in range(1, self.layers + 1):
            tmp = np.ones([self.neuron_num[i], 1])
            self.bias.append(tmp)

    def forward(self, x):
        u = x.copy()
        m = u.shape[0]
        self.a_output.clear()
        self.z_input.clear()
        self.dz.clear()
        self.a_output.append(u)
        self.dz.append(u)
        self.z_input.append(u)
        #z_input, a_output, dz均为m * n的矩阵
        #w_arr[i]为n1 * n的矩阵
        #z_input[i] = a_output[i - 1] * w_arr[i] + bias[i]
        #a_ouput[i] = g(z_input[i])
        #dz[i] = dg(z_input[i])
        # !!! np.array矩阵的切片是一维数组 ！！！
        for i in range(1, self.layers + 1):
            z_t = np.zeros([m, self.neuron_num[i]])
            a_t = np.zeros([m, self.neuron_num[i]])
            dz_t = np.zeros([m, self.neuron_num[i]])
            for j in range(0, self.neuron_num[i]):
                z_t[ : , j] = (self.neuron_net[i][j].get_z(self.a_output[i - 1]) + self.bias[i][j, 0]).reshape(-1)
                a_t[ : , j] = (self.neuron_net[i][j].g_fun(z_t[ : , j].reshape(-1, 1))).reshape(-1)
                dz_t[ : , j] = (self.neuron_net[i][j].dg_fun(z_t[ : , j].reshape(-1, 1))).reshape(-1)
            self.z_input.append(z_t)
            self.a_output.append(a_t)
            self.dz.append(dz_t)
        return self.a_output[self.layers]

    def loss_function(self, x, y):
        y_pre = self.forward(x)
        tmp = np.linalg.norm(y_pre - y)
        return 0.5 * tmp * tmp

    def delta(self, x, y):
        #delta[i]为损失函数对a[i]的导数，维度为n * 1的列向量
        delta_list = []
        delta_list.append(np.zeros([1, 1]))
        y_pre = self.forward(x)
        for i in range(1, self.layers + 1):
            tmp = np.zeros([self.neuron_num[i], 1])
            delta_list.append(tmp)
        #计算最后一层的delta
        delta_list[self.layers] = y_pre - y
        #反向计算其他层的delta
        # for i in range(1, self.layers + 1):
        #     print(i)
        #     print(self.w_arr[i])
        for i in range(self.layers - 1, 0, -1):
            tmp = self.w_arr[i + 1] @ delta_list[i + 1]
            #print(tmp)
            delta_list[i] = tmp * self.dz[i].T
        return delta_list
    
    def gradient_descent(self, x, y, times = 1, obs = False):
        #训练次数为times
        for k in range(0, times):
            delta_list = self.delta(x, y)
            #更新偏置
            for i in range(1, self.layers + 1):
                self.bias[i] = self.bias[i] - self.lr * delta_list[i]
                self.w_arr[i] = self.w_arr[i] - self.lr * (delta_list[i] @ self.a_output[i - 1]).T
            if obs == True:
                print(self.loss_function(x, y))
    def train(self, X, Y, times):
        m = X.shape[0]
        for k in range(0, times):
            r = np.random.randint(m)
            x = X[r, : ].reshape(1, -1)
            y = Y[r, : ].reshape(-1, 1)
            # print(x)
            # print(y)
            self.gradient_descent(x, y)

input_dim = 2
layers = 3
neuron_num = [1, 2, 1]
data_X = np.array([[-2, -2], [-2, 0], [0, -2], [-1, -2], [0, 0], [1, 0], [0, 1], [1, 2], [1, 3]])
data_Y = np.array([[8, 4, 4, 5, 0, 1, 1, 5, 10]]).T
x = np.array([[2, 3]])
y = np.array([[13]])
t = network(input_dim, layers, neuron_num)
# print(x)
# print(y)
#t.gradient_descent(x, y, 100)
t.train(data_X, data_Y, 100)
x_test = np.array([[-2, -2]])
print(t.forward(x_test))        

    
        