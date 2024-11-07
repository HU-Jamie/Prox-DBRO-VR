import numpy as np
from LoadMNIST import getData, data_redistribute
from numpy import linalg as LA
import copy
import os
import sys

"""
定义one_hot函数 + 随机（批量）梯度计算 + 构建损失计算函数 + 构建预测计算模型 + 构建精确度计算模型 + 构建求方差模型 + 步长
"""


class Softmax(object):
    def __init__(self, m_agent, limited_labels=False, balanced=True, setting=None):
        self.m = m_agent
        self.limited_labels = limited_labels
        # self.X_train, self.Y_train = getData('data/MNIST/train-images.idx3-ubyte', 'data/MNIST/train-labels.idx1-ubyte')
        # self.X_test, self.Y_test = getData('data/MNIST/t10k-images.idx3-ubyte', 'data/MNIST/t10k-labels.idx1-ubyte')
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_data()
        self.N = self.X_train.shape[0]  ## total number of data samples
        """
        the shape of training set X: (60000, 784)
        the shape of label set: (60000,)
        """
        self.b = int(self.N / self.m)  ## average local samples
        self.balanced = balanced
        if balanced == False:
            self.split_vec = np.sort(np.random.choice(np.arange(1, self.N), self.m - 1, replace=False))
        self.X, self.Y, self.data_distr = self.distribute_data( shuffle = None )
        # self.X, self.Y, self.data_distr = self.distribute_data("shuffle the data set")
        self.n0 = 10  ## numbers of the feature
        self.n1 = self.X_train.shape[1]  ## dimensions of the feature
        self.dim = self.n1  ## dimensions of the feature
        self.reg_l1 = 1/self.N
        self.reg_l2 = 1/self.N
        self.L, self.kappa = self.smooth_scvx_parameters()
        self.setting = setting
        """
        Initialize the solver of softmax regression

        :param para: model parameter, shape(num_cats, num_features)
        :param config: configuration, type: dictionary
        """
    def load_data(self):
        if os.path.exists('mnist_10c.npz'):
            print( 'Data exists!' )
            data = np.load('mnist_10c.npz', allow_pickle=True)
            X_train = data['X_train']
            Y_train = data['Y_train']
            X_test = data['X_test']
            Y_test = data['Y_test']
        else:
            print( 'downloading data' )
            X_train, Y_train = getData('data/MNIST/train-images.idx3-ubyte', 'data/MNIST/train-labels.idx1-ubyte')
            X_test, Y_test = getData('data/MNIST/t10k-images.idx3-ubyte', 'data/MNIST/t10k-labels.idx1-ubyte')
            np.savez_compressed('mnist_10c', X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
        Y_train = Y_train.astype(int)
        Y_test = Y_test.astype(int)
        return X_train, Y_train, X_test, Y_test

    def distribute_data(self, shuffle):
        if shuffle == None:
            if self.balanced == True:
                X = np.array(np.split(self.X_train, self.m, axis=0))
                Y = np.array(np.split(self.Y_train, self.m, axis=0))
            if self.balanced == False:  ## random distribution
                X = np.array(np.split(self.X_train, self.split_vec, axis=0))
                Y = np.array(np.split(self.Y_train, self.split_vec, axis=0))
        else:
            print('shuffle the data set')
            if self.balanced == True:
                # shufflling data
                seed_shuffle = 10
                np.random.seed(seed_shuffle)
                print("seed of shuffling =", seed_shuffle)
                shuffled_indices = np.random.permutation(self.N)
                # 根据打乱后的索引重排X_train和Y_train
                X_shuffled = self.X_train[shuffled_indices]
                Y_shuffled = self.Y_train[shuffled_indices]
                X = np.array(np.split(X_shuffled, self.m, axis=0))
                Y = np.array(np.split(Y_shuffled, self.m, axis=0))
            if self.balanced == False:  ## random distribution
                X = np.array(np.split(self.X_train, self.split_vec, axis=0))
                Y = np.array(np.split(self.Y_train, self.split_vec, axis=0))
        data_distribution = np.array([len(_) for _ in X])
        return X, Y, data_distribution

    def smooth_scvx_parameters(self):
        Q = np.matmul(self.X_train.T, self.X_train) / self.N
        L_F = max(abs(LA.eigvals(Q))) / 4
        L = L_F + self.reg_l2
        kappa = L / self.reg_l2
        return L, kappa

    def one_hot(self, label):
        """
        Turn the label into the form of one-hot

        :param label: scalar
        """
        n = label.shape[0]  # 输出label的行数，type(label) = numpy.ndarray
        label_onehot = [[1 if j == label[i] else 0 for j in range(10)] for i in range(n)]
        # label_onehot应该是一个多维list,每一个label对应的常数值变为一个vector ---> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        return np.array(label_onehot)  # 将list转化为数组类型的返回值, np.random.randint(low, high,(array_size)) 生成整数随机数组

    def call_loss_cen(self, theta, Reliable_nodes):
        f_local_val = 0
        reg_n1 = 0
        if self.balanced == True:
            for id in Reliable_nodes:
                Y_label = self.one_hot(self.Y[id])
                X_image = self.X[id]
                t1 = np.dot(theta, X_image.T)  # sample * x.T
                t1 = t1 - np.max(t1, axis=0)  # 标准化，不改变结果
                t = np.exp(t1)
                tmp = t / np.sum(t, axis=0)
                reg_n2 = (self.reg_l2 / 2) * (LA.norm(theta) ** 2)
                f_local_val += -np.sum(Y_label.T * np.log(tmp)) / X_image.shape[0] + reg_n2
                reg_n1 += self.reg_l1 * LA.norm(theta, ord=1)
            loss = f_local_val + reg_n1
            return loss
        if self.balanced == False:
            for id in Reliable_nodes:
                Y_label = self.one_hot(self.Y[id])
                X_image = self.X[id]
                t1 = np.dot(theta, X_image.T)  # sample * x.T
                t1 = t1 - np.max(t1, axis=0)  # 标准化，不改变结果
                t = np.exp(t1)
                tmp = t / np.sum(t, axis=0)
                reg_n2 = (self.reg_l2 / 2) * (LA.norm(theta) ** 2)
                f_local_val += -np.sum(Y_label.T * np.log(tmp)) / X_image.shape[0] + reg_n2
                reg_n1 += self.reg_l1 * LA.norm(theta, ord=1)
            loss = f_local_val + reg_n1
            return loss

    def call_loss_dec(self, Model, Reliable_nodes, ave=None):
        f_local_val = 0
        reg_n1 = 0
        if ave == None: # for distance
            if self.balanced == True:
                for id in Reliable_nodes:
                    Y_label = self.one_hot(self.Y[id])
                    X_image = self.X[id]
                    t1 = np.dot(Model[id], X_image.T)  # sample * x.T
                    t1 = t1 - np.max(t1, axis=0)  # 标准化，不改变结果
                    t = np.exp(t1)
                    tmp = t / np.sum(t, axis=0)
                    reg_n2 = (self.reg_l2 / 2) * (LA.norm(Model[id]) ** 2)
                    f_local_val += -np.sum(Y_label.T * np.log(tmp)) / X_image.shape[0] + reg_n2
                    reg_n1 += self.reg_l1 * LA.norm(Model[id], ord=1)
            else:
                for id in Reliable_nodes:
                    Y_label = self.one_hot(self.Y[id])
                    X_image = self.X[id]
                    t1 = np.dot(Model[id], X_image.T)  # sample * x.T
                    t1 = t1 - np.max(t1, axis=0)  # 标准化，不改变结果
                    t = np.exp(t1)
                    tmp = t / np.sum(t, axis=0)
                    reg_n2 = (self.reg_l2 / 2) * (LA.norm(Model[id]) ** 2)
                    f_local_val += -np.sum(Y_label.T * np.log(tmp)) / X_image.shape[0] + reg_n2
                    reg_n1 += self.reg_l1 * LA.norm(Model[id], ord=1)
        else: # for averaged distance
            Model_ave = Model.mean(axis=0) # average of the model
            if self.balanced == True:
                for id in Reliable_nodes:
                    Y_label = self.one_hot(self.Y[id])
                    X_image = self.X[id]
                    t1 = np.dot(Model_ave, X_image.T)  # sample * x.T
                    t1 = t1 - np.max(t1, axis=0)  # 标准化，不改变结果
                    t = np.exp(t1)
                    tmp = t / np.sum(t, axis=0)
                    reg_n2 = (self.reg_l2 / 2) * (LA.norm(Model_ave) ** 2)
                    f_local_val += -np.sum(Y_label.T * np.log(tmp)) / X_image.shape[0] + reg_n2
                    reg_n1 += self.reg_l1 * LA.norm(Model_ave, ord=1)
            else:
                for id in Reliable_nodes:
                    Y_label = self.one_hot(self.Y[id])
                    X_image = self.X[id]
                    t1 = np.dot(Model_ave, X_image.T)  # sample * x.T
                    t1 = t1 - np.max(t1, axis=0)  # 标准化，不改变结果
                    t = np.exp(t1)
                    tmp = t / np.sum(t, axis=0)
                    reg_n2 = (self.reg_l2 / 2) * (LA.norm(Model_ave) ** 2)
                    f_local_val += -np.sum(Y_label.T * np.log(tmp)) / X_image.shape[0] + reg_n2
                    reg_n1 += self.reg_l1 * LA.norm(Model_ave, ord=1)
        loss = f_local_val + reg_n1
        return loss
#     '''
    # (mini)batch/stochastic gradient evaluation
    def localgrad(self, Para, id, j=None, BatchSize=None):  ## idx is the node index, j is local sample index
        Y_label = self.one_hot(self.Y[id])
        if j == None and BatchSize == None:       ## local full batch gradients
            X_image = self.X[id]
            t = np.dot(Para[id], X_image.T)  # para = 训练模型x， v = X.T
            t = t - np.max(t, axis=0)  # 标准化
            pro = np.exp(t) / np.sum(np.exp(t), axis=0)  # 前面的标准化不改变结果
            batch_gradient = - np.dot((Y_label.T - pro), X_image) / X_image.shape[0] + self.reg_l2 * Para[id]
            return batch_gradient
        elif j is None and BatchSize is not None:  ## minibatch stochastic gradients
            Config_seed = 1
            np.random.seed(Config_seed)
            select = np.random.randint(len(self.Y[id]))  # 从1到len(label)中随机选取一个正整数
            X_image = np.array(self.X[id][select: select + BatchSize])  # 选择连续index的样本
            Y_label = np.array(self.Y[id][select: select + BatchSize])  # 选择连续index的标签
            Y = self.one_hot(Y_label)
            t = np.dot(Para[id], X_image.T)  # \tilde x^T * v
            t = t - np.max(t, axis=0)  # 标准化，不改变结果
            pro = np.exp(t) / np.sum(np.exp(t), axis=0)
            minibatch_sto_grad = - np.dot((Y.T - pro), X_image) / BatchSize + self.reg_l2 * Para[id]
            return minibatch_sto_grad
        else:                                      ## local stochastic gradients
            X_image = self.X[id][j]
            t = np.dot(Para[id], X_image.T)  # \tilde x^T * v
            t = t - np.max(t, axis=0)  # 标准化，不改变结果
            pro = np.exp(t) / np.sum(np.exp(t), axis=0)
            temp = Y_label[j].T - pro
            sto_grad = - np.dot(temp[:, np.newaxis], X_image[np.newaxis, :]) + self.reg_l2 * Para[id]  # 求局部随机梯度
            return sto_grad

    def networkgrad(self, Para, idxv=None):  ## network batch/stochastic gradient
        grad = np.zeros((self.m, self.n0, self.n1))
        if idxv is None:  ## full batch
            for id in range(self.m):
                grad[id] = self.localgrad(Para, id)
            return grad
        else:  ## stochastic gradient: one sample
            for id in range(self.m):
                grad[id] = self.localgrad(Para, id, idxv[id])
            return grad

    def grad(self, Para, Reliable_nodes, idx=None):  ## centralized batch/stochastic gradient
        """
        Para: (10, 784)
        X_image: (6000, 784)
        t: (10, 6000)
        """
        if idx == None:  ## full batch
            batch_gradient = np.zeros( (self.n0, self.n1) )
            if self.balanced == True:
                for id in Reliable_nodes:
                    X_image = self.X[id]  # (60000, 784)
                    Y_label = self.one_hot(self.Y[id])  # (60000, 10)
                    t = np.dot(Para, X_image.T)  # 矩阵相乘   para = 训练模型x， v = X.T
                    t = t - np.max(t, axis=0)  # 标准化
                    pro = np.exp(t) / np.sum(np.exp(t), axis=0)  # 前面的标准化不改变结果
                    batch_gradient += - np.dot((Y_label.T - pro), X_image) / X_image.shape[0] + self.reg_l2 * Para  # 求局部全梯度
                return batch_gradient
            else:
                return np.sum(self.networkgrad(np.tile(Para, (len(Reliable_nodes), 1))), axis=0)
        else:
            if self.balanced == True:
                X_image = self.X[idx]
                Y_label = self.one_hot(self.Y[idx])
                local_component_grad = np.zeros((self.n0, self.n1))
                for j in range(self.data_distr[idx]):
                    t = np.dot(Para, X_image[j].T)  # para = 训练模型x， v = X.T
                    t = t - np.max(t, axis=0)  # 标准化
                    pro = np.exp(t) / np.sum(np.exp(t), axis=0)  # 前面的标准化不改变结果
                    local_component_grad += - np.dot((Y_label[j].T - pro), X_image[j]) / X_image.shape[0] + self.reg_l2 * Para  # centralized stochastic
                return local_component_grad
            else:
                sys.exit('data distribution is not balanced !!!')

    # proximal mapping
    def prox_plus(self, X):
        """Operator: max(X, 0)"""
        below = X < 0
        X[below] = 0
        return X

    # def prox_l1(self, X, lamb, Cen = None):
    #     if Cen is None:
    #         for i in range(X.shape[0]):
    #             if np.isscalar(lamb):
    #                 """ Proximal operator for the l1 regularization """
    #                 X[i] = self.prox_plus(np.abs(X[i]) - lamb) * np.sign(X[i])  # * 表示矩阵中对应元素相乘
    #             else:
    #                 X[i] = self.prox_plus(np.abs(X[i]) - lamb[i][i]) * np.sign(X[i])  # * 表示矩阵中对应元素相乘
    #     else:
    #         """ Proximal operator for the l1 regularization """
    #         X = self.prox_plus(np.abs(X) - lamb) * np.sign(X)  # * 表示矩阵中对应元素相乘
    #     return X
    def prox_l1( self, X, lamb, Cen = None ):
        reliable_num = X.shape[0]
        if Cen is None:
            for id in range( reliable_num ):
                if np.isscalar(lamb):
                    """ Proximal operator for the l1 regularization """
                    X[id] = self.prox_plus(np.abs(X[id]) - lamb ) * np.sign(X[id])  # * 表示矩阵中对应元素相乘
                else:
                    X[id] = self.prox_plus(np.abs(X[id]) - lamb[id][id] ) * np.sign(X[id])  # * 表示矩阵中对应元素相乘
        else:
            """ Proximal operator for the l1 regularization """
            X = self.prox_plus( np.abs(X) - lamb ) * np.sign(X)  # * 表示矩阵中对应元素相乘
        return X

    def fast_mix(self, multi_cons, W, eta, var):  # fast_mix for the Byzantine-free case
        assert multi_cons >= 1
        x_0 = copy.deepcopy(var)
        x_1 = copy.deepcopy(var)
        for _ in range(multi_cons):
            x_2 = (1 + eta) * W.dot(x_1) - eta * x_0
            x_0 = x_1
            x_1 = x_2
        var = x_2
        return var

    def fast_mix_Byzan(self, id, Para, multi_cons, W, eta):  # fast_mix for the Byzantine case
        # assert multi_cons >= 1  # 如果expression是True，那么什么反应都没有。但是如果expression是False，那么会报错AssertionError
        # x_0 = copy.deepcopy(Para[id])
        # x_1 = copy.deepcopy(Para[id])
        # for _ in range(multi_cons):
        #     temp = np.zeros((Para[id].shape[0], Para[id].shape[1]))
        #     for jd in neighbors_id:
        #         temp += W[id, jd] * Para[jd]
        #     x_2 = (1 + eta) * temp - eta * x_0
        #     x_0 = x_1
        #     x_1 = x_2
        # var = x_2
        # return var
        assert multi_cons >= 1
        x_0 = copy.deepcopy(Para).reshape(self.m, -1)
        x_1 = copy.deepcopy(Para).reshape(self.m, -1)
        for _ in range(multi_cons):
            x_2 = (1 + eta) * W.dot(x_1) - eta * x_0
            x_0 = x_1
            x_1 = x_2
        # print(x_2[id][np.random.randint(0, 784*10)])
        return x_2[id].reshape(self.n0, self.n1)

    def cal_max_norm_grad(self, Para):
        if np.all(Para == 0):  # np.all()函数用于判断整个数组中的元素的值是否全部满足条件，如果满足条件返回True，否则返回False
            return Para
        tmp = np.abs(Para)
        re = np.where(tmp == np.max(tmp))  # np.max返回矩阵或向量最大的元素；
        # np.where:当 where()内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的坐标为元组（行，列）
        row = re[0][0]  # 定位元组中第一个元素的行
        col = re[1][0]  # 定位元组中第一个元素的列
        max_val = tmp[row, col]  # Para中元素绝对值的最大值
        # tmp_Para = np.zeros_like(Para)
        n = len(re[0])
        Para[tmp != np.max(tmp)] = 0  # 除绝对值为最大元素保留，其它均置0
        Para[Para == -max_val] = -1.0 / n
        Para[Para == max_val] = 1.0 / n
        return Para