########################################################################################################################
####-------------------------------------Logistic Regression for MNIST dataset--------------------------------------####
########################################################################################################################

## Used to help implement decentralized algorithms to classify MNIST dataset

import numpy as np
from numpy import linalg as LA
import copy
import os
import sys

class LR_L2( object ):
    def __init__(self, n_agent, class1 = 1, class2 = 7, train = 10000, balanced = True, limited_labels = False, setting = None ):
        self.class1 = class1
        self.class2 = class2
        self.train = train
        self.limited_labels = limited_labels
        self.n = n_agent 
        self.balanced = balanced
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_data()
        self.N = len(self.X_train)            ## total number of data samples
        self.b = int(self.N/self.n)           ## average local samples
        if balanced == False:
            self.split_vec = np.sort(np.random.choice(np.arange(1,self.N),self.n-1, replace = False )) 
        self.X, self.Y, self.data_distr = self.distribute_data()
        self.p = len(self.X_train[0])         ## dimension of the feature
        self.dim = self.p                     ## dimension of the feature
        self.reg_l1 = 1/self.N
        self.reg_l2 = 1/self.N
        self.L, self.kappa = self.smooth_scvx_parameters()
        self.setting = setting

    def load_data(self):
        if os.path.exists('mnist.npz'):
            print( 'data exists' )
            data = np.load('mnist.npz', allow_pickle=True)
            X = data['X']
            y = data['y']
        else:
            print( 'downloading data' )
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            np.savez_compressed('mnist_2c', X=X, y=y)
        y = y.astype(int)
        print( 'data initialized' )

        ## append 1 to the end of all data points
        X = np.append(X, np.ones((X.shape[0], 1)), axis = 1)
        
        ## data normalization: each data is normalized as a unit vector 
        X = X / LA.norm(X,axis = 1)[:,None]
        
        ## select corresponding classes
        X_C1_C2 = X[ (y == self.class1) | (y == self.class2) ]
        y_C1_C2 = y[ (y == self.class1) | (y == self.class2) ]
        y_C1_C2[ y_C1_C2 == self.class1 ] = 1    
        y_C1_C2[ y_C1_C2 == self.class2 ] = -1
        X_train, X_test = X_C1_C2[ : self.train], X_C1_C2[ self.train : ]
        # print(X_test[2].shape, len(X_test))
        # exit(0) ## 程序无错误退出
        Y_train, Y_test = y_C1_C2[ : self.train], y_C1_C2[ self.train : ]

             
        if self.limited_labels == True:
            permutation = np.argsort(Y_train)
            X_train = X_train[permutation]
            Y_train = np.sort(Y_train)
            
        return X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy() 
    
    def distribute_data(self):
        if self.balanced == True:
           X = np.array( np.split( self.X_train, self.n, axis = 0 ) ) 
           Y = np.array( np.split( self.Y_train, self.n, axis = 0 ) ) 
        if self.balanced == False:   ## random distribution
           X = np.array( np.split(self.X_train, self.split_vec, axis = 0) )
           Y = np.array( np.split(self.Y_train, self.split_vec, axis = 0 ) )
        data_distribution = np.array([ len(_) for _ in X ])
        return X, Y, data_distribution
    
    def smooth_scvx_parameters(self):
        Q = np.matmul(self.X_train.T,self.X_train)/self.N
        L_F = max(abs(LA.eigvals(Q)))/4
        L = L_F + self.reg_l2
        kappa = L/self.reg_l2
        return L, kappa

    # def F_val(self, theta):           ##  objective function value at theta
    #     if self.balanced == True:
    #         f_val = np.sum( np.log( np.exp( np.multiply(-self.Y_train,\
    #                                                 np.matmul(self.X_train,theta)) ) + 1 ) )/self.N
    #         reg_val = (self.reg_l2/2) * (LA.norm(theta) ** 2)
    #         return f_val + reg_val
    #     if self.balanced == False:
    #         temp1 = np.log( np.exp( np.multiply(-self.Y_train,\
    #                           np.matmul(self.X_train,theta)) ) + 1 )
    #         temp2 = np.split(temp1, self.split_vec)
    #         f_val = 0
    #         for i in range(self.n):
    #             f_val += np.sum(temp2[i])/self.data_distr[i]
    #         reg_val = (self.reg_l2/2) * (LA.norm(theta) ** 2)
    #         return f_val/self.n + reg_val

    def call_loss_cen(self, theta, Reliable_nodes):  #  objective function value at theta for centralized optimizer on Byzantine-free case
        f_local_val = 0
        if self.balanced == True:
            for id in Reliable_nodes:
                for j in range(self.b):
                    f_local_val += np.log(np.exp(-self.Y[id][j] * np.inner(self.X[id][j], theta)) + 1)/self.b
            reg_val = (self.reg_l2/2) * (LA.norm(theta) ** 2) + self.reg_l1 * LA.norm(theta, ord=1)
            return f_local_val / len(Reliable_nodes) + reg_val
        if self.balanced == False:
            for id in Reliable_nodes:
                for j in range(self.b):
                    f_local_val += np.log( np.exp(-self.Y[id][j] * np.inner(self.X[id][j], theta)) + 1 )/self.b
            reg_val = (self.reg_l2/2) * (LA.norm(theta) ** 2) + self.reg_l1 * LA.norm(theta, ord=1)
            return f_local_val/len(Reliable_nodes) + reg_val

    def call_loss_dec(self, theta, Reliable_nodes):  #  objective function value at theta (average) for decentralized optimizers on Byzantine case
        f_sum = 0
        reg_val = 0
        if self.balanced == True:
            for id in Reliable_nodes:
                for j in range(self.b):
                    # temp = (-self.Y[id][j] * np.inner(self.X[id][j], theta))
                    # if temp >= 0:
                    #     f_local_temp = np.log( ( np.exp ( -temp ) + 1 )/( np.exp ( -temp ) ) )/self.b
                    # else:
                    #     f_local_temp = np.log( (np.exp(temp) + 1) ) / self.b
                    # f_local_val += f_local_temp
                    f_sum += np.log(np.exp(-self.Y[id][j] * np.inner(self.X[id][j], theta)) + 1)/self.b
                reg_val += (self.reg_l2 / 2) * (LA.norm(theta) ** 2) + self.reg_l1 * LA.norm(theta, ord=1)
            return f_sum/len(Reliable_nodes) + reg_val/len(Reliable_nodes)
        if self.balanced == False:
            for id in Reliable_nodes:
                for j in range(self.data_distr[id]):
                    f_sum += np.log(np.exp(-self.Y[id][j] * np.inner(self.X[id][j], theta)) + 1 )/self.data_distr[id]
                reg_val += (self.reg_l2 / 2) * (LA.norm(theta) ** 2) + self.reg_l1 * LA.norm(theta, ord=1)
            return f_sum / len(Reliable_nodes) + reg_val / len(Reliable_nodes)


    def localgrad(self, theta, idx, j = None ):  ## idx is the node index, j is local sample index
        if j == None:                 ## local full batch gradient
            # temp1 = np.exp(np.matmul(self.X[idx], theta[idx]) * (-self.Y[idx]))  # 一维数组
            # temp2 = (temp1 / (temp1 + 1)) * (-self.Y[idx])
            # grad = self.X[idx] * temp2[:, np.newaxis]
            # return np.sum(grad, axis=0) / self.data_distr[idx] + self.reg_l2 * theta[idx]
            grad_comp_sum = np.zeros(self.X[idx].shape[1], )
            for j in range(self.Y[idx].shape[0]):
                temp = np.inner( self.X[idx][j], theta[idx] ) * (-self.Y[idx][j])
                if temp >= 0:
                    grad_temp = -self.Y[idx][j] * (1 / (1 + np.exp(-temp))) * self.X[idx][j]
                else:
                    grad_temp = -self.Y[idx][j] * ( np.exp(temp)/(np.exp(temp) + 1) ) * self.X[idx][j]
                grad_comp_sum += grad_temp
            return grad_comp_sum/self.data_distr[idx] + self.reg_l2 * theta[idx]
        else:  ## local stochastic gradient
            # temp = np.exp(-self.Y[idx][j] * np.inner(self.X[idx][j], theta[idx]) )
            # grad_lr = -self.Y[idx][j] * (temp/(1+temp)) * self.X[idx][j]
            temp = -self.Y[idx][j] * np.inner(self.X[idx][j], theta[idx])
            if temp >= 0:
                grad_lr = -self.Y[idx][j] * (1/(1 + np.exp(-temp))) * self.X[idx][j]
            else:
                grad_lr = -self.Y[idx][j] * (np.exp(temp)/(1 + np.exp(temp))) * self.X[idx][j]
            grad = grad_lr + self.reg_l2 * theta[idx]
            return grad
        
    def networkgrad(self, theta, idxv = None):  ## network batch/stochastic gradient
        grad = np.zeros( (self.n, self.p) )
        if idxv is None:                        ## full batch
            for i in range(self.n):
                grad[i] = self.localgrad(theta, i)
            return grad
        else:                                   ## stochastic gradient: one sample
            for i in range(self.n):
                grad[i] = self.localgrad(theta, i, idxv[i])
            return grad
    
    def grad(self, theta, Reliable_nodes, idx = None):  ## centralized stochastic/batch gradient
        grad_lr = np.zeros(self.p, )
        if idx == None:    ## full batch
            if self.balanced == True:
                for id in Reliable_nodes:
                    temp1 = np.exp(np.matmul(self.X[id], theta) * (-self.Y[id]))  # 一维数组
                    temp2 = ( temp1 / (temp1 + 1) ) * ( -self.Y[id] )
                    grad_lr += np.sum(self.X[id] * temp2[:, np.newaxis], axis=0)/self.data_distr[id]
                return grad_lr/len(Reliable_nodes) + self.reg_l2 * theta
            else:
                return np.sum(self.networkgrad(np.tile(theta, ( len(Reliable_nodes), 1 )))\
                              , axis = 0)/len(Reliable_nodes)
        else:
            if self.balanced == True:
                for j in range(self.data_distr[idx]):
                    temp = np.exp( self.Y[idx][j]*np.inner(self.X[idx][j], theta) )
                    grad_lr += -self.Y[idx][j]/(1+temp) * self.X[idx][j]//self.b
                grad = grad_lr/len(Reliable_nodes) + self.reg_l2 * theta
                return grad
            else:
                sys.exit( 'data distribution is not balanced !!!' )

    # proximal mapping
    def prox_plus(self, X):
        """Operator: max(X, 0)"""
        below = X < 0
        X[below] = 0
        return X

    def prox_l1(self, X, lamb):
        if np.isscalar( lamb ):
            """ Proximal operator for the l1 regularization """
            X = self.prox_plus(np.abs(X) - lamb) * np.sign(X)  # * 表示矩阵中对应元素相乘
        else:
            X = self.prox_plus( np.abs(X) - np.matmul(lamb, np.ones(( len(lamb[1]), len(X[1]) )) ) ) * np.sign(X)   # * 表示矩阵中对应元素相乘
        return X

    def fast_mix(self, multi_cons, W, eta, var): # fast_mix for the Byzantine-free case
        assert multi_cons >= 1
        x_0 = copy.deepcopy( var )
        x_1 = copy.deepcopy( var )
        for _ in range( multi_cons ):
            x_2 = (1 + eta) * W.dot(x_1) - eta * x_0
            x_0 = x_1
            x_1 = x_2
        var = x_2
        return var

    def fast_mix_Byzan( self, Para, id, multi_cons, W, eta ): # fast_mix for the Byzantine case
        assert multi_cons >= 1
        x_0 = copy.deepcopy( Para[id] )
        x_1 = copy.deepcopy( Para[id] )
        for _ in range( multi_cons ):
            temp = np.zeros( Para.shape[1], )
            for jd in range(self.n):
                temp += W[id, jd] * Para[jd]
            x_2 = (1 + eta) * temp - eta * x_0
            x_0 = x_1
            x_1 = x_2
        var = x_2
        return var

