########################################################################################################################
####---------------------------------------------------Analysis-----------------------------------------------------####
########################################################################################################################

## Used to calculate different types of errors for all algorithms

import numpy as np


class error:
    def __init__(self, problem, model_optimal, loss_optimal):
        self.pr = problem  ## problem class
        self.N = self.pr.N  ## total number of data samples
        self.X = self.pr.X_train  ## feature vectors
        self.Y = self.pr.Y_train  ## label vector
        self.theta_opt = model_optimal
        self.F_opt = loss_optimal

    def path_cls_error(self, iterates):
        iterates = np.array(iterates)
        Y_predict = np.matmul(self.X, iterates.T)
        error_matrix = np.multiply(Y_predict, self.Y[:, np.newaxis]) < 0
        return np.sum(error_matrix, axis=0) / self.N

    def point_cls_error(self, theta):
        Y_predict = np.matmul(self.X, theta)
        error = Y_predict * self.Y < 0
        return sum(error) / self.N

    # def theta_gap_path(self, Para, regular):
    #     K = len(Para)
    #     result = []
    #     for k in range(K):
    #         for id in regular:
    #             result.append((Para[k][id] - self.theta_opt)/len(regular))
    #     return result

    def theta_gap_point(self, para, regular):
        Res = 0
        for id in regular:
            Res += np.linalg.norm(para[id] - self.theta_opt, ord=2) ** 2
            # Res += np.linalg.norm(np.sum(para, axis=0) / len(regular) - self.theta_opt, ord=2) ** 2
        return Res

    def theta_gap_path(self, iterates, regular):
        K = len(iterates)
        Residual = []
        for k in range(K):
            Residual.append(error.theta_gap_point(self, iterates[k], regular))
        return Residual

    def loss_gap_path(self, loss, regular):
        K = len(loss)
        result = []
        for k in range(K):
            result.append((loss[k] - self.F_opt)/len(regular))
        return result

    def loss_gap_point_c(self, theta, regular):
        return (self.pr.call_loss_cen(theta, regular) - self.F_opt)/len(regular)

    def loss_gap_path_c(self, iterates, regular):
        K = len(iterates)
        result = []
        for k in range(K):
            result.append(error.loss_gap_point_c(self, iterates[k], regular))
        return result


def predict(w, test_image):
    """
    Predict the label of the test_image

    :param w: model parameter, shape(10, 784), the so-called optimal point
    :param test_image: shape(784)
    :param test_label: scalar
    """
    mat = np.dot(w, test_image.T)
    predict_label = np.argmax(mat)  # 返回的是a中元素最大值所对应的索引值，根据概率判断标签
    # print("label :",test_label , "predict_label:",predict_label)
    # test_label = test_label
    return predict_label

def get_residual(Reliable_Set, para, optimal_solution):
    Res = 0
    for id in Reliable_Set:
        Res += np.linalg.norm(para[id] - optimal_solution, 2) ** 2  # the deviation of consensus  np.linalg.norm()默认为向量二范数
    return Res

def get_accuracy(para_ave, image, label):
    """
    Compute the accuracy of the method

    :param w: model parameter, shape(10, 784)
    :param image: image, shape(784)
    :param label: label, scalar
    """
    number_sample = len(label)  # 总标签数 = 总样本数
    right = 0
    for i in range(number_sample):
        predict_label = predict(para_ave, image[i])  # 输出为单个标签
        if predict_label == label[i]:
            right += 1
    accuracy = right / number_sample
    # print("the accuracy of training set is :", accuracy)
    return accuracy


def get_consensus_error(Reliable_Set, para, para_ave):
    var = 0
    for id in Reliable_Set:
        var += np.linalg.norm(para[id] - para_ave) ** 2  # the deviation of consensus  np.linalg.norm()默认为向量二范数
    return var / len(Reliable_Set)


"""
def get_consensus_error(Reliable_Set, para):
    '''
    Compute the variation of regualr model parameters

    :param regular: the set of regular workers
    :param W: the set of regular model parameters
    '''
    para_regular = []
    for id in Reliable_Set:
        para_regular.append(para[id])
    para_regular = np.array(para_regular)

    mean = np.mean(para_regular, axis=0)
    var = 0
    num = para_regular.shape[0]
    for id in range(num):
        var += np.linalg.norm(para_regular[id] - mean) ** 2  # the deviation of consensus  np.linalg.norm()默认为向量二范数

    return var / num
"""


