import numpy as np
import copy as cp
import random

"""
Different Byzantine_nodes attacks, include:
same-value attacks, sign-flipping attacks,
sample-duplicating attacks (only conducted in non-iid case)

@:param Para : the set of workers' model parameters
"""


def Gaussian_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable, eta=None, W=None,
                     scale=30):
    # with the same mean and larger variance
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-GA-'
    elif idn in Reliable_nodes:
        Para_temp = np.zeros((Para.shape[1], Para.shape[2]))
        if W is None:
            count = len(neighbors_Reliable)
            for jd in neighbors_Reliable:
                # count += 1
                Para_temp += Para[jd]
            # if count != 0:
            x_bar = Para_temp / count  # There are no isolated reliable nodes, which means count != 0
            for id in neighbors_Byzantine:
                Para[id] = cp.deepcopy(x_bar)
                noise = np.random.randn( Para[idn].shape[0], Para[idn].shape[1] )   # noise: 标准正态分布噪声 x∼N(0, 1)
                Para[id] += scale * noise  # mean = mu, variance = scale**2
        else:
            tmp = 0
            for jd in Reliable_nodes:
                Para_temp += W[idn, jd] * Para[jd]
                tmp += W[idn, jd]  # tmp != 0
            x_bar = Para_temp / tmp
            for id in neighbors_Byzantine:
                Para[id] = cp.deepcopy(x_bar)
                noise = np.random.randn( Para[idn].shape[0], Para[idn].shape[1] )  # 生成标准正态分布噪声 x∼N(0, 1)
                Para[id] += scale * noise  # mean = mu, variance = scale**2
        return Para, '-GA-'


""" Byzantine agents set their status as the flipping values of the average of status with respect to reliable agents"""
def Sign_flipping_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable, eta = None, W=None):
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-SFA-'
    elif idn in Reliable_nodes:
        Para_temp = np.zeros((Para.shape[1], Para.shape[2]))
        if W is None:
            count = len(neighbors_Reliable)
            for jd in neighbors_Reliable:
                Para_temp += Para[jd]
            x_bar = Para_temp / count
            for id in neighbors_Byzantine:
                # np.random.seed(id) # set a seed
                # s = random.choice(np.arange(1, len(neighbors_Byzantine), 2))
                Para[id] = -10 * cp.deepcopy(x_bar)
        else:
            tmp = 0
            for jd in Reliable_nodes:
                Para_temp += W[idn, jd] * Para[jd]
                tmp += W[idn, jd]
            x_bar = Para_temp / tmp
            for id in neighbors_Byzantine:
                Para[id] = -1 * cp.deepcopy(x_bar)
        return Para, '-SFA-'


""" 
# Byzantine agents set their status as the flipping values of the status of their own
def Sign_flipping_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable, W = None):
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-SFA-'
    else:
        for id in neighbors_Byzantine:
            Para[id] *= -1
        return Para, '-SFA-'
"""


def Sample_duplicating_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable,
                               *args):
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-SDA-'
    else:
        duplicate_index = random.choice(neighbors_Reliable)
        for id in neighbors_Byzantine:
            Para[id] = Para[duplicate_index]  # sample-duplicating attacks
        return Para, '-SDA-'


def Same_value_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable,\
                       eta=None, W=None):
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-SVA-'
    else:
        for id in neighbors_Byzantine:
            # np.random.seed(10)
            # Para[id] = np.random.randint( 1000, 10000, size = (Para.shape[1], Para.shape[2]) )
            Para[id] = 1000*np.ones( (Para.shape[1], Para.shape[2]) )
    return Para, '-SVA-'


def Isolation_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable, eta = None, W=None):
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-IA-'
    elif idn in Reliable_nodes:
        count_B = len(neighbors_Byzantine)
        Para_temp = np.zeros((Para.shape[1], Para.shape[2]))
        if W is None:
            count_R = len(neighbors_Reliable)
            for jd in neighbors_Reliable:
                Para_temp += Para[jd]
            x_bar = Para_temp / count_R
            tmp = len(neighbors_Reliable) * x_bar
            for id in neighbors_Byzantine:
                Para[id] = (Para[idn] - tmp) / count_B
            return Para, '-IA-'
        else:
            tmp = 0
            for jd in neighbors_Reliable:
                Para_temp += W[idn, jd] * Para[jd]
                tmp += W[idn, jd]
            x_bar = Para_temp / tmp
            temp_weight = 0
            for idm in neighbors_Byzantine:
                temp_weight += W[idn, idm]
            x_temp = np.zeros(Para.shape[1], )
            for jd in neighbors_Reliable:
                x_temp += W[idn, jd] * x_bar
            for idm in neighbors_Byzantine:
                Para[idm] = (Para[idn] - x_temp) / temp_weight
            return Para, '-IA-'


def Zero_sum_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable, eta=None, W=None):
    # At each iteration, the sum of information (including the self-loop information) received by reliable agent idn is zero
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-ZSA-'
    elif idn in Reliable_nodes:
        count_B = len(neighbors_Byzantine)
        Para_temp = np.zeros((Para.shape[1], Para.shape[2]))
        if W is None:
            for jd in neighbors_Reliable:
                Para_temp += Para[jd]
            for id in neighbors_Byzantine:
                Para[id] = -1 * Para_temp / count_B
            return Para, '-ZSA-'
        elif eta is None:
            for jd in neighbors_Reliable:
                Para_temp += W[idn, jd] * Para[jd]
            for id in neighbors_Byzantine:
                Para[id] = -1 * Para_temp / count_B / W[idn, id]
            return Para, '-ZSA-'
        else:
            for jd in neighbors_Reliable:
                Para_temp += W[idn, jd] * Para[jd]
            for id in neighbors_Byzantine:
                # Para[id] = -1 * Para_temp / count_B / W[idn, id]
                temp = (1+eta)*Para_temp - eta * Para[idn]
                Para[id] = -1 * ( ( ( temp + eta * Para[idn] ) / (1+eta) ) / count_B / W[idn, id] )
            return Para, '-ZSA-'


def Hybrid_attacks(idn, Para, Byzantine_nodes, Reliable_nodes, neighbors_Byzantine, neighbors_Reliable, W=None,
                   scale=10):
    # At each iteration, the sum of information (including the self-loop information) received by reliable agent idn is zero
    if len(neighbors_Byzantine) == 0 or idn in Byzantine_nodes:  # 如果没有Byzantine节点或者当前节点为Byzantine节点，直接跳出函数
        return Para, '-HA-'
    elif idn in Reliable_nodes:
        count_B = len(neighbors_Byzantine)
        Para_temp = np.zeros((Para.shape[1], Para.shape[2]))
        if W is None:
            for jd in neighbors_Reliable:
                Para_temp += Para[jd]
            for id in neighbors_Byzantine:
                Para[id] = -1 * Para_temp / count_B
                noise = np.random.randn(Para[idn].shape[0])  # noise: 标准正态分布噪声 x∼N(0, 1)
                Para[id] += scale * noise
            return Para, '-HA-'
        else:
            for jd in neighbors_Reliable:
                Para_temp += W[idn, jd] * Para[jd]
            for id in neighbors_Byzantine:
                Para[id] = -1 * Para_temp / count_B / W[idn, id]
                noise = np.random.randn(Para[idn].shape[0])  # noise: 标准正态分布噪声 x∼N(0, 1)
                Para[id] += scale * noise
            return Para, '-HA-'