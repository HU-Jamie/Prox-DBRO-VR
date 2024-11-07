########################################################################################################################
####-----------------------------------------------Geometric Network------------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Geometric directed graphs using logistic regression.
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import copy as cp
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Byzan_graph
from analysis import error
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from Optimizers import DOPTIMIZER_Gradient_Tracking as dgt
import Attacks
from Problems.logistic_regression import LR_L2
from Problems.soft_max_regression import Softmax
import seaborn as sns
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import datetime

time1 = datetime.datetime.now()
print("Current Wall Time:", time1)
sns.set(context='notebook',
        style='darkgrid',
        palette='deep',
        font='sans-serif',
        font_scale=1,
        color_codes=True,
        rc=None)
####----------------------------------------------MNIST Classification----------------------------------------------####

"""
Data processing for MNIST
"""
m = 60  # the total number of nodes
ByzantineSize = 20  # the total number of Byzantine nodes
ReliableSize = int(m - ByzantineSize)
SM = Softmax(m, limited_labels=False, balanced=True, setting='MNIST')  ## instantiate the problem class
n0 = SM.n0  ## dimension of the model
n1 = SM.n1
L = SM.L  ## L-smooth constant
N = SM.N  ## total number of training samples
print("The number of training samples =", N)
b = SM.b  ## average number of local samples
print("the shape of training set:", SM.X_train.shape)
image_test = SM.X_test
label_test = SM.Y_test
print("the shape of labels:", label_test.shape)
print("the number of testing samples =", len(label_test))
print("L1 regularized constant: {}, L2 regularized constant: {}".format(SM.reg_l1, SM.reg_l2))
"""
Decentralized Attacks
"""
Attack = Attacks.Sign_flipping_attacks
print('Attack:', Attack)
if Attack == Attacks.Zero_sum_attacks:
    attack_type = '_ZSA'
elif Attack == Attacks.Gaussian_attacks:
    attack_type = '_GA'
elif Attack == Attacks.Same_value_attacks:
    attack_type = '_SVA'
elif Attack == Attacks.Sign_flipping_attacks:
    attack_type = '_SFA'

epochs = 150
print("epochs =", epochs)
optConfig = {
    'Iterations': epochs,
    'NodeSize': m,
    'ByzantineSize': ByzantineSize,
    'ReliableSize': ReliableSize,
    'StepSize': 0
}

# randomly generate Byzantine workers
seed = 1
random.seed(seed)
print('Selection seed =', seed)
Byzantine = random.sample(range(optConfig['NodeSize']), optConfig['ByzantineSize'])
print('Byzantine set:', Byzantine)
# 随机选取攻击节点， optConfig['byzantineSize']=攻击节点个数；
# random.sample=截取列表的指定长度的随机数，但是不会改变列表本身的排序；返回值为一个list
Reliable = list(set(range(optConfig['NodeSize'])).difference(Byzantine))  # 剩下的为正常节点的集合
print('Reliable set:', Reliable)
optConfig['ByzantineSet'] = Byzantine
optConfig['ReliableSet'] = Reliable
# generate the adjacent matrix
Adj, G, H = Byzan_graph(m).undirected(Byzantine)
# generate the weight matrix
M = Weight_matrix(Adj).metroplis()
optConfig['ByzantineNetwork'] = G
optConfig['ReliableNetwork'] = H
optConfig['AdjacentMatrix'] = Adj
optConfig['WeightMatrix'] = M

"""
optConfig = {
    'Iterations': epochs 
    'NodeSize': int(n),
    'ByzantineSize': ByzantineSize,
    'ByzantineSize': int(n - ByzantineSize),
    'StepSize': 0,
    'ByzantineSet': Byzantine,
    'ReliableSet': Reliable,
    'ByzantineNetwork': G,
    'ReliableNetwork': H,
    'AdjacentMatrix': Adj,
    'WeightMatrix': M,
    'Initialization': np.random.random( (n,p) )
    'Triggered Probability':  int( SM.m/SM.N/2 )
}
"""

Config_seed = 1  # 1 这个地方会对算法运行结果产生很大的影响
np.random.seed(Config_seed)

"""
Constant step-size setup and initialization for CPGD
"""
CPGD_Config = optConfig.copy()
CPGD_Config['StepSize'] = 1 / L * 0.3  # 0.5, 0.5, 0.3
print('Step-size of CPGD =', "1/L * 0.3")

CPGD_Config['Initialization'] = np.random.random((n0, n1))
CPGD_Config['Iterations'] = int( epochs * 10 )  # 5， 10， 10

"""
Constant step-size setup and initialization for decentralized algorithms
"""
PG_EXTRA_Config = optConfig.copy()
PG_EXTRA_Config['StepSize'] = 0.5
PG_EXTRA_Config['Initialization'] = np.random.random((m, n0, n1))

step_NIDS = 0.3
print('Step of NIDS =', step_NIDS)
NIDS_Config = optConfig.copy()
NIDS_Config['StepSize'] = step_NIDS * np.eye(m) + 0.05 * np.diag(np.random.random(m))
NIDS_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])

step_NIDS_SAGA = 0.1
print('Step of NIDS-SAGA =', step_NIDS_SAGA)
NIDS_SAGA_Config = optConfig.copy()
NIDS_SAGA_Config['StepSize'] = step_NIDS_SAGA * np.eye(m) + np.diag(np.random.random(m))
NIDS_SAGA_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
NIDS_SAGA_Config['Iterations'] = int(epochs * SM.b)

step_NIDS_LSVRG = 2
print('Step of NIDS-LSVRG =', step_NIDS_LSVRG)
NIDS_LSVRG_Config = optConfig.copy()
NIDS_LSVRG_Config['StepSize'] = step_NIDS_LSVRG * np.eye(m) + np.diag(np.random.random(m))
NIDS_LSVRG_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
NIDS_LSVRG_Config['Triggered Probability'] = int(SM.m / SM.N / 2)

PMGT_LSVRG_Config = optConfig.copy()
PMGT_LSVRG_Config['StepSize'] = 0.5
print('Step-size of PMGT-LSVRG =', PMGT_LSVRG_Config['StepSize'])
PMGT_LSVRG_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
PMGT_LSVRG_Config['Triggered Probability'] = SM.m / SM.N
a, d, c = np.linalg.svd(M)
eta = (1 - np.sqrt(1 - d[1] ** 2)) / (1 + np.sqrt(1 - d[1] ** 2))
PMGT_LSVRG_Config['Mixing parameter'] = eta
# print("Metroplis matrix: {}, column sum: {}, row sum: {}".format(M, np.sum(M, axis = 0), np.sum(M, axis = 1)))
PMGT_LSVRG_Config['Multi communications'] = 1
print('Multi communications of PMGT-LSVRG =', PMGT_LSVRG_Config['Multi communications'])

PMGT_SAGA_Config = optConfig.copy()
PMGT_SAGA_Config['StepSize'] = 0.3
print('Step-size of PMGT-SAGA =', PMGT_SAGA_Config['StepSize'])
PMGT_SAGA_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
PMGT_SAGA_Config['Iterations'] = int(epochs * SM.b)
PMGT_SAGA_Config['Mixing parameter'] = eta
# print("Metroplis matrix: {}, column sum: {}, row sum: {}".format(M, np.sum(M, axis = 0), np.sum(M, axis = 1)))
PMGT_SAGA_Config['Multi communications'] = 2
print('Multi communications of PMGT-SAGA =', PMGT_SAGA_Config['Multi communications'])

Prox_BRIDGE_T_Config = optConfig.copy()
Prox_BRIDGE_T_Config['StepSize'] = 0.35
print('Step-size of Prox-BRIDGE-T =', Prox_BRIDGE_T_Config['StepSize'])
Prox_BRIDGE_T_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])

Prox_BRIDGE_M_Config = optConfig.copy()
Prox_BRIDGE_M_Config['StepSize'] = 0.3
print('Step-size of Prox-BRIDGE-M =', Prox_BRIDGE_M_Config['StepSize'])
Prox_BRIDGE_M_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])

Prox_BRIDGE_K_Config = optConfig.copy()
Prox_BRIDGE_K_Config['StepSize'] = 0.4
print('Step-size of Prox-BRIDGE-K =', Prox_BRIDGE_K_Config['StepSize'])
Prox_BRIDGE_K_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])

Prox_BRIDGE_B_Config = optConfig.copy()
Prox_BRIDGE_B_Config['StepSize'] = 0.4
print('Step-size of Prox-BRIDGE-B =', Prox_BRIDGE_B_Config['StepSize'])
Prox_BRIDGE_B_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])

Prox_GeoMed_Config = optConfig.copy()
Prox_GeoMed_Config['StepSize'] = 0.4
print('Step-size of Prox-GeoMed =', Prox_GeoMed_Config['StepSize'])
Prox_GeoMed_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])

Prox_Peng_Config = optConfig.copy()
Prox_Peng_Config['StepSize'] = 0.4
print('Step-size of Prox-Peng =', Prox_Peng_Config['StepSize'])
Prox_Peng_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
Prox_Peng_Config['PenaltyPara'] = 0.0005
print('penalty parameters of Prox-Peng =', Prox_Peng_Config['PenaltyPara'])
Prox_Peng_Config['BatchSize'] = int( N/m/10 )
print('BatchSize =', Prox_Peng_Config['BatchSize'])
Prox_Peng_Config['Iterations'] = int(epochs * SM.b / Prox_Peng_Config['BatchSize'])

Prox_DBRO_LSVRG_Config = optConfig.copy()
Prox_DBRO_LSVRG_Config['StepSize'] = 0.001
print('Step-size of Prox-DBRO-LSVRG =', Prox_DBRO_LSVRG_Config['StepSize'])
Prox_DBRO_LSVRG_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
Prox_DBRO_LSVRG_Config['PenaltyPara'] = 0.05 + 0.005 * np.random.random((optConfig['NodeSize'],))
# print('penalty parameters of Prox-DBRO-LSVRG:', Prox_DBRO_LSVRG_Config['PenaltyPara'])
Prox_DBRO_LSVRG_Config['Triggered Probability'] = SM.m / SM.N * np.ones(m) / 2 + SM.m / SM.N * np.random.rand(m) / 2

Prox_DBRO_SAGA_Config = optConfig.copy()
Prox_DBRO_SAGA_Config['StepSize'] = 0.005
print('Step-size of Prox-DBRO-SAGA =', Prox_DBRO_SAGA_Config['StepSize'])
Prox_DBRO_SAGA_Config['Initialization'] = cp.deepcopy(PG_EXTRA_Config['Initialization'])
Prox_DBRO_SAGA_Config['PenaltyPara'] = 0.01 + 0.005 * np.random.random((optConfig['NodeSize'],))  # 0.00001
# print('penalty parameters of Prox-DBRO-SAGA =', Prox_DBRO_SAGA_Config['PenaltyPara'])
Prox_DBRO_SAGA_Config['Iterations'] = int(epochs * SM.b)

# generate the weight matrix for NIDS-type algorithms and PG-EXTRA
temp1 = 1 / (2 * np.max(NIDS_Config['StepSize']))
print("the weighted matrix parameter of NIDS =", temp1)
W_NIDS = np.eye(m) - temp1 * np.matmul(NIDS_Config['StepSize'], np.eye(m) - M)
# if temn1 is sufficeintly large, there is no communication happening, which further means the consensus error becomes large.

temp2 = 1 / (2 * np.max(NIDS_SAGA_Config['StepSize']))
print("the weighted matrix parameter of NIDS-SAGA =", temp2)
W_NIDS_SAGA = np.eye(m) - temp2 * np.matmul(NIDS_SAGA_Config['StepSize'], np.eye(m) - M)

temp3 = 1 / (2 * np.max(NIDS_LSVRG_Config['StepSize']))
print("the weighted matrix parameter of NIDS-LSVRG =", temp3)
W_NIDS_LSVRG = np.eye(m) - temp3 * np.matmul(NIDS_LSVRG_Config['StepSize'], np.eye(m) - M)

temp4 = 0.25  # [0,0.5]
# print("the weighted matrix parameter of PG-EXTRA =", temp4)
W_2 = (1 - temp4) * M + temp4 * np.eye(m)
PG_EXTRA_Config['WeightMatrix2'] = W_2

"""save results"""
dir_save_txt = 'results/txtdata/'
dir_save_pdf = 'results/pdfdata/'
if os.path.exists('theta_F_opt' + attack_type + str(ByzantineSize / SM.m) + '.npz'):  # 最优解和最优值已知
    data = np.load('theta_F_opt' + attack_type + str(ByzantineSize / SM.m) + '.npz')
    theta_opt = data['theta_opt']
    F_opt = data['F_opt']
    optConfig['OptimalModel'] = theta_opt
    optConfig['OptimalValue'] = F_opt
else:
    print("Please run ByzantineNetwork.py first!")

if __name__ == '__main__':  # 主程序入口
    """
    Centralized solutions
    """
    if os.path.exists('theta_F_opt' + attack_type + str(ByzantineSize / SM.m) + '.npz'):  # 最优解和最优值已知
        data = np.load('theta_F_opt' + attack_type + str(ByzantineSize / SM.m) + '.npz')
        theta_opt = data['theta_opt']
        F_opt = data['F_opt']
        print('The optimal model and value of CPGD theta_opt: {}, F_opt: {}'.format(theta_opt, F_opt))
        error_SM = error(SM, theta_opt, F_opt)
    else:  # 最优解和最优值未知
        # solve the optimal solution of Logistic regression
        x_CPGD, theta_opt, F_opt = copt.CPGD(SM, CPGD_Config, image_test, label_test, len(Reliable))
        error_SM = error(SM, theta_opt, F_opt)
        F_gap_CPGD = error_SM.loss_gap_path_c(x_CPGD, CPGD_Config['ReliableSet'])
        # 存储 theta_opt 和 F_opt
        np.savez('theta_F_opt' + attack_type + str(ByzantineSize / SM.m) + '.npz', theta_opt=theta_opt, F_opt=F_opt)

    # ------------------------  Below are decentralized Byzantine-vunerable methods ----------------------#
    ## PG-EXTRA
    # x_PG_EXTRA, x_PG_EXTRA_ave_list, time_PG_EXTRA, res_NIDS, loss_PG_EXTRA = dopt.PG_EXTRA( SM, PG_EXTRA_Config, image_test, label_test,
    # attack=Attack, dsgd_optimal=theta_opt)
    # F_gap_PG_EXTRA = error_SM.loss_gap_path( loss_PG_EXTRA, Reliable )
    # x_res_PG_EXTRA = error_SM.theta_gap_path( x_PG_EXTRA, Reliable )

    # NIDS
    # x_NIDS, x_NIDS_ave_list, time_NIDS, res_NIDS, loss_NIDS, Setting = dopt.NIDS(SM, NIDS_Config, image_test, label_test, attack=Attack,
    # dsgd_optimal=theta_opt)
    # F_gap_NIDS = error_SM.loss_gap_path( loss_NIDS, Reliable )
    # x_res_NIDS = error_SM.theta_gap_path(x_NIDS, Reliable)
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'NIDS' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # F_gap_NIDS )

    """
    # NIDS-SAGA
    # x_NIDS_SAGA, x_NIDS_ave_SAGA_list, time_NIDS_SAGA, loss_NIDS_SAGA, Setting = dopt.NIDS_SAGA( SM, NIDS_SAGA_Config, image_test, label_test, attack = Attacks )
    # F_gap_NIDS_SAGA = error_SM.loss_gap_path( loss_NIDS_SAGA, Reliable )
    # x_res_NIDS_SAGA = error_SM.theta_gap_path( x_NIDS_SAGA, Reliable )

    # NIDS-LSVRG
    # x_NIDS_LSVRG, x_NIDS_ave_LSVRG_list, time_NIDS_LSVRG, loss_NIDS_LSVRG, Setting = dopt.NIDS_LSVRG( SM, NIDS_LSVRG_Config, image_test, label_test, attack = Attack )
    # F_gap_NIDS_LSVRG = error_SM.loss_gap_path( loss_NIDS_LSVRG, Reliable )
    # x_res_NIDS_LSVRG = error_SM.theta_gap_path( x_NIDS_LSVRG, Reliable )
    """

    # PMGT-SAGA
    # x_PMGT_SAGA, x_PMGT_SAGA_ave_list, time_PMGT_SAGA, res_PMGT_SAGA, loss_PMGT_SAGA, Setting = dgt.PMGT_SAGA( SM, PMGT_SAGA_Config, image_test,
    # label_test, attack=Attack, dsgd_optimal=theta_opt)
    # x_res_PMGT_SAGA = error_SM.theta_gap_path( x_PMGT_SAGA, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-Residual-' + 'PMGT-SAGA' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # x_res_PMGT_SAGA )
    # F_gap_PMGT_SAGA = error_SM.loss_gap_path( loss_PMGT_SAGA, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'PMGT-SAGA' + Setting + str(ByzantineSize / SM.m) + '.txt', F_gap_PMGT_SAGA )

    # PMGT-LSVRG
    # x_PMGT_LSVRG, x_PMGT_LSVRG_ave_list, time_PMGT_LSVRG, res_PMGT_LSVRG, loss_PMGT_LSVRG, Setting = dgt.PMGT_LSVRG( SM, PMGT_LSVRG_Config,
    # image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # x_res_PMGT_LSVRG = error_SM.theta_gap_path( x_PMGT_LSVRG, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-Residual-' + 'PMGT-LSVRG' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # x_res_PMGT_LSVRG )
    # F_gap_PMGT_LSVRG = error_SM.loss_gap_path( loss_PMGT_LSVRG, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'PMGT-LSVRG' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # F_gap_PMGT_LSVRG)

    # ------------------------ Prox-BRIDGE framework with trimmed-mean, Median, Krum, Bulyan ----------------------#
    # Prox-BRIDGE-T
    # x_Prox_BRIDGE_T, x_Prox_BRIDGE_T_ave_list, time_Prox_BRIDGE_T, res_Prox_BRIDGE_T, loss_Prox_BRIDGE_T, Setting = dopt.Prox_BRIDGE_T( SM,
    # Prox_BRIDGE_T_Config, image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # x_res_Prox_BRIDGE_T = error_SM.theta_gap_path( x_Prox_BRIDGE_T, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-Residual-' + 'Prox-BRIDGE-T' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # x_res_Prox_BRIDGE_T )
    # F_gap_Prox_BRIDGE_T = error_SM.loss_gap_path( loss_Prox_BRIDGE_T, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'Prox-BRIDGE-T' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # F_gap_Prox_BRIDGE_T )

    # Prox-BRIDGE-M
    # x_Prox_BRIDGE_M, x_Prox_BRIDGE_M_ave_list, time_Prox_BRIDGE_M, res_Prox_BRIDGE_M, loss_Prox_BRIDGE_M, Setting = dopt.Prox_BRIDGE_M( SM,
    # Prox_BRIDGE_M_Config, image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # # x_res_Prox_BRIDGE_M = error_SM.theta_gap_path( x_Prox_BRIDGE_M, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-Residual-' + 'Prox-BRIDGE-M' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # res_Prox_BRIDGE_M )
    # F_gap_Prox_BRIDGE_M = error_SM.loss_gap_path( loss_Prox_BRIDGE_M, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'Prox-BRIDGE-M' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # F_gap_Prox_BRIDGE_M )

    # Prox-BRIDGE-K
    # x_Prox_BRIDGE_K, x_Prox_BRIDGE_K_ave, time_Prox_BRIDGE_K, res_Prox_BRIDGE_K, loss_Prox_BRIDGE_K, Setting = dopt.Prox_BRIDGE_K( SM,
    # Prox_BRIDGE_K_Config, image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # # x_res_Prox_BRIDGE_K = error_SM.theta_gap_path( x_Prox_BRIDGE_K, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-Residual-' + 'Prox-BRIDGE-K' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # res_Prox_BRIDGE_K )
    # F_gap_Prox_BRIDGE_K = error_SM.loss_gap_path( loss_Prox_BRIDGE_K, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'Prox-BRIDGE-K' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # F_gap_Prox_BRIDGE_K )

    # Prox-BRIDGE-B
    # x_Prox_BRIDGE_B, x_Prox_BRIDGE_B_ave, time_Prox_BRIDGE_B, res_Prox_BRIDGE_B, loss_Prox_BRIDGE_B, Setting = dopt.Prox_BRIDGE_B( SM,
    # Prox_BRIDGE_B_Config, image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # F_gap_Prox_BRIDGE_B = error_SM.loss_gap_path( loss_Prox_BRIDGE_B, Reliable )
    # x_res_Prox_BRIDGE_B = error_SM.theta_gap_path( x_Prox_BRIDGE_B, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'Prox-BRIDGE-B' + Setting + str(ByzantineSize / SM.m) + '.txt',\
    # F_gap_Prox_BRIDGE_B )

    # Prox-GeoMed
    # x_Prox_GeoMed, x_Prox_GeoMed_ave_list, time_Prox_GeoMed, res_Prox_GeoMed, loss_Prox_GeoMed, Setting = dopt.Prox_GeoMed( SM, Prox_GeoMed_Config, image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # F_gap_Prox_GeoMed = error_SM.loss_gap_path( loss_Prox_GeoMed, Reliable )
    # x_res_Prox_GeoMed = error_SM.theta_gap_path(x_Prox_GeoMed, Reliable)
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'Prox-GeoMed' + Setting + str(ByzantineSize / SM.m) + '.txt', F_gap_Prox_GeoMed )

    # --------------------------------------- Prox-Peng framework -------------------------------------#
    # Prox-Peng
    # x_Prox_Peng, x_Prox_Peng_ave_list, time_Prox_Peng, res_Prox_Peng, loss_Prox_Peng, Setting, PN_Setting = dopt.Prox_Peng( SM, Prox_Peng_Config,
    # image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # F_gap_Prox_Peng = error_SM.loss_gap_path( loss_Prox_Peng, Reliable )
    # x_res_Prox_Peng = error_SM.theta_gap_path( x_Prox_Peng, Reliable )
    # np.savetxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-Peng' + Setting + str(ByzantineSize / SM.m) + '.txt', F_gap_Prox_Peng)

    # --------------------------------------- Prox-DBRO-VR framework -------------------------------------#
    # Prox-DBRO-LSVRG
    x_Prox_DBRO_LSVRG, x_Prox_DBRO_LSVRG_ave_list, time_Prox_DBRO_LSVRG, res_Prox_DBRO_LSVRG, loss_Prox_DBRO_LSVRG, Setting, PN_Setting =\
        dopt.Prox_DBRO_LSVRG( SM, Prox_DBRO_LSVRG_Config, image_test, label_test,  attack=Attack, dsgd_optimal=theta_opt)
    # x_res_Prox_DBRO_LSVRG = error_SM.theta_gap_path( x_Prox_DBRO_LSVRG, Reliable )
    np.savetxt(dir_save_txt + SM.setting + '-Residual-' + 'Prox-DBRO-LSVRG' + Setting + str(ByzantineSize / SM.m) + '.txt', res_Prox_DBRO_LSVRG)
    F_gap_Prox_DBRO_LSVRG = error_SM.loss_gap_path( loss_Prox_DBRO_LSVRG, Reliable )
    np.savetxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-LSVRG' + Setting + str(ByzantineSize / SM.m) + '.txt', F_gap_Prox_DBRO_LSVRG)

    # Prox-DBRO-SAGA
    # x_Prox_DBRO_SAGA, x_Prox_DBRO_SAGA_ave_list, time_Prox_DBRO_SAGA, res_Prox_DBRO_SAGA, loss_Prox_DBRO_SAGA, Setting, PN_Setting = dopt.Prox_DBRO_SAGA( SM, Prox_DBRO_SAGA_Config, image_test, label_test, attack=Attack, dsgd_optimal=theta_opt)
    # np.savetxt(
    #     dir_save_txt + SM.setting + '-Residual-' + 'Prox-DBRO-SAGA' + Setting + str(ByzantineSize / SM.m) + '.txt', res_Prox_DBRO_SAGA)
    # F_gap_Prox_DBRO_SAGA = error_SM.loss_gap_path( loss_Prox_DBRO_SAGA, Reliable )
    # # x_res_Prox_DBRO_SAGA = error_SM.theta_gap_path( x_Prox_DBRO_SAGA, Reliable )
    # np.savetxt( dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-SAGA' + Setting + str(ByzantineSize / SM.m) + '.txt', F_gap_Prox_DBRO_SAGA )

    sys.exit(0)  # skip the following execution
    # Plot results
    Marker_Size = 10
    mark_distance = 50
    line_width = 1.5
    font = FontProperties()
    font.set_size(16)
    font2 = FontProperties()
    font2.set_size(10)
    # # color and label
    # colors = ['orange', '#00C9A7','#A35EC2', '#FF6F91', '#008CCB', 'red']
    # markers = ['<', '>', '^', 'v', 'o', 's']
    colors = ['#1F77b4', '#9467BD', '#2CA02C', '#E377C2', '#FF7F0E', '#8C564B', '#17BECF', '#BCBD22', 'blue', 'red','#7F7F7F', '#FFC75F']
    markers = ['s', 'P', 'X', '^', 'v', '<', '>', 'o', 'd', 'D', 'p', '+', 'x', 'h', 'H', '*']
    # methods = ['NIDS', 'PMGT-LSVRG', 'PMGT-SAGA', 'Prox-BRIDGE-T', 'Prox-BRIDGE-M', 'Prox-BRIDGE-K', 'Prox-GeoMed', 'Prox-DBRO-LSVRG', 'Prox-DBRO-SAGA']
    # labels = ['NIDS', 'PMGT_LSVRG', 'PMGT_SAGA', 'Prox_BRIDGE_T', 'Prox_BRIDGE_M', 'Prox_BRIDGE_K', 'Prox_GeoMed', 'Prox_DBRO_LSVRG', 'Prox_DBRO_SAGA']
    methods = ['NIDS', 'PMGT-LSVRG', 'PMGT-SAGA', 'Prox-BRIDGE-T', 'Prox-BRIDGE-M', 'Prox-BRIDGE-K', 'Prox-GeoMed', 'Prox-Peng', 'Prox-DBRO-LSVRG', 'Prox-DBRO-SAGA']
    labels = ['NIDS', 'PMGT_LSVRG', 'PMGT_SAGA', 'Prox_BRIDGE_T', 'Prox_BRIDGE_M', 'Prox_BRIDGE_K', 'Prox_GeoMed', 'Prox_Peng', 'Prox_DBRO_LSVRG', 'Prox_DBRO_SAGA']

    """ Save results """
    F_gap = []
    Res = []
    time_list = []
    for i in range(len(methods)):
        # Save running time of all tested algorithms
        time_list.append(eval('time_' + labels[i]))
        np.savetxt(
            dir_save_txt + SM.setting + '-time-' + methods[i] + Setting + str(ByzantineSize / SM.m) + '.txt', time_list[i])
        # Save the Function gap of all tested algorithms
        Res.append(eval('Res_' + labels[i]))
        np.savetxt(
            dir_save_txt + SM.setting + '-Residual-' + methods[i] + Setting + str(ByzantineSize / SM.m) + '.txt', Res[i])
        F_gap.append(eval('F_gap_' + labels[i]))
        np.savetxt(
            dir_save_txt + SM.setting + '-F-gap-' + methods[i] + Setting + str(ByzantineSize / SM.m) + '.txt', F_gap[i])

    # -------------------------------Plot residual in terms of epochs-------------------------------#
    print("Below plots results:")
    FigSize = (6.4, 4.8)
    plt.figure(0, figsize=FigSize)
    X_limit = epochs / 2
    for i in range(len(methods)):
        Res = eval('Res_' + labels[i])
        # Plot the residual of all tested algorithms in terms of epochs
        plt.plot(Res, color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance,\
                 label=methods[i])
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(1e-3, 1e3)
    plt.xlabel('Epochs', fontproperties=font)
    plt.ylabel('Residual', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig(dir_save_pdf + SM.setting + '-epochs-Residual-Byzan' + Setting + str(ByzantineSize / SM.m) + '.pdf', format='pdf',\
                dpi=4000, bbox_inches='tight')

    # -------------------------------Plot optimality gap in terms of epochs-------------------------------#
    print("Below plots results:")
    FigSize = (6.4, 4.8)
    plt.figure(1, figsize=FigSize)
    """Plot each line at a time:
    # # plt.plot(F_gap_CPGD, color = colors[5], marker = markers[5], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_PG_EXTRA, color = colors[], marker = markers[], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_PMGT_SAGA, color = colors[], marker = markers[], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_PMGT_LSVRG, color = colors[], marker = markers[], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Peng, color = colors[], marker = markers[], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_NIDS, color = colors[0], marker = markers[0], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_NIDS_SAGA, color = colors[], marker = markers[], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_NIDS_LSVRG, color = colors[], marker = markers[], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_BRIDGE_T, color = colors[1], marker = markers[1], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_BRIDGE_M, color = colors[2], marker = markers[2], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_BRIDGE_K, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_BRIDGE_B, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_GeoMed, color = colors[5], marker = markers[5], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_DBRO_SAGA, color = colors[6], marker = markers[6], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    # plt.plot(F_gap_Prox_DBRO_LSVRG, color = colors[7], marker = markers[7], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    """
    X_limit = epochs / 2
    for i in range(len(methods)):
        F_gap = eval('F_gap_' + labels[i])
        # Plot the Function gap of all tested algorithms in terms of epochs
        plt.plot(F_gap, color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance,\
                 label=methods[i])
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(1e-3, 1e3)
    plt.xlabel('Epochs', fontproperties=font)
    plt.ylabel('Optimality gap', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig(dir_save_pdf + SM.setting + '-epochs-F-gap-Byzan' + Setting + str(ByzantineSize / SM.m) + '.pdf', format='pdf',\
                dpi=4000, bbox_inches='tight')

    # -------------------------------Plot optimality gap in terms of time-------------------------------#
    plt.figure(2, figsize=FigSize)
    """ Plot each line at a time:
    plt.plot(time_Prox_Prox_DBRO_SAGA, F_gap_Prox_Prox_DBRO_SAGA, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    plt.plot(time_Prox_DBRO_LSVRG, F_gap_Prox_DBRO_LSVRG, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    """
    for i in range(len(methods)):
        time_list = eval('time_' + labels[i])
        F_gap = eval('F_gap_' + labels[i])
        plt.plot(time_list, F_gap, color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size,
                 markevery=mark_distance, label=methods[i])
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = '-No Byzantine agents-'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(1e-3, 1e3)
    plt.xlabel('Time (s)', fontproperties=font)
    plt.ylabel('Optimality gap', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig(dir_save_pdf + SM.setting + '-Time-F-gap-Byzan' + Setting + str(ByzantineSize / SM.m) + '.pdf', format='pdf',\
                dpi=4000, bbox_inches='tight')

    # -------------------------------Plot consensus errors in terms of epochs-------------------------------#
    """Load results of consensus errors"""
    Consensus_error = []
    for i in range(len(methods)):
        # Load the consensus errors of all tested algorithms
        tmp_consensus_error = np.loadtxt(dir_save_txt + SM.setting + '-consensus-error-' + methods[i] + Setting + \
                                         str(optConfig['ByzantineSize'] / SM.m) + '.txt', dtype=float, delimiter=None)
        Consensus_error.append(tmp_consensus_error)

    plt.figure(3, figsize=FigSize)
    for i in range(len(methods)):
        plt.plot(Consensus_error[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, \
                 markevery=mark_distance, label=methods[i])
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(1e-8, 1e5)
    plt.xlabel('epochs', fontproperties=font)
    plt.ylabel('Consensus error', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig(
        dir_save_pdf + SM.setting + '-Epoch-consensus-error-Byzan' + Setting + str(ByzantineSize / SM.m) + '.pdf', \
        format='pdf', dpi=4000, bbox_inches='tight')

    # -------------------------------Plot consensus errors in terms of time-------------------------------#
    plt.figure(4, figsize=FigSize)
    """ Plot each line at a time:
    plt.plot(time_Prox_Prox_DBRO_SAGA, F_gap_Prox_Prox_DBRO_SAGA, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    plt.plot(time_Prox_DBRO_LSVRG, F_gap_Prox_DBRO_LSVRG, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    """
    for i in range(len(methods)):
        plt.plot(time_list, Consensus_error[i], color=colors[i], marker=markers[i], linewidth=line_width,
                 markersize=Marker_Size, markevery=mark_distance, label=methods[i])
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(1e-8, 1e5)
    plt.xlabel('Time (s)', fontproperties=font)
    plt.ylabel('Consensus error', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig(
        dir_save_pdf + SM.setting + '-Time-consensus-error-Byzan' + Setting + str(ByzantineSize / SM.m) + '.pdf',
        format='pdf', dpi=4000, bbox_inches='tight')

    # -------------------------------Plot testing accuracy in terms of epochs-------------------------------#
    """Load results of testing accuracy"""
    Testing_accuracy = []
    for i in range(len(methods)):
        # Load the testing accuracy of all tested algorithms
        tmp_testing_accuracy = np.loadtxt(dir_save_txt + SM.setting + '-acc-' + methods[i] + Setting + \
                                          str(optConfig['ByzantineSize'] / SM.m) + '.txt', dtype=float, delimiter=None)
        Testing_accuracy.append(tmp_testing_accuracy)

    plt.figure(5, figsize=FigSize)
    for i in range(len(methods)):
        plt.plot(Testing_accuracy[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, \
                 markevery=mark_distance, label=methods[i])
    plt.grid(True)
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(0.75, 1)
    plt.xlabel('epochs', fontproperties=font)
    plt.ylabel('Testing Accuracy', fontproperties=font)
    plt.legend(prop=font2, loc="lower right")
    plt.savefig(
        dir_save_pdf + SM.setting + '-Epoch-testing-accuracy-Byzan' + Setting + str(
            ByzantineSize / SM.m) + '.pdf', format='pdf', dpi=4000, bbox_inches='tight')

    # -----------------------------Plot testing accuracy in terms of time-------------------------------#
    plt.figure(6, figsize=FigSize)
    """ Plot each line at a time:
    plt.plot(time_Prox_Prox_DBRO_SAGA, F_gap_Prox_Prox_DBRO_SAGA, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    plt.plot(time_Prox_DBRO_LSVRG, F_gap_Prox_DBRO_LSVRG, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
    """
    for i in range(len(methods)):
        plt.plot(time_list, Testing_accuracy[i], color=colors[i], marker=markers[i], linewidth=line_width,
                 markersize=Marker_Size, markevery=mark_distance, label=methods[i])
    plt.grid(True)
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attacks.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(-5, X_limit)
    plt.ylim(0.65, 1)
    plt.xlabel('Time (s)', fontproperties=font)
    plt.ylabel('Testing accuracy', fontproperties=font)
    plt.legend(prop=font2, loc="lower right")
    plt.savefig(
        dir_save_pdf + SM.setting + '-Time-testing-accuracy-Byzan' + Setting + str(ByzantineSize / SM.m) + '.pdf', \
        format='pdf', dpi=4000, bbox_inches='tight')

    # generate Byzantine and reliable graphs
    """-------------------------------Plot the Byzantine graphs-------------------------------"""
    pos = nx.random_layout(G)
    labels = {x: x for x in G.nodes}
    plt.figure(7, figsize=FigSize)
    nx.draw_networkx_labels(G, pos, labels, font_size=8.5, font_color='w')
    # a = '''
    nodes = {
        '#FFA326': Reliable,
        '#FF0026': Byzantine,
    }
    for node_color, nodelist in nodes.items():
        nx.draw_networkx_nodes(G, pos, node_size=120, nodelist=nodelist, node_color=node_color)

    edge_list = {x: x for x in G.edges}
    # for i, j in edge_list:
    #     G.add_edge(i, j)
    nx.draw_networkx_edges(G, pos, edge_list, edge_color='#0073BD')
    plt.axis('off')
    plt.savefig(dir_save_pdf + SM.setting + '-Byzantine-Graph' + Setting + str(ByzantineSize / SM.m) + '.pdf',
                format='pdf', dpi=4000, bbox_inches='tight')

    plt.figure(8, figsize=FigSize)
    nx.draw(H, pos, with_labels=True, font_size=8.5, font_color='w', node_size=120, node_color='#FFA326', width=1,
            node_shape=None, alpha=None, edge_color='#0073BD')
    # nx.draw(G, pos, with_labels=False, node_size=120, node_color='#FFA326', node_shape=None, alpha=None, linewidths=8, edge_color = '#0073BD')
    plt.axis('off')
    plt.savefig(dir_save_pdf + SM.setting + '-Reliable-Graph' + Setting + str(ByzantineSize / SM.m) + '.pdf',
                format='pdf', dpi=4000, bbox_inches='tight')
    plt.show()
    plt.close()
