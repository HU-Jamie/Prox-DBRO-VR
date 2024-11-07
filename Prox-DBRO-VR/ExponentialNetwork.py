########################################################################################################################
####----------------------------------------------Exponential Network-----------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Byzan_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt

########################################################################################################################
####----------------------------------------------MNIST Classification----------------------------------------------####
########################################################################################################################
"""
Data processing for MNIST
"""
n = 16                                                      ## number of nodes 
Problem = LR_L2(n, limited_labels = False, balanced = True )   ## instantiate the problem class
p = Problem.p                                                  ## dimension of the model
L = Problem.L                                                  ## L-smooth constant
N = Problem.N                                                  ## total number of training samples
b = Problem.b                                                  ## average number of local samples
step_size = 1/L/2                                           ## selecting an appropriate step-size

"""
Initializing variables
"""
CEPOCH_base = N/b * 10
depoch = N/b
theta_c0 = np.random.normal(0, 1, p)
theta_0 = np.random.normal(0, 1, (n, p))
UG = Exponential_graph(n).directed()
B = Weight_matrix(UG).column_stochastic()

"""
Centralized solutions
"""
## solve the optimal solution of Logistic regression
_, theta_opt, F_opt = copt.CGD(Problem, 10*1/L, CEPOCH_base, theta_c0)
error_Problem = error(Problem,theta_opt,F_opt)

"""
Decentralized Algorithms
"""
## GP
theta_GP = dopt.GP(Problem,B,step_size,int( depoch),theta_0)
res_F_GP = error_Problem.cost_gap_path( np.sum(theta_GP,axis = 1)/n)
## ADDOPT
theta_ADDOPT = dopt.ADDOPT(Problem,B,B,step_size,int( depoch ),theta_0)
res_F_ADDOPT = error_Problem.cost_gap_path( np.sum(theta_ADDOPT,axis = 1)/n)
## SGP
theta_SGP = dopt.SGP(Problem,B,step_size,int( depoch*b),theta_0)
res_F_SGP = error_Problem.cost_gap_path( np.sum(theta_SGP,axis = 1)/n)
## SADDOPT               
theta_SADDOPT = dopt.SADDOPT(Problem,B,B,step_size,int( depoch*b),theta_0)
res_F_SADDOPT = error_Problem.cost_gap_path( np.sum(theta_SADDOPT,axis = 1)/n)
## PushSAGA     
theta_PushSAGA = dopt.Push_SAGA(Problem,B,B,step_size,int( depoch*b),theta_0)
res_F_PushSAGA = error_Problem.cost_gap_path( np.sum(theta_PushSAGA,axis = 1)/n)

"""
Save data
"""
np.savetxt('plots_MnistResGP.txt', res_F_GP)
np.savetxt('plots_MnistResADDOPT.txt', res_F_ADDOPT)
np.savetxt('plots_MnistResSGP.txt', res_F_SGP)
np.savetxt('plots_MnistResSADDOPT.txt', res_F_SADDOPT)
np.savetxt('plots_MnistResPushSAGA.txt', res_F_PushSAGA)

"""
Save plot
"""
mark_every = 10
font = FontProperties()
font.set_size(18)
font2 = FontProperties()
font2.set_size(10)
plt.figure(1)
plt.plot(res_F_GP,'-vb', markevery = mark_every)
plt.plot(res_F_ADDOPT,'-^m', markevery = mark_every)
plt.plot(res_F_SGP,'-dy', markevery = mark_every)
plt.plot(res_F_SADDOPT,'->c', markevery = mark_every)
plt.plot(res_F_PushSAGA,'-sr', markevery = mark_every)
plt.grid(True)
plt.yscale('log')
plt.tick_params(labelsize='large', width=3)
plt.title('MNIST', fontproperties=font)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Optimality Gap', fontproperties=font)
plt.legend(('GP', 'ADDOPT', 'SGP', 'SADDOPT', 'PushSAGA'), prop=font2)
plt.savefig('plotsExpMnist.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')

plt.figure(2)
G = nx.from_numpy_matrix(np.matrix(B), create_using=nx.DiGraph)
layout = nx.circular_layout(G)
nx.draw(G, layout)
plt.savefig('plotsExpGraph.pdf', format = 'pdf', dpi = 4000, pad_inches=0,\
            bbox_inches ='tight')

########################################################################################################################
####--------------------------------------------CIFAR-10 Classification---------------------------------------------####
########################################################################################################################

"""
Data processing for CIFAR
"""
lr_1 = LR_L4(n, limited_labels = False, balanced = True )   ## instantiate the problem class
p = lr_1.p                                                  ## dimension of the model 
L = lr_1.L                                                  ## L-smooth constant
N = lr_1.N                                                  ## total number of training samples
b = lr_1.b                                                  ## average number of local samples
step_size = 1/L/2                                           ## selecting an appropriate step-size

"""
Initializing variables
"""
CEPOCH_base = 5000
depoch = 100
theta_c0 = np.random.normal(0,1,p)
theta_0 = np.random.normal(0,1,(n,p)) 

"""
Centralized solutions
"""
## solve the optimal solution of Logistic regression
_, theta_opt, F_opt = copt.CGD(lr_1,10*1/L, CEPOCH_base,theta_c0) 
error_lr_1 = error(lr_1,theta_opt,F_opt)


"""
Decentralized Algorithms
"""
## GP
theta_GP = dopt.GP(lr_1,B,step_size,int( depoch),theta_0)  
res_F_GP = error_lr_1.cost_gap_path( np.sum(theta_GP,axis = 1)/n)
## ADDOPT
theta_ADDOPT = dopt.ADDOPT(lr_1,B,B,step_size,int( depoch ),theta_0)  
res_F_ADDOPT = error_lr_1.cost_gap_path( np.sum(theta_ADDOPT,axis = 1)/n)
## SGP
theta_SGP = dopt.SGP(lr_1,B,step_size,int( depoch*b),theta_0)  
res_F_SGP = error_lr_1.cost_gap_path( np.sum(theta_SGP,axis = 1)/n)
## SADDOPT               
theta_SADDOPT = dopt.SADDOPT(lr_1,B,B,step_size,int( depoch*b),theta_0)  
res_F_SADDOPT = error_lr_1.cost_gap_path( np.sum(theta_SADDOPT,axis = 1)/n)
## PushSAGA     
theta_PushSAGA = dopt.Push_SAGA(lr_1,B,B,step_size,int( depoch*b),theta_0)  
res_F_PushSAGA = error_lr_1.cost_gap_path( np.sum(theta_PushSAGA,axis = 1)/n)

"""
Save data
"""
np.savetxt('plots_ExpCifarResGP.txt', res_F_GP)
np.savetxt('plots_ExpCifarResADDOPT.txt', res_F_ADDOPT)
np.savetxt('plots_ExpCifarResSGP.txt', res_F_SGP)
np.savetxt('plots_ExpCifarResSADDOPT.txt', res_F_SADDOPT)
np.savetxt('plots_ExpCifarResPushSAGA.txt', res_F_PushSAGA)

"""
Save plot
"""
mark_every = 10
font = FontProperties()
font.set_size(18)
font2 = FontProperties()
font2.set_size(10)
plt.figure(3)
plt.plot(res_F_GP,'-vb', markevery = mark_every)
plt.plot(res_F_ADDOPT,'-^m', markevery = mark_every)
plt.plot(res_F_SGP,'-dy', markevery = mark_every)
plt.plot(res_F_SADDOPT,'->c', markevery = mark_every)
plt.plot(res_F_PushSAGA,'-sr', markevery = mark_every)
plt.grid(True)
plt.yscale('log')
plt.tick_params(labelsize='large', width=3)
plt.title('CIFAR-10', fontproperties=font)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Optimality Gap', fontproperties=font)
plt.legend(('GP', 'ADDOPT', 'SGP', 'SADDOPT', 'PushSAGA'), prop=font2)
plt.savefig('plotsExpCifar.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')
