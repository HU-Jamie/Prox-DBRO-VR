################################################################################################################################
##---------------------------------------------------Centralized Optimizers---------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
from tqdm import tqdm
from analysis import get_consensus_error, get_accuracy


## Centralized proximal-gradient descent
def CPGD(pr, CPGD_Config, image_test, label_test, local_obj_num):
    theta = [ CPGD_Config['Initialization'] ]
    acc_CPGD_list = []
    loss_CPGD_list = []
    for k in tqdm( range( CPGD_Config['Iterations'] ) ):
        theta.append( pr.prox_l1( theta[-1] - CPGD_Config['StepSize'] * pr.grad(theta[-1], CPGD_Config['ReliableSet']), ( pr.reg_l1 * local_obj_num ) * CPGD_Config['StepSize'], 'CPGD' ) )
        if k % 1 == 0:
            acc = get_accuracy( theta[-1], image_test, label_test )
            acc_CPGD_list.append(acc)
            loss = pr.call_loss_cen( theta[-1], CPGD_Config['ReliableSet'] )
            loss_CPGD_list.append(loss)
            if k % (CPGD_Config['Iterations']/10) == 0:
                print('CPGD of the {}th iteration loss: {}, acc: {}'.format(k, loss, acc))
        ut.monitor( 'CPGD', k, CPGD_Config['Iterations'] )
    theta_opt = theta[-1]
    F_opt = loss_CPGD_list[-1]
    print('CPGD of the final iteration loss: {}, acc: {}'.format(loss_CPGD_list[-1], acc_CPGD_list[-1]))
    return theta, theta_opt, F_opt

## Centralized gradient descent
def CGD(pr,learning_rate,K,theta_0):
    theta = [theta_0]
    for k in range(K):
        theta.append( theta[-1] - learning_rate * pr.grad(theta[-1]) )
        ut.monitor('CGD', k, K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    print('optimal function value of CGD:', F_opt)
    # print('optimal grad:', pr.grad(theta[-1]))
    return theta, theta_opt, F_opt

## Centralized gradient descent with momentum
def CNGD(pr,learning_rate,momentum,K,theta_0, Reliable, image_test, label_test,):
    theta = [theta_0]  
    theta_aux = cp.deepcopy(theta_0)
    acc_CGD = []
    for k in range(K):
        grad = pr.grad(theta[-1])
        theta_aux_last = cp.deepcopy(theta_aux)
        theta_aux = theta[-1] - learning_rate * grad 
        theta.append( theta_aux + momentum * ( theta_aux - theta_aux_last ) )
        if (k+1) % ( pr.b ) == 0:   # DBRO-SAGA requires prd.b/2 iterations to finish an epoch.
            acc = get_accuracy( theta[-1], image_test, label_test )
            acc_CGD.append(acc)
            if (k + 1) % (K/10) == 0:  # DBRO-SAGA requires prd.b/2 iterations to finish an epoch.
                print('the {}th iteration acc: {}'.format(k, acc))
        ut.monitor('CNGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.call_loss_cen(theta[-1], Reliable)
    print('optimal function value of CGD:', F_opt)
    return theta, theta_opt, F_opt

## Centralized stochastic gradient descent
def CSGD(pr,learning_rate,K,theta_0):
    N = pr.N
    theta = cp.deepcopy(theta_0)
    theta_epoch = [ theta_0 ]
    for k in range(K):
        idx = np.random.randint(0,N)
        grad = pr.grad(theta,idx)
        theta -= learning_rate * grad 
        if (k+1) % N == 0:
            theta_epoch.append( cp.deepcopy(theta) )
        ut.monitor('CSGD',k,K)
    return theta_epoch

## Centralized gradient descent with variance reduction using SAGA
def CSAGA(pr,learning_rate,K,theta_0):
    N = pr.N
    theta = cp.deepcopy( theta_0 )
    slots_gradient = np.zeros((N,pr.p))    
    for i in range(N):
        slots_gradient[i] = pr.grad(theta, i)
    sum_gradient = np.sum(slots_gradient,axis = 0)
    theta_epoch = [ theta_0 ]
    for k in range(K-1):
        idx = np.random.randint(0,N)
        grad = pr.grad(theta, idx)
        gradf = grad - slots_gradient[idx]
        SAGA = gradf + sum_gradient/N
        sum_gradient += gradf
        slots_gradient[idx] = cp.deepcopy(grad)
        theta -= learning_rate * SAGA
        if (k+1) % N == 0:
            theta_epoch.append( cp.deepcopy(theta) )    
        ut.monitor('CSAGA',k,K)
    return theta_epoch