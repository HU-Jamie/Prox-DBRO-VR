################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################
import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA
from tqdm import tqdm
from analysis import get_residual, get_consensus_error, get_accuracy
import time

def get_neighbors(prd, id, G, Byzantine_set):
    neighbor_list = []
    neighbors_Byzantine = []
    neighbors_reliable = []
    for jd in range( prd.m ):
        if (id, jd) in G.edges() or id == jd:
            neighbor_list.append(jd)
            if jd in Byzantine_set:
                neighbors_Byzantine.append(jd)
            else:
                neighbors_reliable.append(jd)
    return neighbor_list, neighbors_Byzantine, neighbors_reliable

def Push_SAGA(prd,B1,B2,learning_rate,K,theta_0, warmup = 0):
    theta_epoch = [ cp.deepcopy(theta_0) ]
    # if warmup > 0:
    #     warm = DSGD(prd, B1, learning_rate, warmup * prd.b, theta_0)
    #     for _ in warm[1:]:
    #         theta_epoch.append( _ )
    theta = cp.deepcopy( theta_epoch[-1] )
    slots = np.array([np.zeros((prd.data_distr[i],prd.dim)) for i in range(prd.m)])
    for id in range(prd.m):
        for j in range(prd.data_distr[id]):
            slots[id][j] = prd.localgrad( theta, id, j )
    sum_grad = np.zeros((prd.m,prd.dim))
    Y = np.ones(B1.shape[1])
    for id in range(prd.m):
        sum_grad[id] = np.sum(slots[id], axis = 0)
    SAGA = np.zeros( (prd.m,prd.dim) )
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])
    for id in range(prd.m):
        SAGA[id] = slots[id][sample_vec[id]]
    tracker = cp.deepcopy(SAGA)
    for k in range(K):
        SAGA_last = cp.deepcopy(SAGA)
        theta = np.matmul( B1, theta - learning_rate * tracker )
        Y = np.matmul( B1, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta )
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])
        grad = prd.networkgrad( z, sample_vec )   # local stocashtic gradient with index sample_vec
        for id in range(prd.m):
            StoGrad_Diff = grad[id] - slots[id][sample_vec[id]]
            SAGA[id] =  StoGrad_Diff + sum_grad[id]/prd.data_distr[id]
            sum_grad[id] += StoGrad_Diff
            slots[id][sample_vec[id]] = grad[id]
        tracker = np.matmul(B2, tracker + SAGA - SAGA_last )
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )
        ut.monitor('PushSAGA', k, K)
    return theta_epoch

def PMGT_SAGA(prd, PMGT_SAGA_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_PMGT_SAGA = []
    res_PMGT_SAGA = []
    loss_PMGT_SAGA = []
    var_PMGT_SAGA = []   # 构建List类型空列表
    acc_PMGT_SAGA = []
    para_ave_iter = []
    loss = []
    para_epoch = [ cp.deepcopy( PMGT_SAGA_Config['Initialization'] ) ]
    workerPara = cp.deepcopy( para_epoch[-1] )
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in PMGT_SAGA_Config['ReliableSet']:
        para_ave += para[id] / PMGT_SAGA_Config['ReliableSize']
    var = get_consensus_error(PMGT_SAGA_Config['ReliableSet'], para, para_ave)
    var_PMGT_SAGA.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_PMGT_SAGA.append(acc)
    para_ave_iter.append(para_ave)
    loss.append(prd.call_loss_dec(workerPara, PMGT_SAGA_Config['ReliableSet']))

    slots = np.array([np.zeros((prd.data_distr[i], prd.n0, prd.n1)) for i in range( prd.m )])
    # This variable ( gradient table ) is expensive in memory since its dimension features n * q_i * p = 智能体的个数 * 本地样本个数 * 单个样本维度,
    # initialization:
    for id in range( prd.m ):
        for j in range(prd.data_distr[id]):
            slots[id][j] = prd.localgrad( workerPara, id, j )  # gradient table
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])
    StoGrad1 = np.zeros( (prd.m, prd.n0, prd.n1) )
    sum_grad = np.zeros( (prd.m, prd.n0, prd.n1) )
    SAGA = np.zeros( (prd.m, prd.n0, prd.n1) )
    SAGA_last = np.zeros( (prd.m, prd.n0, prd.n1) )
    for id in range(prd.m):
        StoGrad1[id] = prd.localgrad(PMGT_SAGA_Config['Initialization'], id, sample_vec[id])
        slots[id][sample_vec[id]] = StoGrad1[id]  # update of the gradient table position/slot
        sum_grad[id] = np.sum(slots[id], axis = 0)
        StoGrad_Diff = StoGrad1[id] - slots[id][sample_vec[id]]  # prd.localgrad() is the stochastic gradient 1
        SAGA_last[id] = StoGrad_Diff + sum_grad[id] / prd.data_distr[id]
    tracker = cp.deepcopy(SAGA_last)
    aggregation_tracker = np.zeros( (prd.m, prd.n0, prd.n1) )
    aggregation_workerPara = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range(int( PMGT_SAGA_Config['Iterations'] ))):
        workerPara_temp = cp.deepcopy(workerPara)
        tracker_temp = cp.deepcopy(tracker)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range( prd.m )])
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in PMGT_SAGA_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            # local stochastic gradient estimation
            StoGrad1[id] = prd.localgrad( workerPara_temp, id, sample_vec[id] )
            StoGrad_Diff = StoGrad1[id] - slots[id][sample_vec[id]]  # prd.localgrad() is the stochastic gradient 1
            SAGA[id] = StoGrad_Diff + sum_grad[id]/prd.data_distr[id]
            sum_grad[id] += StoGrad_Diff
            slots[id][sample_vec[id]] = StoGrad1[id]  # update of the gradient table position/slot
        tracker = tracker_temp + SAGA - SAGA_last
        for id in PMGT_SAGA_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbors_id, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, PMGT_SAGA_Config['ByzantineNetwork'], PMGT_SAGA_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # tracker attack
                tracker, last_str = attack(id, tracker, PMGT_SAGA_Config['ByzantineSet'], PMGT_SAGA_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable,\
                                           PMGT_SAGA_Config['Mixing parameter'], PMGT_SAGA_Config['WeightMatrix'])
            else:
                last_str = '-No-Attacks-'
            aggregation_tracker[id] = prd.fast_mix_Byzan(id, tracker, PMGT_SAGA_Config['Multi communications'], PMGT_SAGA_Config['WeightMatrix'],\
                                                         PMGT_SAGA_Config['Mixing parameter'])
        # proximal-gradient and gradient-descent step
        para_temp = prd.prox_l1( workerPara_temp - PMGT_SAGA_Config['StepSize'] * aggregation_tracker, (prd.reg_l1/prd.m) * PMGT_SAGA_Config['StepSize'] )
        for id in PMGT_SAGA_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbors_id, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, PMGT_SAGA_Config['ByzantineNetwork'], PMGT_SAGA_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # workerPara attack
                para_temp, last_str = attack(id, para_temp, PMGT_SAGA_Config['ByzantineSet'], PMGT_SAGA_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable,\
                                           PMGT_SAGA_Config['Mixing parameter'], PMGT_SAGA_Config['WeightMatrix'])
            else:
                last_str = '-No-Attacks-'
            aggregation_workerPara[id] = prd.fast_mix_Byzan(id, para_temp, PMGT_SAGA_Config['Multi communications'], PMGT_SAGA_Config['WeightMatrix'], PMGT_SAGA_Config['Mixing parameter'])
        workerPara = cp.deepcopy(aggregation_workerPara)
        SAGA_last = cp.deepcopy(SAGA)
        if k % prd.b == 0:
            end = time.perf_counter()
            time_axis_PMGT_SAGA.append(end - start)
            para_epoch.append(cp.deepcopy(workerPara))
            res = get_residual(PMGT_SAGA_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_PMGT_SAGA.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in PMGT_SAGA_Config['ReliableSet']:
                para_ave += workerPara[id] / PMGT_SAGA_Config['ReliableSize']
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_PMGT_SAGA.append(acc)
            var = get_consensus_error(PMGT_SAGA_Config['ReliableSet'], workerPara, para_ave)
            var_PMGT_SAGA.append(var)
            if k % 1 == 0:
            # if k % (int(PMGT_SAGA_Config['Iterations'] / 10)) == 0:
                print('PMGT-SAGA of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor('PMGT-SAGA', k, PMGT_SAGA_Config['Iterations'])
            para_ave_iter.append(para_ave)
            loss.append(prd.call_loss_dec(workerPara, PMGT_SAGA_Config['ReliableSet']))
    # Save the experiment results
    np.savetxt('results/txtdata/' + prd.setting + '-res-Prox-DBRO-SAGA' + last_str + str(PMGT_SAGA_Config['ByzantineSize'] / prd.m) + '.txt', res_PMGT_SAGA)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-DBRO-SAGA' + last_str + str(PMGT_SAGA_Config['ByzantineSize'] / prd.m) + '.txt', loss_PMGT_SAGA)
    np.savetxt('results/txtdata/' + prd.setting + '-acc-PMGT-LSVRG' + last_str + str(PMGT_SAGA_Config['ByzantineSize'] / prd.m) + '.txt', acc_PMGT_SAGA)
    np.savetxt('results/txtdata/' + prd.setting + '-consensus-error-PMGT-LSVRG' + last_str + str(PMGT_SAGA_Config['ByzantineSize'] / prd.m) + '.txt', var_PMGT_SAGA)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_PMGT_SAGA[-1], acc_PMGT_SAGA[-1], var_PMGT_SAGA[-1]))
    print('StepSize of PMGT-SAGA =', PMGT_SAGA_Config['StepSize'])
    print('Loss of PMGT-SAGA =', loss[-1])
    time_axis_PMGT_SAGA.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_PMGT_SAGA), time_axis_PMGT_SAGA[-1]))
    return para_epoch, para_ave_iter, time_axis_PMGT_SAGA, loss, last_str


def PMGT_LSVRG(prd, PMGT_LSVRG_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_PMGT_LSVRG = []
    res_PMGT_LSVRG = []
    count_trigger = 0
    loss_PMGT_LSVRG = []
    var_PMGT_LSVRG = []
    acc_PMGT_LSVRG = []  # 构建List类型空列表
    para_ave_iter = []
    loss = []
    para_epoch = [ cp.deepcopy( PMGT_LSVRG_Config['Initialization'] ) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in PMGT_LSVRG_Config['ReliableSet']:
        para_ave += para[id] / PMGT_LSVRG_Config['ReliableSize']
    var = get_consensus_error(PMGT_LSVRG_Config['ReliableSet'], para, para_ave)
    var_PMGT_LSVRG.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_PMGT_LSVRG.append(acc)
    para_ave_iter.append(para_ave)
    loss.append(prd.call_loss_dec(para, PMGT_LSVRG_Config['ReliableSet']))
    # if warmup > 0:
    #     warm = DSGD(prd, B1, learning_rate, warmup * prd.b, theta_0)
    #     for _ in warm[1:]:
    #         theta_epoch.append( _ )
    workerPara = cp.deepcopy( para_epoch[-1] )
    w = workerPara
    sample_vec = np.array([np.random.choice( prd.data_distr[i] ) for i in range( prd.m )])
    StoGrad1 = np.zeros( (prd.m, prd.n0, prd.n1) )
    StoGrad2 = np.zeros( (prd.m, prd.n0, prd.n1) )
    for id in range( prd.m ):
        StoGrad1[id] = prd.localgrad( workerPara, id, sample_vec[id] )
        StoGrad2[id] = prd.localgrad( w, id, sample_vec[id] )  # computation of stochastic gradient 2
    network_grad = prd.networkgrad( workerPara )  # local batch gradient
    LSVRG_last = StoGrad1 - StoGrad2 + network_grad
    tracker = cp.deepcopy(LSVRG_last)
    LSVRG = np.zeros( (prd.m, prd.n0, prd.n1) )
    k = 0
    Epoch_observation = 0
    aggregation_tracker = np.zeros( (prd.m, prd.n0, prd.n1) )
    aggregation_workerPara = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    while Epoch_observation < PMGT_LSVRG_Config['Iterations']:
        workerPara_temp = cp.deepcopy( workerPara )
        tracker_temp = cp.deepcopy(tracker)
        sample_vec = np.array([np.random.choice( prd.data_distr[id]) for id in range( prd.m )] )
        trigger = 0
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in PMGT_LSVRG_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            # local stochastic gradient estimation
            StoGrad1[id] = prd.localgrad( workerPara_temp, id, sample_vec[id] )  # computation of stochastic gradient 1
            StoGrad2[id] = prd.localgrad( w, id, sample_vec[id] )  # computation of stochastic gradient 2
            LSVRG[id] = StoGrad1[id] - StoGrad2[id] + network_grad[id]
            if np.random.random() < PMGT_LSVRG_Config['Triggered Probability']:              # uncoordinated triggered probabilities
                w[id] = workerPara_temp[id]
                network_grad[id] = prd.localgrad( workerPara_temp, id )  # compute the local full gradients
                count_trigger += 1
                trigger = 1
            else:
                pass
        tracker = tracker_temp + LSVRG - LSVRG_last
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in PMGT_LSVRG_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbors_id, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, PMGT_LSVRG_Config['ByzantineNetwork'], PMGT_LSVRG_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # tracker attack
                tracker, last_str = attack( id, tracker, PMGT_LSVRG_Config['ByzantineSet'], PMGT_LSVRG_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable,\
                                           PMGT_LSVRG_Config['Mixing parameter'], PMGT_LSVRG_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            aggregation_tracker[id] = prd.fast_mix_Byzan( id, tracker, PMGT_LSVRG_Config['Multi communications'], PMGT_LSVRG_Config['WeightMatrix'],\
                                                         PMGT_LSVRG_Config['Mixing parameter'] )
        # proximal-gradient and gradient-descent step
        para_temp = prd.prox_l1( workerPara_temp - PMGT_LSVRG_Config['StepSize'] * aggregation_tracker, ( prd.reg_l1/prd.m ) * PMGT_LSVRG_Config['StepSize'] )
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in PMGT_LSVRG_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbors_id, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, PMGT_LSVRG_Config['ByzantineNetwork'], PMGT_LSVRG_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # workerPara attack
                para_temp, last_str = attack( id, para_temp, PMGT_LSVRG_Config['ByzantineSet'], PMGT_LSVRG_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable,\
                                              PMGT_LSVRG_Config['Mixing parameter'], PMGT_LSVRG_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            aggregation_workerPara[id] = prd.fast_mix_Byzan(id, para_temp, PMGT_LSVRG_Config['Multi communications'], PMGT_LSVRG_Config['WeightMatrix'],\
                                                            PMGT_LSVRG_Config['Mixing parameter'] )
        workerPara = cp.deepcopy(aggregation_workerPara)
        LSVRG_last = cp.deepcopy(LSVRG)
        if k % (prd.b / 2) == 0 or trigger == 1:
            end = time.perf_counter()
            para_epoch.append(cp.deepcopy(workerPara))
            res = get_residual(PMGT_LSVRG_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_PMGT_LSVRG.append(res)
            time_axis_PMGT_LSVRG.append(end - start)
            Epoch_observation += 1
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in PMGT_LSVRG_Config['ReliableSet']:
                para_ave += workerPara[id] / PMGT_LSVRG_Config['ReliableSize']
            loss = prd.call_loss_dec(workerPara, PMGT_LSVRG_Config['ReliableSet'])
            loss_PMGT_LSVRG.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_PMGT_LSVRG.append(acc)
            var = get_consensus_error(PMGT_LSVRG_Config['ReliableSet'], workerPara, para_ave)
            var_PMGT_LSVRG.append(var)
            # if Epoch_observation % ( int(PMGT_LSVRG_Config['Iterations']/10) ) == 0:
            if Epoch_observation % 1 == 0:
                print('PMGT-LSVRG of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor('PMGT-LSVRG', Epoch_observation, PMGT_LSVRG_Config['Iterations'])
            para_ave_iter.append(para_ave)
            loss.append(prd.call_loss_dec(workerPara, PMGT_LSVRG_Config['ReliableSet']))
        k += 1
    # Save the experiment results
    np.savetxt('results/txtdata/' + prd.setting + '-res-Prox-DBRO-SAGA' + last_str + str( PMGT_LSVRG_Config['ByzantineSize'] / prd.m) + '.txt', res_PMGT_LSVRG )
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-DBRO-SAGA' + last_str + str( PMGT_LSVRG_Config['ByzantineSize'] / prd.m) + '.txt', loss_PMGT_LSVRG )
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-PMGT-LSVRG' + last_str + str(PMGT_LSVRG_Config['ByzantineSize'] / prd.m) + '.txt', acc_PMGT_LSVRG)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-PMGT-LSVRG' + last_str + str(PMGT_LSVRG_Config['ByzantineSize'] / prd.m) + '.txt', var_PMGT_LSVRG)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_PMGT_LSVRG[-1], acc_PMGT_LSVRG[-1], var_PMGT_LSVRG[-1]))
    print("number of trigger =", count_trigger)
    # print("the final iteration:", k)
    print("the triggered probability =", PMGT_LSVRG_Config['Triggered Probability'])
    print('StepSize of PMGT-LSVRG =', PMGT_LSVRG_Config['StepSize'])
    print('Loss of PMGT-LSVRG =', loss[-1])
    time_axis_PMGT_LSVRG.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_PMGT_LSVRG), time_axis_PMGT_LSVRG[-1]))
    # print('selected ID:', Prox_DBRO_LSVRG_Config['selected id of testing agent'])
    # print('optimal grad:', LSVRG[select])
    return para_epoch, para_ave_iter, time_axis_PMGT_LSVRG, loss, last_str