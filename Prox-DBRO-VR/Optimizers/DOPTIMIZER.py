################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################
import numpy as np
import copy as cp
import utilities as ut
from tqdm import tqdm
from analysis import get_residual, get_accuracy, get_consensus_error
import StepSize
import time
import torch

"""
optConfig = {
    'Iterations': int( 2*N/n ),
    'NodeSize': int( n ),
    'ByzantineSize': Byzantine_nodes,
    'ReliableSize': int( n - Byzantine_nodes ),
    'StepSize': 0,
    'ByzantineSet': Byzantine,
    'ReliableSet': Reliable,
    'ByzantineNetwork': G,
    'ReliableNetwork': H,
    'AdjacentMatrix': Adj,
    'WeightMatrix': M,
    'Initialization': np.random.random( (n, p) )
    'Triggered Probability':  int( lr_0.n/lr_0.N/2 )
}
"""

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


def Prox_DBRO_SAGA(prd, Prox_DBRO_SAGA_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_Prox_DBRO_SAGA = []
    res_Prox_DBRO_SAGA = []
    loss_Prox_DBRO_SAGA = []
    var_Prox_DBRO_SAGA = []   # 构建List类型空列表
    acc_Prox_DBRO_SAGA = []
    para_ave_list = []
    workerPara = Prox_DBRO_SAGA_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_DBRO_SAGA_Config['ReliableSet']:
        para_ave += para[id] / Prox_DBRO_SAGA_Config['ReliableSize']
    var = get_consensus_error( Prox_DBRO_SAGA_Config['ReliableSet'], para, para_ave )
    var_Prox_DBRO_SAGA.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_DBRO_SAGA.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_DBRO_SAGA_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_DBRO_SAGA_Config['ReliableSet'])
    loss_Prox_DBRO_SAGA.append(loss)
    # initialization:
    StoGrad1 = np.zeros( (prd.m, prd.n0, prd.n1) )
    slots = np.array([np.zeros((prd.data_distr[i], prd.n0, prd.n1)) for i in range( prd.m )])
    # This variable ( gradient table ) is expensive in memory
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])
    sum_grad = np.zeros( (prd.m, prd.n0, prd.n1) )
    for id in range(prd.m):
        slots[id][sample_vec[id]] = prd.localgrad(workerPara, id, sample_vec[id])
        # update of the gradient table position/slot
        sum_grad[id] = np.sum(slots[id], axis = 0)
    SAGA = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    step_size = Prox_DBRO_SAGA_Config['StepSize']  # StepSize.get_decaying_step(10, k)
    for k in tqdm(range(Prox_DBRO_SAGA_Config['Iterations'])):
        # constant step-size
        # step_size = Prox_DBRO_LSVRG_Config['StepSize']
        # decaying step-size
        step_size = StepSize.get_decaying_step(1, k)  # decaying step-size

        workerPara_temp = cp.deepcopy(workerPara)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range( prd.m )])
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in Prox_DBRO_SAGA_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors( prd, id,
            Prox_DBRO_SAGA_Config['ByzantineNetwork'], Prox_DBRO_SAGA_Config['ByzantineSet'] )
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_DBRO_SAGA_Config['ByzantineSet'], Prox_DBRO_SAGA_Config['ReliableSet'], neighbors_Byzantine,\
                                                    neighbors_reliable )
            else:
                last_str = '-No-Attacks-'
            # local stochastic gradient estimation
            StoGrad1[id] = prd.localgrad(workerPara_temp, id, sample_vec[id])
            StoGrad_Diff = StoGrad1[id] - slots[id][sample_vec[id]]
            SAGA[id] = StoGrad_Diff + sum_grad[id]/prd.data_distr[id]
            slots[id][sample_vec[id]] = StoGrad1[id]  # update of the gradient table position/slot
            sum_grad[id] += StoGrad_Diff
            # aggregation
            penalty = np.zeros( (prd.n0, prd.n1) )
            for jd in neighbor_list:
                if last_str == '-ZSA-':
                    # max norm
                    penalty += prd.cal_max_norm_grad( workerPara[id] - workerPara_temp[jd] )
                    pn_setting = 'LMax-'
                elif last_str == '-GA-':
                    # L2 norm
                    tmp = np.linalg.norm( workerPara[id] - workerPara_temp[jd], ord=2 )
                    if tmp == 0:
                        tmp = 1e-7
                    penalty += ( workerPara[id] - workerPara_temp[jd] ) / tmp  # gradient of L2 norm
                    pn_setting = 'L2-'
                elif last_str == '-SVA-':
                    # L1 norm
                    penalty += np.sign(workerPara[id] - workerPara_temp[jd])  # gradient of L1 norm
                    pn_setting = 'L1-'
                elif last_str == '-SFA-':
                    # L2 norm
                    tmp = np.linalg.norm( workerPara[id] - workerPara_temp[jd], ord=2 )
                    if tmp == 0:
                        tmp = 1e-7
                    penalty += ( workerPara[id] - workerPara_temp[jd] ) / tmp  # gradient of L2 norm
                    pn_setting = 'L2-'
                else:
                    print("no attacks happen!")
            aggregate_gradient = Prox_DBRO_SAGA_Config['PenaltyPara'][id] * penalty
            # para[id] = para[id] - step_size * (SAGA[id] + aggregate_gradient)
            para_temp = workerPara[id] - step_size * (SAGA[id] + aggregate_gradient)
            workerPara[id] = prd.prox_l1( para_temp, ( prd.reg_l1 ) * step_size )
            # proximal-gradient descent
        if k % prd.b == 0:   # Prox-DBRO-SAGA requires prd.b iterations to finish an epoch.
            end = time.perf_counter()
            para_epoch.append( cp.deepcopy(workerPara) )
            time_axis_Prox_DBRO_SAGA.append( end - start )
            res = get_residual(Prox_DBRO_SAGA_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_DBRO_SAGA.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_DBRO_SAGA_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_DBRO_SAGA_Config['ReliableSize']
            para_ave_list.append(para_ave)
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_DBRO_SAGA_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_DBRO_SAGA_Config['ReliableSet'], ave = None)
            loss_Prox_DBRO_SAGA.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_DBRO_SAGA.append(acc)
            var = get_consensus_error( Prox_DBRO_SAGA_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_DBRO_SAGA.append(var)
            if k % ( prd.b ) == 0:
            # if k % ( int(Prox_DBRO_SAGA_Config['Iterations'] / 10) ) == 0:
                print('Prox-DBRO-SAGA of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-DBRO-SAGA', k, Prox_DBRO_SAGA_Config['Iterations'] )
    # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-Prox-DBRO-SAGA' + last_str + str(Prox_DBRO_SAGA_Config['ByzantineSize']/prd.m) + '.txt', res_Prox_DBRO_SAGA)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-DBRO-SAGA' + last_str + str(Prox_DBRO_SAGA_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_DBRO_SAGA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-DBRO-SAGA' + last_str + str(Prox_DBRO_SAGA_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_DBRO_SAGA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-Prox-DBRO-SAGA' + last_str + str( Prox_DBRO_SAGA_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_DBRO_SAGA)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_DBRO_SAGA[-1], acc_Prox_DBRO_SAGA[-1], var_Prox_DBRO_SAGA[-1]))
    print('StepSize of Prox-DBRO-SAGA:', step_size )
    print('Loss of Prox-DBRO-SAGA =', loss_Prox_DBRO_SAGA[-1])
    print('penalty parameters of Prox-DBRO-SAGA:', Prox_DBRO_SAGA_Config['PenaltyPara'])
    time_axis_Prox_DBRO_SAGA.insert(0, 0) # for a same starting point
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_DBRO_SAGA), time_axis_Prox_DBRO_SAGA[-1]))
    print("epochs =", Prox_DBRO_SAGA_Config['Iterations'])
    return para_epoch, para_ave_list, time_axis_Prox_DBRO_SAGA, res_Prox_DBRO_SAGA, loss_Prox_DBRO_SAGA, last_str, pn_setting


def Prox_DBRO_LSVRG(prd, Prox_DBRO_LSVRG_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_Prox_DBRO_LSVRG = []
    count_trigger = 0
    res_Prox_DBRO_LSVRG = []
    loss_Prox_DBRO_LSVRG = []
    acc_Prox_DBRO_LSVRG = []  # 构建List类型空列表
    var_Prox_DBRO_LSVRG = []
    para_ave_list = []
    workerPara = Prox_DBRO_LSVRG_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # if warmup > 0:
    #     warm = DSGD(prd, B1, step_size, warmup * prd.b, theta_0)
    #     for _ in warm[1:]:
    #         theta_epoch.append( _ )

    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_DBRO_LSVRG_Config['ReliableSet']:
        para_ave += para[id] / Prox_DBRO_LSVRG_Config['ReliableSize']
    var = get_consensus_error(Prox_DBRO_LSVRG_Config['ReliableSet'], para, para_ave )
    var_Prox_DBRO_LSVRG.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_DBRO_LSVRG.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_DBRO_LSVRG_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_DBRO_LSVRG_Config['ReliableSet'])
    loss_Prox_DBRO_LSVRG.append(loss)
    w = cp.deepcopy( workerPara )
    StoGrad1 = np.zeros( (prd.m, prd.n0, prd.n1) )
    StoGrad2 = np.zeros( (prd.m, prd.n0, prd.n1) )
    network_grad = prd.networkgrad( workerPara )  # local batch gradient
    LSVRG = np.zeros( (prd.m, prd.n0, prd.n1) )
    k = 0
    Epoch_observation = 0
    start = time.perf_counter()
    # for k in tqdm(range(int( K ))):
    while Epoch_observation < Prox_DBRO_LSVRG_Config['Iterations']:
        # constant step-size
        # step_size = Prox_DBRO_LSVRG_Config['StepSize']

        # decaying step-size
        step_size = StepSize.get_decaying_step(1, k)  # decaying step-size
        workerPara_temp = cp.deepcopy(workerPara)
        sample_vec = np.array([np.random.choice(prd.data_distr[id]) for id in range( prd.m )])
        trigger = 0
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in Prox_DBRO_LSVRG_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors( prd, id, Prox_DBRO_LSVRG_Config['ByzantineNetwork'], Prox_DBRO_LSVRG_Config['ByzantineSet'] )
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_DBRO_LSVRG_Config['ByzantineSet'], Prox_DBRO_LSVRG_Config['ReliableSet'], neighbors_Byzantine, \
                                                    neighbors_reliable )
            else:
                last_str = '-No-Attacks-'
            # local stochastic gradient estimation
            StoGrad1[id] = prd.localgrad( workerPara_temp, id, sample_vec[id] )  # computation of stochastic gradient 1
            StoGrad2[id] = prd.localgrad( w, id, sample_vec[id] )  # computation of stochastic gradient 2
            LSVRG[id] = StoGrad1[id] - StoGrad2[id] + network_grad[id]
            if np.random.random() <= Prox_DBRO_LSVRG_Config['Triggered Probability'][id]:
            # uncoordinated triggered probabilities to trigger the computation of local-batch gradient
                w[id] = workerPara_temp[id]
                network_grad[id] = prd.localgrad( workerPara_temp, id )  # compute the local full gradients
                count_trigger += 1
                trigger = 1
            else:
                pass
            #   network_grad[id] = network_grad[id]
            # aggregation
            penalty = np.zeros( (prd.n0, prd.n1) )
            for jd in neighbor_list:
                if last_str == '-ZSA-':
                    # max norm
                    penalty += prd.cal_max_norm_grad( workerPara[id] - workerPara_temp[jd] )
                    pn_setting = 'LMax-'

                elif last_str == '-GA-':
                    # L2 norm
                    tmp = np.linalg.norm( workerPara[id] - workerPara_temp[jd], ord=2 )
                    if tmp == 0:
                        tmp = 1e-7
                    penalty += ( workerPara[id] - workerPara_temp[jd] ) / tmp  # gradient of L2 norm
                    pn_setting = 'L2-'

                elif last_str == '-SVA-':
                    # L1 norm
                    penalty += np.sign(workerPara[id] - workerPara_temp[jd])  # gradient of L1 norm
                    pn_setting = 'L1-'

                elif last_str == '-SFA-':
                    # L2 norm
                    tmp = np.linalg.norm( workerPara[id] - workerPara_temp[jd], ord=2 )
                    if tmp == 0:
                        tmp = 1e-7
                    penalty += ( workerPara[id] - workerPara_temp[jd] ) / tmp  # gradient of L2 norm
                    pn_setting = 'L2-'

                else:
                    print("no attacks happen!")
            aggregate_gradient = Prox_DBRO_LSVRG_Config['PenaltyPara'][id] * penalty
            para_temp = workerPara[id] - step_size * ( LSVRG[id] + aggregate_gradient )
            workerPara[id] = prd.prox_l1( para_temp, ( prd.reg_l1 ) * step_size )  # proximal-gradient descent
        if k % ( prd.b/2 ) == 0 or trigger == 1:
            end = time.perf_counter()
            para_epoch.append( cp.deepcopy(workerPara) )
            time_axis_Prox_DBRO_LSVRG.append(end-start)
            Epoch_observation += 1
            res = get_residual(Prox_DBRO_LSVRG_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_DBRO_LSVRG.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_DBRO_LSVRG_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_DBRO_LSVRG_Config['ReliableSize']
            para_ave_list.append(para_ave)
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_DBRO_LSVRG_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_DBRO_LSVRG_Config['ReliableSet'], ave = None )
            loss_Prox_DBRO_LSVRG.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_DBRO_LSVRG.append(acc)
            var = get_consensus_error( Prox_DBRO_LSVRG_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_DBRO_LSVRG.append(var)
            # if Epoch_observation % ( int(Prox_DBRO_LSVRG_Config['Iterations']/10) ) == 0:
            if Epoch_observation % 1 == 0:
                print('Prox-DBRO-LSVRG of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-DBRO-LSVRG', Epoch_observation, Prox_DBRO_LSVRG_Config['Iterations'] )
        k += 1
    # Save the experiment results
    np.savetxt('results/txtdata/' + prd.setting + '-res-Prox-DBRO-LSVRG' + last_str + str( Prox_DBRO_LSVRG_Config['ByzantineSize']/prd.m) + '.txt', res_Prox_DBRO_LSVRG)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-DBRO-LSVRG' + last_str + str(Prox_DBRO_LSVRG_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_DBRO_LSVRG)
    np.savetxt('results/txtdata/' + prd.setting + '-acc-Prox-DBRO-LSVRG' + last_str + str( Prox_DBRO_LSVRG_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_DBRO_LSVRG)
    np.savetxt('results/txtdata/' + prd.setting + '-consensus-error-Prox-DBRO-LSVRG' + last_str + str( Prox_DBRO_LSVRG_Config['ByzantineSize']/prd.m) + '.txt', var_Prox_DBRO_LSVRG)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_DBRO_LSVRG[-1], acc_Prox_DBRO_LSVRG[-1], var_Prox_DBRO_LSVRG[-1]))
    print('StepSize of Prox-DBRO-LSVRG:', step_size)
    # print("initial value of decaying step-size =", initial_stepsize)
    print("number of trigger =", count_trigger)
    # print("the final iteration:", k)
    print("the triggered probability:", Prox_DBRO_LSVRG_Config['Triggered Probability'])
    print('Loss of Prox-DBRO-LSVRG =', loss_Prox_DBRO_LSVRG[-1])
    print('penalty parameters of Prox-DBRO-LSVRG:', Prox_DBRO_LSVRG_Config['PenaltyPara'])
    time_axis_Prox_DBRO_LSVRG.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_DBRO_LSVRG), time_axis_Prox_DBRO_LSVRG[-1]))
    print("epochs =", Prox_DBRO_LSVRG_Config['Iterations'])
    return para_epoch, para_ave_list, time_axis_Prox_DBRO_LSVRG, res_Prox_DBRO_LSVRG, loss_Prox_DBRO_LSVRG, last_str, pn_setting

def Prox_Peng(prd, Prox_Peng_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_Prox_Peng = []
    res_Prox_Peng = []
    loss_Prox_Peng = []
    acc_Prox_Peng = []  # 构建List类型空列表
    var_Prox_Peng = []
    para_ave_list = []
    workerPara = Prox_Peng_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # if warmup > 0:
    #     warm = DSGD(prd, B1, step_size, warmup * prd.b, theta_0)
    #     for _ in warm[1:]:
    #         theta_epoch.append( _ )
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_Peng_Config['ReliableSet']:
        para_ave += para[id] / Prox_Peng_Config['ReliableSize']
    var = get_consensus_error(Prox_Peng_Config['ReliableSet'], para, para_ave )
    var_Prox_Peng.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_Peng.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_Peng_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_Peng_Config['ReliableSet'])
    loss_Prox_Peng.append(loss)
    StoGrad = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range(Prox_Peng_Config['Iterations'])):
        # step_size = Prox_DBRO_SAGA_Config['StepSize']  # StepSize.get_decaying_step(10, k)
        step_size = StepSize.get_decaying_step(1.2, k)
        workerPara_temp = cp.deepcopy(workerPara)
        # for id in range( prd.m ): # this type of range serves for sign-flipping attacks with Byzantine agents engaging the update
        for id in Prox_Peng_Config['ReliableSet']: # this type of loop saves the computational time on one machine for decentralized methods
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors( prd, id, Prox_Peng_Config['ByzantineNetwork'], Prox_Peng_Config['ByzantineSet'] )
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_Peng_Config['ByzantineSet'], Prox_Peng_Config['ReliableSet'], neighbors_Byzantine, \
                                                    neighbors_reliable )
            else:
                last_str = '-No-Attacks-'
            # local stochastic gradient estimation
            StoGrad[id] = prd.localgrad( workerPara_temp, id, j=None, BatchSize=Prox_Peng_Config['BatchSize'] )  # computation of stochastic gradient 1
            penalty = np.zeros( (prd.n0, prd.n1) )
            for jd in neighbor_list:
                if last_str == '-ZSA-':
                    # max norm
                    penalty += prd.cal_max_norm_grad(workerPara[id] - workerPara_temp[jd])
                    pn_setting = 'LMax-'
                elif last_str == '-GA-':
                    # L2 norm
                    tmp = np.linalg.norm(workerPara[id] - workerPara_temp[jd], ord=2)
                    if tmp == 0:
                        tmp = 1e-7
                    penalty += (workerPara[id] - workerPara_temp[jd]) / tmp  # gradient of L2 norm
                    pn_setting = 'L2-'
                elif last_str == '-SVA-':
                    # L1 norm
                    penalty += np.sign(workerPara[id] - workerPara_temp[jd])  # gradient of L1 norm
                    pn_setting = 'L1-'
                else:
                    print("no attacks happen!")
            aggregate_gradient = Prox_Peng_Config['PenaltyPara'] * penalty
            para_temp = workerPara[id] - step_size * ( StoGrad[id] + aggregate_gradient )
            workerPara[id] = prd.prox_l1( para_temp, ( prd.reg_l1 ) * step_size )  # proximal-gradient descent
        if k % ( prd.b/Prox_Peng_Config['BatchSize'] ) == 0:
            end = time.perf_counter()
            para_epoch.append( cp.deepcopy(workerPara) )
            time_axis_Prox_Peng.append( end - start )
            res = get_residual(Prox_Peng_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_Peng.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_Peng_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_Peng_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_Peng_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_Peng_Config['ReliableSet'])
            loss_Prox_Peng.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_Peng.append(acc)
            var = get_consensus_error( Prox_Peng_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_Peng.append(var)
            # if Epoch_observation % ( int(Prox_Peng_Config['Iterations']/10) ) == 0:
            if k % ( int(Prox_Peng_Config['Iterations'] / 10) ) == 0:
                print('Prox-Peng of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-Peng', k, Prox_Peng_Config['Iterations'] )
        k += 1
    # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-Prox-Peng' + last_str + str( Prox_Peng_Config['ByzantineSize']/prd.m ) + '.txt', res_Prox_Peng)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-Peng' + last_str + str(Prox_Peng_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_Peng)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-Peng' + last_str + str( Prox_Peng_Config['ByzantineSize']/prd.m ) + '.txt', acc_Prox_Peng)
    np.savetxt('results/txtdata/' + prd.setting + '-consensus-error-Prox-Peng' + last_str + str( Prox_Peng_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_Peng)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_Peng[-1], acc_Prox_Peng[-1], var_Prox_Peng[-1]))
    print('StepSize of Prox-Peng:', step_size)
    # print("the final iteration:", k)
    print('Loss of Prox-Peng =', loss_Prox_Peng[-1])
    print('penalty parameters of Prox-Peng:', Prox_Peng_Config['PenaltyPara'])
    time_axis_Prox_Peng.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_Peng), time_axis_Prox_Peng[-1]))
    print('BatchSize =', Prox_Peng_Config['BatchSize'])
    # print('selected ID:', Prox_Peng_Config['selected id of testing agent'])
    return para_epoch, para_ave_list, time_axis_Prox_Peng, res_Prox_Peng, loss_Prox_Peng, last_str, pn_setting


def Prox_BRIDGE_T(prd, Prox_BRIDGE_T_Config, image_test, label_test, attack, dsgd_optimal):
    # coordinate-wise trimmed-mean screening
    time_axis_Prox_BRIDGE_T = []
    res_Prox_BRIDGE_T = []
    loss_Prox_BRIDGE_T = []
    acc_Prox_BRIDGE_T = []  # 构建List类型空列表
    var_Prox_BRIDGE_T = []
    para_ave_list = []
    workerPara = Prox_BRIDGE_T_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_BRIDGE_T_Config['ReliableSet']:
        para_ave += para[id] / Prox_BRIDGE_T_Config['ReliableSize']
    var = get_consensus_error( Prox_BRIDGE_T_Config['ReliableSet'], para, para_ave )
    var_Prox_BRIDGE_T.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_BRIDGE_T.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_T_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_T_Config['ReliableSet'])
    loss_Prox_BRIDGE_T.append(loss)
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range( Prox_BRIDGE_T_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy( workerPara )
        for id in range(prd.m):
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, Prox_BRIDGE_T_Config['ByzantineNetwork'], Prox_BRIDGE_T_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_BRIDGE_T_Config['ByzantineSet'],
                Prox_BRIDGE_T_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, Prox_BRIDGE_T_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            neighbors_para = []
            for jd in neighbor_list:
                neighbors_para.append( workerPara_temp[jd] )
            number_neighbors = len(neighbors_para) # 领居的数量
            neighbors_para = np.array(neighbors_para).reshape(number_neighbors, -1)
            first = np.sort(neighbors_para.T)  # 对转置后的数组，按每行对其coordinate进行从小（左）到大（右）排序
            b = int( Prox_BRIDGE_T_Config['ByzantineSize'] )  # 仿真中的clairvoyant信息
            # b = int( b/2 ) # only for 20 Byzantine agents out of 60 total agents
            second = first[:, b: number_neighbors - b]
            # 取BRIDGE-T中节点信息在安全范围内的数据, 剔除2*b个数据, requirement: neighbors >= 2 * b + 1
            aggregation_id = np.sum(second, axis=1) / (number_neighbors - 2 * b) # only for 20 Byzantine agents out of 60 total agents
            aggregation[id] = aggregation_id.reshape(prd.n0, prd.n1)
            # axis = 1 按行求和， third指BRIDGE-T中的[y_j(t)]_k
        # vector of local batch gradients
        grad = prd.networkgrad( workerPara_temp )
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( aggregation - Prox_BRIDGE_T_Config['StepSize'] * grad,  prd.reg_l1 * Prox_BRIDGE_T_Config['StepSize'] )
        if k % 1 == 0:
            end = time.perf_counter()
            time_axis_Prox_BRIDGE_T.append( end - start )
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(Prox_BRIDGE_T_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_BRIDGE_T.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_BRIDGE_T_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_BRIDGE_T_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_T_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_T_Config['ReliableSet'])
            loss_Prox_BRIDGE_T.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_BRIDGE_T.append(acc)
            var = get_consensus_error( Prox_BRIDGE_T_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_BRIDGE_T.append(var)
            # if k % 1 == 0:
            if k % ( int(Prox_BRIDGE_T_Config['Iterations'] / 10) ) == 0:
                print('Prox-BRIDGE-T of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-BRIDGE-T', k, Prox_BRIDGE_T_Config['Iterations'] )
     # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-Prox-BRIDGE-T' + last_str + str(Prox_BRIDGE_T_Config['ByzantineSize']/prd.m) + '.txt', res_Prox_BRIDGE_T)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-BRIDGE-T' + last_str + str(Prox_BRIDGE_T_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_BRIDGE_T)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-BRIDGE-T' + last_str + str(Prox_BRIDGE_T_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_BRIDGE_T)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-Prox-BRIDGE-T' + last_str + str( Prox_BRIDGE_T_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_BRIDGE_T)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_BRIDGE_T[-1], acc_Prox_BRIDGE_T[-1], var_Prox_BRIDGE_T[-1]))
    print('StepSize of Prox-BRIDGE-T:', Prox_BRIDGE_T_Config['StepSize'] )
    print('Loss of Prox-BRIDGE-T =', loss_Prox_BRIDGE_T[-1])
    time_axis_Prox_BRIDGE_T.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_BRIDGE_T), time_axis_Prox_BRIDGE_T[-1]))
    return para_epoch, para_ave_list, time_axis_Prox_BRIDGE_T, res_Prox_BRIDGE_T, loss_Prox_BRIDGE_T, last_str


def Prox_BRIDGE_M(prd, Prox_BRIDGE_M_Config, image_test, label_test, attack, dsgd_optimal):
    # coordinate-wise median screening
    time_axis_Prox_BRIDGE_M = []
    res_Prox_BRIDGE_M = []
    loss_Prox_BRIDGE_M = []
    acc_Prox_BRIDGE_M = []   # 构建List类型空列表
    var_Prox_BRIDGE_M = []
    para_ave_list = []
    workerPara = Prox_BRIDGE_M_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_BRIDGE_M_Config['ReliableSet']:
        para_ave += para[id] / Prox_BRIDGE_M_Config['ReliableSize']
    var = get_consensus_error( Prox_BRIDGE_M_Config['ReliableSet'], para, para_ave )
    var_Prox_BRIDGE_M.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_BRIDGE_M.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_M_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_M_Config['ReliableSet'])
    loss_Prox_BRIDGE_M.append(loss)
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range( Prox_BRIDGE_M_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy(workerPara)
        for id in range(prd.m):
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, Prox_BRIDGE_M_Config['ByzantineNetwork'], Prox_BRIDGE_M_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_BRIDGE_M_Config['ByzantineSet'],
                Prox_BRIDGE_M_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, Prox_BRIDGE_M_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            neighbors_para = []
            for jd in neighbor_list:
                neighbors_para.append(workerPara_temp[jd])
            aggregation[id] = np.median( neighbors_para, axis = 0 )  # 按每行取中位数
        # vector of local batch gradients
        grad = prd.networkgrad( workerPara_temp )
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( aggregation - Prox_BRIDGE_M_Config['StepSize'] * grad, ( prd.reg_l1 ) * Prox_BRIDGE_M_Config['StepSize'] )
        if k % 1 == 0:
            end = time.perf_counter()
            time_axis_Prox_BRIDGE_M.append( end - start )
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(Prox_BRIDGE_M_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_BRIDGE_M.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_BRIDGE_M_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_BRIDGE_M_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_M_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_M_Config['ReliableSet'])
            loss_Prox_BRIDGE_M.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_BRIDGE_M.append(acc)
            var = get_consensus_error( Prox_BRIDGE_M_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_BRIDGE_M.append(var)
            if k % ( int( Prox_BRIDGE_M_Config['Iterations']/10) ) == 0:
                print('Prox-BRIDGE-M of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-BRIDGE-M', k, Prox_BRIDGE_M_Config['Iterations'] )
     # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-Prox-BRIDGE-M' + last_str + str(Prox_BRIDGE_M_Config['ByzantineSize']/prd.m) + '.txt', res_Prox_BRIDGE_M)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-BRIDGE-M' + last_str + str(Prox_BRIDGE_M_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_BRIDGE_M)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-BRIDGE-M' + last_str + str(Prox_BRIDGE_M_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_BRIDGE_M)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-Prox-BRIDGE-M' + last_str + str( Prox_BRIDGE_M_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_BRIDGE_M)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_BRIDGE_M[-1], acc_Prox_BRIDGE_M[-1], var_Prox_BRIDGE_M[-1]))
    print('StepSize of Prox-BRIDGE-M:', Prox_BRIDGE_M_Config['StepSize'] )
    print('Loss of Prox-BRIDGE-M =', loss_Prox_BRIDGE_M[-1])
    time_axis_Prox_BRIDGE_M.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_BRIDGE_M), time_axis_Prox_BRIDGE_M[-1]))
    return para_epoch, para_ave_list, time_axis_Prox_BRIDGE_M, res_Prox_BRIDGE_M, loss_Prox_BRIDGE_M, last_str


def Prox_BRIDGE_K(prd, Prox_BRIDGE_K_Config, image_test, label_test, attack, dsgd_optimal):
    # Krum screening
    time_axis_Prox_BRIDGE_K = []
    res_Prox_BRIDGE_K = []
    loss_Prox_BRIDGE_K = []
    acc_Prox_BRIDGE_K = [] # 构建List类型空列表
    var_Prox_BRIDGE_K = []
    para_ave_list = []
    workerPara = Prox_BRIDGE_K_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_BRIDGE_K_Config['ReliableSet']:
        para_ave += para[id] / Prox_BRIDGE_K_Config['ReliableSize']
    var = get_consensus_error( Prox_BRIDGE_K_Config['ReliableSet'], para, para_ave )
    var_Prox_BRIDGE_K.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_BRIDGE_K.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_K_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_K_Config['ReliableSet'])
    loss_Prox_BRIDGE_K.append(loss)
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range( Prox_BRIDGE_K_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy(workerPara)
        for id in range(prd.m):
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id,\
            Prox_BRIDGE_K_Config['ByzantineNetwork'], Prox_BRIDGE_K_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_BRIDGE_K_Config['ByzantineSet'],
                Prox_BRIDGE_K_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, Prox_BRIDGE_K_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            # for jd in neighbor_list:
            #     neighbors_jd, _, _ = get_neighbors(prd, jd, Prox_BRIDGE_K_Config['ByzantineNetwork'],\
            #                                           Prox_BRIDGE_K_Config['ByzantineSet'])
            score_w = []
            neighborhood_workerPara_id = [workerPara_temp[jd] for jd in neighbor_list]
            # Iterate through all neighbors of the current node
            b = int( Prox_BRIDGE_K_Config['ByzantineSize'] )
            for workerPara_jd in neighborhood_workerPara_id:
                dist_w = [np.linalg.norm(workerPara_hd - workerPara_jd) for workerPara_hd in neighborhood_workerPara_id]
                dist_w = np.sort(dist_w)
                # Sum up closest n-b-2 vectors to g_w and g_b
                score_w.append(np.sum(dist_w[:(len(neighborhood_workerPara_id) - b - 2)]))
                # requirement: neighbors >= b + 2
            ind_w = score_w.index(min(score_w))
            aggregation[id] = cp.deepcopy(neighborhood_workerPara_id[ind_w])
        grad = prd.networkgrad( workerPara_temp )
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( aggregation - Prox_BRIDGE_K_Config['StepSize'] * grad, ( prd.reg_l1/prd.m ) * Prox_BRIDGE_K_Config['StepSize'] )
        if k % 1 == 0:
            end = time.perf_counter()
            time_axis_Prox_BRIDGE_K.append( end - start )
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(Prox_BRIDGE_K_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_BRIDGE_K.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_BRIDGE_K_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_BRIDGE_K_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_K_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_K_Config['ReliableSet'])
            loss_Prox_BRIDGE_K.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_BRIDGE_K.append(acc)
            var = get_consensus_error( Prox_BRIDGE_K_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_BRIDGE_K.append(var)
            if k % ( int( Prox_BRIDGE_K_Config['Iterations']/10) ) == 0:
                print('Prox-BRIDGE-K of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-BRIDGE-K', k, Prox_BRIDGE_K_Config['Iterations'] )
    # Save the experiment results
    np.savetxt('results/txtdata/' + prd.setting + '-res-Prox-BRIDGE-K' + last_str + str(Prox_BRIDGE_K_Config['ByzantineSize'] / prd.m) + '.txt', res_Prox_BRIDGE_K)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-BRIDGE-K' + last_str + str(Prox_BRIDGE_K_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_BRIDGE_K)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-BRIDGE-K' + last_str + str(Prox_BRIDGE_K_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_BRIDGE_K)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-Prox-BRIDGE-K' + last_str + str( Prox_BRIDGE_K_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_BRIDGE_K)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_BRIDGE_K[-1], acc_Prox_BRIDGE_K[-1], var_Prox_BRIDGE_K[-1]))
    print('StepSize of Prox-BRIDGE-K:', Prox_BRIDGE_K_Config['StepSize'] )
    print('Loss of Prox-BRIDGE-K =', loss_Prox_BRIDGE_K[-1])
    time_axis_Prox_BRIDGE_K.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_BRIDGE_K), time_axis_Prox_BRIDGE_K[-1]))
    return para_epoch, para_ave_list, time_axis_Prox_BRIDGE_K, res_Prox_BRIDGE_K, res_Prox_BRIDGE_K, loss_Prox_BRIDGE_K, last_str


def Prox_BRIDGE_B(prd, Prox_BRIDGE_B_Config, image_test, label_test, attack, dsgd_optimal):
    # Bulyan screening
    time_axis_Prox_BRIDGE_B = []
    res_Prox_BRIDGE_B = []
    loss_Prox_BRIDGE_B = []
    acc_Prox_BRIDGE_B = []   # 构建List类型空列表
    var_Prox_BRIDGE_B = []
    para_ave_list = []
    workerPara = Prox_BRIDGE_B_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_BRIDGE_B_Config['ReliableSet']:
        para_ave += para[id] / Prox_BRIDGE_B_Config['ReliableSize']
    var = get_consensus_error(Prox_BRIDGE_B_Config['ReliableSet'], para, para_ave)
    var_Prox_BRIDGE_B.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_BRIDGE_B.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_B_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_B_Config['ReliableSet'])
    loss_Prox_BRIDGE_B.append(loss)
    aggregation = np.zeros((prd.m, prd.n0, prd.n1))
    aggregation_temp = np.zeros( (prd.m, prd.n0 * prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range( Prox_BRIDGE_B_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy(workerPara)
        for id in range(prd.m):
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, Prox_BRIDGE_B_Config['ByzantineNetwork'], Prox_BRIDGE_B_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_BRIDGE_B_Config['ByzantineSet'], Prox_BRIDGE_B_Config['ReliableSet'], neighbors_Byzantine,\
                                                    neighbors_reliable, Prox_BRIDGE_B_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            S_w = []
            b = int( Prox_BRIDGE_B_Config[ 'ByzantineSize' ] / 2 )
            """ Part 1 of Bulyan using Krum screening to screen for W matrix """
            for _ in range( len( neighbor_list ) - 2 * b ):
                # repeating the Krum-based screening ( len( neighbor_list ) − 2*Prox_BRIDGE_B_Config['ByzantineSize'] ) times
                score_w = []
                neighborhood_workerPara_id = [workerPara_temp[jd] for jd in neighbor_list]
                for workerPara_jd in neighborhood_workerPara_id:
                    dist_w = [np.abs(workerPara_hd - workerPara_jd) for workerPara_hd in neighborhood_workerPara_id]
                    dist_w = np.sort(dist_w)
                    score_w.append( np.sum( dist_w[:(len(neighborhood_workerPara_id) - b - 2)] ) )
                ind_w = score_w.index( min(score_w) )
                S_w.append( neighborhood_workerPara_id.pop( ind_w ) )  # .pop()方法用于删除后返回列表中的一个元素
                # S_w 是一个( len( neighbor_list ) − 2*Prox_BRIDGE_B_Config['ByzantineSize'] )个元素的list
            """ Part 2 of trimmed mean """
            # Dimension of the gradient we are screening for
            S_w = np.array(S_w).reshape( len(S_w), -1 )
            for dim in range(prd.n0 * prd.n1):  # coordinate-wise is computationally expensive
                m_i = [w[dim] for w in S_w.tolist()]  # w是S_w的每一维度的遍历
                m_i = np.sort(m_i, axis = 0)  # 对行操作，按列从小到大排列
                if Prox_BRIDGE_B_Config['ByzantineSize'] != 0:
                    m_i = m_i[b: -b]
                    # remove the beginning and last 2*b information
                else:
                    m_i = m_i
                aggregation_temp[id][dim] = np.mean(m_i, axis = 0)  # requirement: neighbors = max(4b, 3b + 2)
            aggregation[id] = aggregation_temp[id].reshape(prd.n0, prd.n1)
        grad = prd.networkgrad( workerPara_temp )
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( aggregation - Prox_BRIDGE_B_Config['StepSize'] * grad, ( prd.reg_l1/prd.m ) *\
                                  Prox_BRIDGE_B_Config['StepSize'] )
        if k % 1 == 0:
            end = time.perf_counter()
            time_axis_Prox_BRIDGE_B.append( end - start )
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(Prox_BRIDGE_B_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_BRIDGE_B.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_BRIDGE_B_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_BRIDGE_B_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_BRIDGE_B_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_BRIDGE_B_Config['ReliableSet'])
            loss_Prox_BRIDGE_B.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_BRIDGE_B.append(acc)
            var = get_consensus_error( Prox_BRIDGE_B_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_BRIDGE_B.append(var)
            # if k % 1 == 0:
            if k % ( int( Prox_BRIDGE_B_Config['Iterations']/10) ) == 0:
                print('Prox-BRIDGE-B of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-BRIDGE-B', k, Prox_BRIDGE_B_Config['Iterations'] )
    # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-Prox-BRIDGE-B' + last_str + str(Prox_BRIDGE_B_Config['ByzantineSize']/prd.m) + '.txt', res_Prox_BRIDGE_B)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-BRIDGE-B' + last_str + str(Prox_BRIDGE_B_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_BRIDGE_B)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-BRIDGE-B' + last_str + str(Prox_BRIDGE_B_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_BRIDGE_B)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-Prox-BRIDGE-B' + last_str + str( Prox_BRIDGE_B_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_BRIDGE_B)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(acc_Prox_BRIDGE_B[-1], acc_Prox_BRIDGE_B[-1], var_Prox_BRIDGE_B[-1]))
    print('StepSize of Prox-BRIDGE-B:', Prox_BRIDGE_B_Config['StepSize'] )
    print('Loss of Prox-BRIDGE-B =', loss_Prox_BRIDGE_B[-1])
    time_axis_Prox_BRIDGE_B.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_BRIDGE_B), time_axis_Prox_BRIDGE_B[-1]))
    return para_epoch, para_ave_list, time_axis_Prox_BRIDGE_B, res_Prox_BRIDGE_B, loss_Prox_BRIDGE_B, last_str


def geometric_median(wList, max_iter=80, err=1e-5):
    # solve the argmin using an iterative method
    guess = torch.mean(wList, dim=0)  # average of the status of all neighbors, torch.Size([10, 784])
    for _ in range(max_iter):
        temp = (wList-guess).reshape(wList.shape[0], -1)
        # print(temp.shape)
        dist_li = torch.norm(temp, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temn1 = torch.sum(torch.stack([w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temn1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next.reshape(wList.shape[1], wList.shape[2])
        if guess_movement <= err:
            break
    return guess.numpy()

def Prox_GeoMed(prd, Prox_GeoMed_Config, image_test, label_test, attack, dsgd_optimal):
    # geometric median screening
    time_axis_Prox_GeoMed = []
    res_Prox_GeoMed = []
    loss_Prox_GeoMed = []
    acc_Prox_GeoMed = []   # 构建List类型空列表
    var_Prox_GeoMed = []
    para_ave_list = []
    workerPara = Prox_GeoMed_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in Prox_GeoMed_Config['ReliableSet']:
        para_ave += para[id] / Prox_GeoMed_Config['ReliableSize']
    var = get_consensus_error( Prox_GeoMed_Config['ReliableSet'], para, para_ave )
    var_Prox_GeoMed.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_Prox_GeoMed.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, Prox_GeoMed_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, Prox_GeoMed_Config['ReliableSet'])
    loss_Prox_GeoMed.append(loss)
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range( Prox_GeoMed_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy(workerPara)
        for id in range(prd.m):
            neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, Prox_GeoMed_Config['ByzantineNetwork'],\
                                                                                  Prox_GeoMed_Config['ByzantineSet'])
            # Byzantine attacks
            if attack != None:  # 存在Byzantine节点
                workerPara_temp, last_str = attack( id, workerPara_temp, Prox_GeoMed_Config['ByzantineSet'],\
                                                    Prox_GeoMed_Config['ReliableSet'], neighbors_Byzantine,\
                                                    neighbors_reliable, Prox_GeoMed_Config['WeightMatrix'] )
            else:
                last_str = '-No-Attacks-'
            aggregation[id] = cp.deepcopy( geometric_median( torch.tensor(workerPara_temp[neighbor_list]) ) )  # find the geometric median
        grad = prd.networkgrad( workerPara_temp )
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( aggregation - Prox_GeoMed_Config['StepSize'] * grad, ( prd.reg_l1/prd.m ) *\
                                  Prox_GeoMed_Config['StepSize'] )
        if k % 1 == 0:
            end = time.perf_counter()
            time_axis_Prox_GeoMed.append( end - start )
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(Prox_GeoMed_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_Prox_GeoMed.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in Prox_GeoMed_Config['ReliableSet']:
                para_ave += workerPara[id]/Prox_GeoMed_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, Prox_GeoMed_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, Prox_GeoMed_Config['ReliableSet'])
            loss_Prox_GeoMed.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_Prox_GeoMed.append(acc)
            var = get_consensus_error( Prox_GeoMed_Config['ReliableSet'], workerPara, para_ave )
            var_Prox_GeoMed.append(var)
            if k % ( int( Prox_GeoMed_Config['Iterations']/10) ) == 0:
                print('Prox-GeoMed of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor( 'Prox-GeoMed', k, Prox_GeoMed_Config['Iterations'] )
    # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-Prox-GeoMed' + last_str + str(Prox_GeoMed_Config['ByzantineSize']/prd.m) + '.txt', res_Prox_GeoMed)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-Prox-GeoMed' + last_str + str(Prox_GeoMed_Config['ByzantineSize'] / prd.m) + '.txt', loss_Prox_GeoMed)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-Prox-GeoMed' + last_str + str(Prox_GeoMed_Config['ByzantineSize']/prd.m) + '.txt', acc_Prox_GeoMed)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-Prox-GeoMed' + last_str + str( Prox_GeoMed_Config['ByzantineSize']/prd.m ) + '.txt', var_Prox_GeoMed)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_Prox_GeoMed[-1], acc_Prox_GeoMed[-1], var_Prox_GeoMed[-1]))
    print( 'StepSize of Prox-GeoMed:', Prox_GeoMed_Config['StepSize'] )
    print( 'Loss of Prox-GeoMed =', loss_Prox_GeoMed[-1] )
    time_axis_Prox_GeoMed.insert(0, 0)
    print( 'time_slots: {}, total_time_cost: {}'.format(len(time_axis_Prox_GeoMed), time_axis_Prox_GeoMed[-1]) )
    return para_epoch, para_ave_list, time_axis_Prox_GeoMed, res_Prox_GeoMed, loss_Prox_GeoMed, last_str

def PG_EXTRA(prd, PG_EXTRA_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_PG_EXTRA = []
    res_PG_EXTRA = []
    loss_PG_EXTRA = []
    acc_PG_EXTRA = []   # 构建List类型空列表
    var_PG_EXTRA = []
    para_ave_list = []
    workerPara = PG_EXTRA_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    workerPara_last = cp.deepcopy( para_epoch[-1] )
    grad_update = np.zeros( (prd.m, prd.n0, prd.n1) )
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in PG_EXTRA_Config['ReliableSet']:
        para_ave += para[id] / PG_EXTRA_Config['ReliableSize']
    var = get_consensus_error(PG_EXTRA_Config['ReliableSet'], para, para_ave )
    var_PG_EXTRA.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_PG_EXTRA.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, PG_EXTRA_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, PG_EXTRA_Config['ReliableSet'])
    loss_PG_EXTRA.append(loss)
    start = time.perf_counter()
    for k in tqdm(range( PG_EXTRA_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy(workerPara)
        # Byzantine attacks
        if k == 0:
            grad_last = prd.networkgrad( workerPara_last )
            temp = 0
            for id in range(prd.m):
                neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, PG_EXTRA_Config['ByzantineNetwork'],\
                                                                                      PG_EXTRA_Config['ByzantineSet'])
                if attack != None:  # 存在Byzantine节点
                    workerPara_last, last_str = attack(id, workerPara_last, PG_EXTRA_Config['ByzantineSet'],
                    PG_EXTRA_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, PG_EXTRA_Config['WeightMatrix'])
                for jd in neighbor_list:
                    # if (id, jd) in PG_EXTRA_Config['ReliableSet'].edges():
                    temp += PG_EXTRA_Config['WeightMatrix'][id, jd] * workerPara_last[jd]
            aggregation = cp.deepcopy(temp)
            z = aggregation - PG_EXTRA_Config['StepSize'] * grad_last
            workerPara_temp = prd.prox_l1( z, (prd.reg_l1 / PG_EXTRA_Config['ReliableSize']) * PG_EXTRA_Config['StepSize'] )
        else:
            grad_update = prd.networkgrad( workerPara_temp )
            grad_diff = -grad_update + grad_last
            for id in range(prd.m):
                neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, PG_EXTRA_Config['ByzantineNetwork'],\
                                                                                      PG_EXTRA_Config['ByzantineSet'])
                if attack != None:  # 存在Byzantine节点
                    workerPara_temp, last_str = attack( id, workerPara_temp, PG_EXTRA_Config['ByzantineSet'],
                    PG_EXTRA_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, PG_EXTRA_Config['WeightMatrix'] )
                else:
                    last_str = '-No-Attacks-'
                temp = 0
                for jd in neighbor_list:
                    temp += PG_EXTRA_Config['WeightMatrix'][id, jd] * workerPara_temp[jd]
                aggregation[id] = cp.deepcopy(temp)
            z = aggregation + z - np.matmul( PG_EXTRA_Config['WeightMatrix2'], workerPara_last ) + PG_EXTRA_Config['StepSize'] * grad_diff
            # The second matrix is adjustable M, which make a big dfference to the performace of PG-EXTRA.
            # proximal-gradient and gradient-descent step
            workerPara = prd.prox_l1( z, ( prd.reg_l1 ) * PG_EXTRA_Config['StepSize'] )
        workerPara_last = cp.deepcopy(workerPara_temp)
        grad_last = cp.deepcopy(grad_update)
        if k % 1 == 0:   # PG-EXTRA requires prd.b iterations to finish an epoch.
            end = time.perf_counter()
            time_axis_PG_EXTRA.append( end - start )
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(PG_EXTRA_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_PG_EXTRA.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in PG_EXTRA_Config['ReliableSet']:
                para_ave += workerPara[id]/PG_EXTRA_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, PG_EXTRA_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, PG_EXTRA_Config['ReliableSet'])
            loss_PG_EXTRA.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_PG_EXTRA.append(acc)
            var = get_consensus_error( PG_EXTRA_Config['ReliableSet'], workerPara, para_ave )
            var_PG_EXTRA.append(var)
            if k % ( int(PG_EXTRA_Config['Iterations'] / 10) ) == 0:
                print('the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor('PG-EXTRA', k, PG_EXTRA_Config['Iterations'])
    # Save the experiment results
    np.savetxt('results/txtdata/' + prd.setting + '-res-PG-EXTRA' + last_str + str(PG_EXTRA_Config['ByzantineSize'] / prd.m) + '.txt', res_PG_EXTRA)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-PG-EXTRA' + last_str + str(PG_EXTRA_Config['ByzantineSize'] / prd.m) + '.txt', loss_PG_EXTRA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-PG-EXTRA' + last_str + str(PG_EXTRA_Config['ByzantineSize']/prd.m) + '.txt', acc_PG_EXTRA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-PG-EXTRA' + last_str + str( PG_EXTRA_Config['ByzantineSize']/prd.m ) + '.txt', var_PG_EXTRA)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_PG_EXTRA[-1], acc_PG_EXTRA[-1], var_PG_EXTRA[-1]))
    print('StepSize of PG-EXTRA:', PG_EXTRA_Config['StepSize'] )
    print('Loss of PG-EXTRA =', loss_PG_EXTRA[-1])
    time_axis_PG_EXTRA.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_PG_EXTRA), time_axis_PG_EXTRA[-1]))
    return para_epoch, para_ave_list, time_axis_PG_EXTRA, res_PG_EXTRA, loss_PG_EXTRA, last_str

def NIDS(prd, NIDS_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_NIDS = []
    res_NIDS = []
    loss_NIDS = []
    acc_NIDS = []    # 构建List类型空列表
    var_NIDS = []
    para_ave_list = []
    workerPara = NIDS_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    workerPara_last = cp.deepcopy( para_epoch[-1] )
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    grad_last = np.zeros( (prd.m, prd.n0, prd.n1) )

    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in NIDS_Config['ReliableSet']:
        para_ave += para[id] / NIDS_Config['ReliableSize']
    var = get_consensus_error( NIDS_Config['ReliableSet'], para, para_ave )
    var_NIDS.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_NIDS.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, NIDS_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, NIDS_Config['ReliableSet'])
    loss_NIDS.append(loss)
    start = time.perf_counter()
    for k in tqdm(range( NIDS_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy( workerPara )
        if k == 0:
            grad_update = prd.networkgrad( workerPara_temp )
            z = workerPara_temp - np.matmul( NIDS_Config['StepSize'], grad_update.reshape(prd.m, -1) ).reshape(prd.m, prd.n0, prd.n1)
        else:
            grad_update = prd.networkgrad( workerPara_temp )
            grad_diff = -grad_update + grad_last
            Para_temp = 2*workerPara_temp - workerPara_last + np.matmul( NIDS_Config['StepSize'], grad_diff.reshape(prd.m, -1) ).reshape(prd.m, prd.n0, prd.n1)
            for id in range(prd.m):
                neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, NIDS_Config['ByzantineNetwork'],\
                                                                                      NIDS_Config['ByzantineSet'])
                # Byzantine attacks
                if attack != None:  # 存在Byzantine节点
                    Para_temp, last_str = attack( id, Para_temp, NIDS_Config['ByzantineSet'],\
                    NIDS_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, NIDS_Config['WeightMatrix'] )
                else:
                    last_str = '-No-Attacks-'
                temp = np.zeros( (prd.n0, prd.n1) )
                for jd in neighbor_list:
                    temp += NIDS_Config['WeightMatrix'][id, jd] * Para_temp[jd]
                aggregation[id] = cp.deepcopy( temp )
            z = z - workerPara_temp + aggregation
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( z, ( prd.reg_l1/prd.m ) * NIDS_Config['StepSize'] )
        grad_last = cp.deepcopy( grad_update )
        workerPara_last = cp.deepcopy( workerPara_temp )
        if k % 1 == 0:
            end = time.perf_counter()
            time_axis_NIDS.append(end - start)
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(NIDS_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_NIDS.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in NIDS_Config['ReliableSet']:
                para_ave += workerPara[id]/NIDS_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, NIDS_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, NIDS_Config['ReliableSet'])
            loss_NIDS.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_NIDS.append(acc)
            var = get_consensus_error( NIDS_Config['ReliableSet'], workerPara, para_ave )
            var_NIDS.append(var)
            if k % ( int(NIDS_Config['Iterations'] / 10) ) == 0:
                print('NIDS of the {}th iteration res: {}, loss: {}, acc: {}, vars: {}'.format(k, res, loss, acc, var))
            ut.monitor('NIDS', k, NIDS_Config['Iterations'])
    # Save the experiment results
    np.savetxt('results/txtdata/' + prd.setting + '-res-NIDS' + last_str + str(NIDS_Config['ByzantineSize'] / prd.m) + '.txt', res_NIDS)
    np.savetxt('results/txtdata/' + prd.setting + '-loss-NIDS' + last_str + str(NIDS_Config['ByzantineSize'] / prd.m) + '.txt', loss_NIDS)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-NIDS' + last_str + str( NIDS_Config['ByzantineSize']/prd.m ) + '.txt', acc_NIDS)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-NIDS' + last_str + str( NIDS_Config['ByzantineSize']/prd.m ) + '.txt', var_NIDS)
    print('the final iteration res: {}, acc: {}, vars: {}'.format(res_NIDS[-1], acc_NIDS[-1], var_NIDS[-1]))
    print('StepSize of NIDS:', NIDS_Config['StepSize'] )
    time_axis_NIDS.insert(0, 0)
    print('Loss of NIDS =', loss_NIDS[-1])
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_NIDS), time_axis_NIDS[-1]))
    return para_epoch, para_ave_list, time_axis_NIDS, res_NIDS, loss_NIDS, last_str


""" The following algorithms are not tested.
def NIDS_SAGA(prd, NIDS_SAGA_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_NIDS_SAGA = []
    res_NIDS_SAGA = []
    loss_NIDS_SAGA = []
    acc_NIDS_SAGA = []
    var_NIDS_SAGA = []
    para_ave_list = []
    workerPara = NIDS_SAGA_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in NIDS_SAGA_Config['ReliableSet']:
        para_ave += para[id] / NIDS_SAGA_Config['ReliableSize']
    var = get_consensus_error( NIDS_SAGA_Config['ReliableSet'], para, para_ave )
    var_NIDS_SAGA.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_NIDS_SAGA.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, NIDS_SAGA_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, NIDS_SAGA_Config['ReliableSet'])
    loss_NIDS_SAGA.append(loss)
    workerPara_last = cp.deepcopy( para_epoch[-1] )
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    StoGrad1 = np.zeros( (prd.m, prd.n0, prd.n1) )  # initialization
    slots = np.array([np.zeros((prd.data_distr[i], prd.dim)) for i in range(prd.m)])
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])
    sum_grad = np.zeros( (prd.m, prd.n0, prd.n1) )
    for id in range(prd.m):
        slots[id][sample_vec[id]] = prd.localgrad(workerPara, id, sample_vec[id])
        sum_grad[id] = np.sum(slots[id], axis = 0)
    SAGA = np.zeros( (prd.m, prd.n0, prd.n1) )
    SAGA_last = np.zeros( (prd.m, prd.n0, prd.n1) )
    start = time.perf_counter()
    for k in tqdm(range( NIDS_SAGA_Config['Iterations'] )):
        workerPara_temp = cp.deepcopy( workerPara )
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])
        for id in range(prd.m):
            # local stochastic gradient estimation
            StoGrad1[id] = prd.localgrad(workerPara_temp, id, sample_vec[id])
            StoGrad_Diff = StoGrad1[id] - slots[id][sample_vec[id]]  # prd.localgrad() is the stochastic gradient 1
            SAGA[id] = StoGrad_Diff + sum_grad[id] / prd.data_distr[id]
            sum_grad[id] += StoGrad_Diff
            slots[id][sample_vec[id]] = StoGrad1[id]  # update of the gradient table position/slot
        if k == 0:
            z = workerPara_temp - np.matmul( NIDS_SAGA_Config['StepSize'], SAGA )
        else:
            grad_diff = -SAGA + SAGA_last
            Para_temp = 2 * workerPara_temp - workerPara_last + np.matmul( NIDS_SAGA_Config['StepSize'], grad_diff )
            for id in range(prd.m):
                neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id,\
                NIDS_SAGA_Config['ByzantineNetwork'], NIDS_SAGA_Config['ByzantineSet'])
                # Byzantine attacks
                if attack != None:  # 存在Byzantine节点
                    Para_temp, last_str = attack(id, Para_temp, NIDS_SAGA_Config['ByzantineSet'],
                    NIDS_SAGA_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, NIDS_SAGA_Config['WeightMatrix'])
                else:
                    last_str = '-No-Attacks-'
                temp = np.zeros( (prd.n0, prd.n1) )
                for jd in neighbor_list: # for jd in prd.m
                    temp += NIDS_SAGA_Config['WeightMatrix'][id, jd] * Para_temp[jd]
                aggregation[id] = cp.deepcopy(temp)
            z = z - workerPara_temp + aggregation
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( z, ( prd.reg_l1/prd.m ) * NIDS_SAGA_Config['StepSize'] )
        SAGA_last = cp.deepcopy( SAGA )
        workerPara_last = cp.deepcopy( workerPara_temp )
        if k % prd.b == 0:
            end = time.perf_counter()
            time_axis_NIDS_SAGA.append(end - start)
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(NIDS_SAGA_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_NIDS_SAGA.append(res)
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in NIDS_SAGA_Config['ReliableSet']:
                para_ave += workerPara[id] / NIDS_SAGA_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, NIDS_SAGA_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, NIDS_SAGA_Config['ReliableSet'])
            loss_NIDS_SAGA.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_NIDS_SAGA.append(acc)
            var = get_consensus_error( NIDS_SAGA_Config['ReliableSet'], workerPara, para_ave )
            var_NIDS_SAGA.append(var)
            if k % ( int( NIDS_SAGA_Config['Iterations'] / 10) ) == 0:
                print('NIDS-SAGA of the {}th iteration loss: {}, acc: {}, vars: {}'.format(k, loss, acc, var))
            ut.monitor('NIDS-SAGA', k, NIDS_SAGA_Config['Iterations'])
    # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-NIDS-SAGA' + last_str + str( NIDS_SAGA_Config['ByzantineSize']/prd.m ) + '.txt', res_NIDS_SAGA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-loss-NIDS-SAGA' + last_str + str( NIDS_SAGA_Config['ByzantineSize']/prd.m ) + '.txt', loss_NIDS_SAGA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-NIDS-SAGA' + last_str + str( NIDS_SAGA_Config['ByzantineSize']/prd.m ) + '.txt', acc_NIDS_SAGA)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-NIDS-SAGA' + last_str + str( NIDS_SAGA_Config['ByzantineSize']/prd.m ) + '.txt', var_NIDS_SAGA)
    print('the final iteration acc: {}, vars: {}'.format(acc_NIDS_SAGA[-1], var_NIDS_SAGA[-1]))
    print('StepSize of NIDS-SAGA:', NIDS_SAGA_Config['StepSize'] )
    print('Loss of NIDS-SAGA =', loss_NIDS_SAGA[-1])
    time_axis_NIDS_SAGA.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_NIDS_SAGA), time_axis_NIDS_SAGA[-1]))
    return para_epoch, para_ave_list, time_axis_NIDS_SAGA, loss_NIDS_SAGA, last_str


def NIDS_LSVRG(prd, NIDS_LSVRG_Config, image_test, label_test, attack, dsgd_optimal):
    time_axis_NIDS_LSVRG = []
    count_trigger = 0
    loss_NIDS_LSVRG = []
    res_NIDS_LSVRG = []
    acc_NIDS_LSVRG = []
    var_NIDS_LSVRG = []
    para_ave_list = []
    workerPara = NIDS_LSVRG_Config['Initialization']
    para_epoch = [ cp.deepcopy(workerPara) ]
    # for a same starting point
    para = cp.deepcopy( para_epoch[-1] )
    para_ave = np.zeros( (prd.n0, prd.n1) )
    for id in NIDS_LSVRG_Config['ReliableSet']:
        para_ave += para[id] / NIDS_LSVRG_Config['ReliableSize']
    var = get_consensus_error( NIDS_LSVRG_Config['ReliableSet'], para, para_ave )
    var_NIDS_LSVRG.append(var)
    acc = get_accuracy(para_ave, image_test, label_test)
    acc_NIDS_LSVRG.append(acc)
    para_ave_list.append(para_ave)
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    # loss = prd.call_loss_dec(para_ave, NIDS_LSVRG_Config['ReliableSet'], 'averaged')
    # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
    loss = prd.call_loss_dec(workerPara, NIDS_LSVRG_Config['ReliableSet'])
    loss_NIDS_LSVRG.append(loss)
    workerPara_last = cp.deepcopy( para_epoch[-1] )
    w = cp.deepcopy(workerPara_last)
    StoGrad1 = np.zeros( (prd.m, prd.n0, prd.n1) )
    StoGrad2 = np.zeros( (prd.m, prd.n0, prd.n1) )
    sum_grad = prd.networkgrad( workerPara_last )  # local batch gradient
    aggregation = np.zeros( (prd.m, prd.n0, prd.n1) )
    LSVRG = np.zeros( (prd.m, prd.n0, prd.n1) )
    LSVRG_last = np.zeros( (prd.m, prd.n0, prd.n1) )
    k = 0
    Epoch_observation = 0
    start = time.perf_counter()
    while Epoch_observation < NIDS_LSVRG_Config['Iterations']:
    # for k in tqdm(range(int( K ))):
        workerPara_temp = cp.deepcopy( workerPara )
        sum_grad_last = cp.deepcopy(sum_grad)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.m)])  # changes at every iteration
        trigger = 0
        for id in range(prd.m):
            # local stochastic gradient estimation
            StoGrad1[id] = prd.localgrad(workerPara_temp, id, sample_vec[id])  # computation of stochastic gradient 1
            StoGrad2[id] = prd.localgrad(w, id, sample_vec[id])  # computation of stochastic gradient 2
            LSVRG[id] = StoGrad1[id] - StoGrad2[id] + sum_grad[id]
            if np.random.random() < NIDS_LSVRG_Config['Triggered Probability']:  # uncoordinated triggered probabilities
                w[id] = workerPara_temp[id]
                sum_grad[id] = prd.localgrad(workerPara_temp, id)  # compute the local full gradients
                count_trigger += 1
                trigger = 1
            else:
                sum_grad[id] = sum_grad_last[id]
        if k == 0:
            z = workerPara_temp - np.matmul( NIDS_LSVRG_Config['StepSize'], LSVRG )
        else:
            grad_diff = -LSVRG + LSVRG_last
            Para_temp = 2 * workerPara_temp - workerPara_last + np.matmul(NIDS_LSVRG_Config['StepSize'], grad_diff)
            for id in range(prd.m):
                neighbor_list, neighbors_Byzantine, neighbors_reliable = get_neighbors(prd, id, NIDS_LSVRG_Config['ByzantineNetwork'], NIDS_LSVRG_Config['ByzantineSet'])
                # Byzantine attacks
                if attack != None:  # 存在Byzantine节点
                    Para_temp, last_str = attack( id, Para_temp, NIDS_LSVRG_Config['ByzantineSet'], NIDS_LSVRG_Config['ReliableSet'], neighbors_Byzantine, neighbors_reliable, NIDS_LSVRG_Config['WeightMatrix'] )
                else:
                    last_str = '-No-Attacks-'
                temp = np.zeros( (prd.n0, prd.n1) )
                for jd in neighbor_list: # for jd in prd.m
                    temp += NIDS_LSVRG_Config['WeightMatrix'][id, jd] * Para_temp[jd]
                aggregation[id] = cp.deepcopy(temp)
            z = z - workerPara_temp + aggregation
            # z = z - workerPara_temp + np.matmul(W, 2*workerPara_temp - workerPara_last + np.matmul(step_size, grad_diff))
        # proximal-gradient and gradient-descent step
        workerPara = prd.prox_l1( z, ( prd.reg_l1/prd.m ) * NIDS_LSVRG_Config['StepSize'] )
        LSVRG_last = cp.deepcopy(LSVRG)
        workerPara_last = cp.deepcopy(workerPara_temp)
        if k % ( prd.b/2) == 0 or trigger == 1:
            end = time.perf_counter()
            para_epoch.append( cp.deepcopy(workerPara) )
            res = get_residual(NIDS_LSVRG_Config['ReliableSet'], workerPara, dsgd_optimal)
            res_NIDS_LSVRG.append(res)
            time_axis_NIDS_LSVRG.append(end-start)
            Epoch_observation += 1
            para_ave = np.zeros( (prd.n0, prd.n1) )
            for id in NIDS_LSVRG_Config['ReliableSet']:
                para_ave += workerPara[id]/NIDS_LSVRG_Config['ReliableSize']
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{\bar x_k}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            # loss = prd.call_loss_dec(para_ave, NIDS_LSVRG_Config['ReliableSet'], 'averaged')
            # for \sum\nolimits_{i \in \mathcal{R}} {\left( {{h_i}\left( {{x_{i,k}}} \right) - {h_i}\left( {{{\tilde x}^*}} \right)} \right)}
            loss = prd.call_loss_dec(workerPara, NIDS_LSVRG_Config['ReliableSet'])
            loss_NIDS_LSVRG.append(loss)
            acc = get_accuracy(para_ave, image_test, label_test)
            acc_NIDS_LSVRG.append(acc)
            var = get_consensus_error( NIDS_LSVRG_Config['ReliableSet'], workerPara, para_ave )
            var_NIDS_LSVRG.append(var)
            if k % ( int( NIDS_LSVRG_Config['Iterations'] / 10) ) == 0:
                print('NIDS-LSVRG of the {}th iteration loss: {}, acc: {}, vars: {}'.format(k, loss, acc, var))
            ut.monitor('NIDS-LSVRG', k, NIDS_LSVRG_Config['Iterations'])
        k += 1
    # Save the experiment results
    np.savetxt( 'results/txtdata/' + prd.setting + '-res-NIDS-LSVRG' + last_str + str(NIDS_LSVRG_Config['ByzantineSize']/prd.m) + '.txt', res_NIDS_LSVRG)
    np.savetxt( 'results/txtdata/' + prd.setting + '-acc-NIDS-LSVRG' + last_str + str(NIDS_LSVRG_Config['ByzantineSize']/prd.m) + '.txt', acc_NIDS_LSVRG)
    np.savetxt( 'results/txtdata/' + prd.setting + '-consensus-error-NIDS-LSVRG' + last_str + str(NIDS_LSVRG_Config['ByzantineSize']/prd.m) + '.txt', var_NIDS_LSVRG)
    print('the final iteration acc: {}, vars: {}'.format(acc_NIDS_LSVRG[-1], var_NIDS_LSVRG[-1]))
    print("number of trigger =", count_trigger)
    # print("the final iteration:", k)
    print("the triggered probability:", NIDS_LSVRG_Config['Triggered Probability'])
    print('StepSize of NIDS-LSVRG:', NIDS_LSVRG_Config['StepSize'])
    print('Loss of NIDS-LSVRG =', loss[-1])
    time_axis_NIDS_LSVRG.insert(0, 0)
    print('time_slots: {}, total_time_cost: {}'.format(len(time_axis_NIDS_LSVRG), time_axis_NIDS_LSVRG[-1]))
    # print('selected ID:', Prox_DBRO_LSVRG_Config['selected id of testing agent'])
    # print('optimal grad:', LSVRG[select])
    return para_epoch, para_ave_list, time_axis_NIDS_LSVRG, loss, last_str
"""
