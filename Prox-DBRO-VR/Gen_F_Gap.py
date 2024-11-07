import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch
import networkx as nx
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ByzantineNetwork import lr_0, epochs, optConfig, Attack
import seaborn as sns
from analysis import error

Setting = '-SVA-'
pn_setting = 'L1-'

methods = ['NIDS', 'PMGT-LSVRG', 'PMGT-SAGA', 'Prox-BRIDGE-T', 'Prox-BRIDGE-M', 'Prox-BRIDGE-K', 'Prox-GeoMed', 'Prox-Peng', 'Prox-DBRO-LSVRG', 'Prox-DBRO-SAGA']
labels = ['NIDS', 'PMGT_LSVRG', 'PMGT_SAGA', 'Prox_BRIDGE_T', 'Prox_BRIDGE_M', 'Prox_BRIDGE_K', 'Prox_GeoMed', 'Prox_Peng', 'Prox_DBRO_LSVRG', 'Prox_DBRO_SAGA']

"""load results"""
dir_save_txt = 'results/txtdata/'
dir_save_pdf = 'results/pdfdata/'

# """
# recover losses according to function gaps
for i in range(len(methods)):
    i = 1
    # Load the Function gap of all tested algorithms
    F_gap_tmp = np.loadtxt(dir_save_txt + lr_0.setting + '-F-gap-' + methods[i] + Setting + pn_setting
                   + str(optConfig['ByzantineSize'] / lr_0.m) + '.txt', dtype=float, delimiter=None)
    K = len( F_gap_tmp )
    loss_tmp = []
    for k in range(K):
        loss_tmp.append((F_gap_tmp[k] * optConfig['ReliableSize'] + optConfig['OptimalValue']) )
    np.savetxt(dir_save_txt + lr_0.setting + '-loss-' + methods[i] + Setting + str(optConfig['ByzantineSize'] / lr_0.m)
               + '.txt', loss_tmp)
# """ # generate the function gaps according to their losses
# time_list = []
# for i in range(len(methods)):
#     # Load the Function gap of all tested algorithms
#     F = np.loadtxt(dir_save_txt + lr_0.setting + '-loss-' + methods[i] + Setting + str(optConfig['ByzantineSize'] / lr_0.m) + '.txt', dtype=float, delimiter=None)
#     error_lr_0 = error( lr_0, optConfig['OptimalModel'], optConfig['OptimalValue'] )
#     tmp_F_gap = error_lr_0.loss_gap_path( F, optConfig['ReliableSet'] )
#     # x_res_NIDS = error_lr_0.theta_gap_path(x_NIDS, Reliable)
#     # save optimal function gaps
#     np.savetxt(dir_save_txt + lr_0.setting + '-F-gap-' + methods[i] + Setting + pn_setting + str(optConfig['ByzantineSize'] / lr_0.m)
#                + '.txt', tmp_F_gap)
# """

