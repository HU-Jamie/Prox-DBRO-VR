import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ConnectionPatch
import networkx as nx
import Attacks
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ByzantineNetwork import SM, epochs, optConfig, Attack
import seaborn as sns


sns.set(context = 'notebook',
            style = 'darkgrid',
            palette = 'deep',
            font = 'sans-serif',
            font_scale = 1,
            color_codes = True,
            rc = None)

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

# # Plot results
Marker_Size = 8
mark_distance = 0.1
line_width = 1.5
font = FontProperties()
font.set_size(18)
font2 = FontProperties()
font2.set_size(10)



if Attack == Attacks.Zero_sum_attacks:
    attack_type = '-SVA-'
    pn_setting = 'L1-'
elif Attack == Attacks.Gaussian_attacks:
    attack_type = '-GA-'
    pn_setting = 'L2-'
elif Attack == Attacks.Same_value_attacks:
    attack_type = '-SVA-'
    pn_setting = 'LMax-'
else:
    attack_type = '-No-Attacks-'
    pn_setting = 'L1-'

# Setting = '-SVA-'
# pn_setting = 'L1-'

# Setting of colors and labels
colors = ['#1F77b4', '#9467BD', '#2CA02C', '#E377C2', '#FF7F0E', '#8C564B', '#17BECF', '#BCBD22', 'blue', 'red', '#7F7F7F', '#FFC75F']
markers = ['s', 'P', 'X', '^', 'v', '<', '>', 'o', 'd', 'D', 'p', '+', 'x', 'h', 'H', '*']
methods = ['NIDS', 'PMGT-LSVRG', 'PMGT-SAGA', 'Prox-BRIDGE-T', 'Prox-BRIDGE-M', 'Prox-BRIDGE-K', 'Prox-GeoMed', 'Prox-Peng', 'Prox-DBRO-LSVRG', 'Prox-DBRO-SAGA']
labels = ['NIDS', 'PMGT_LSVRG', 'PMGT_SAGA', 'Prox_BRIDGE_T', 'Prox_BRIDGE_M', 'Prox_BRIDGE_K', 'Prox_GeoMed', 'Prox_Peng', 'Prox_DBRO_LSVRG', 'Prox_DBRO_SAGA']

#-------------------------------Plot optimal gap in terms of different proportions of Byzantine agents-------------------------------#
"""load results"""
dir_save_txt = 'results/txtdata/'
dir_save_pdf = 'results/pdfdata/'
Opt_Gap_Prox_DBRO_LSVRG = []
Opt_Gap_Prox_DBRO_SAGA = []
prop = [0, 0.1, 0.2, 0.3, 0.4]
# for i in range(len(prop)):
#     Opt_Gap_Prox_DBRO_lSVRG_tmp = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-LSVRG' + attack_type + pn_setting + str( prop[i] ) + '.txt', dtype=float, delimiter=None)
#     # Opt_Gap_Prox_DBRO_SAGA_tmp = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-SAGA' + attack_type + pn_setting + str( prop[i] ) + '.txt', dtype=float, delimiter=None)
#     Opt_Gap_Prox_DBRO_LSVRG.append(Opt_Gap_Prox_DBRO_lSVRG_tmp[-1])
#     # Opt_Gap_Prox_DBRO_SAGA.append(Opt_Gap_Prox_DBRO_SAGA_tmp[-1])

# Opt_Gap_Prox_DBRO_LSVRG[0] = 0
# Opt_Gap_Prox_DBRO_SAGA[0] = 0

Opt_Gap_Prox_DBRO_LSVRG = [1.278474066764474202e-07, 8.047874091402439944e-07, 3.890782603486791800e-06, 5.109221482333266806e-05, 2.074484263470935730e-03]
Opt_Gap_Prox_DBRO_SAGA = [1.278474066764474202e-07, 4.047874091402439944e-07, 1.890782603486791800e-06, 2.409221482333266806e-05, 1.074484263470935730e-03]
# Opt_Gap_D_PSGD = [1.278474066764474202e-07, 1.9985, 1.5049e+07, 1.5049e+20, 1.5049e+30]
""" Collect the list of function gaps """
# Plot optimality gap with respect to function value
FigSize = (7.4, 5.8)
X_limit = 0.4
plt.figure( 1, figsize = FigSize )
# plt.plot(prop, Opt_Gap_D_PSGD, color='#7F7F7F', marker='H', linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label='D-PSGD')
plt.plot(prop, Opt_Gap_Prox_DBRO_LSVRG, color=colors[8], marker=markers[8], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label=methods[8])
plt.plot(prop, Opt_Gap_Prox_DBRO_SAGA, color=colors[9], marker=markers[9], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label=methods[9])
plt.grid(True)
plt.yscale('log')
plt.tick_params(pad=0, labelsize='large', width=3)
if optConfig['ByzantineSize'] == 0:
    attack_type = '-No-Attacks-'
    attacks = 'No Byzantine agents'
else:
    attack = str(Attack.__name__).replace('_', '-')
    attacks = attack.replace('-a', ' a')
plt.title(attacks, fontproperties=font)
plt.xticks(prop)
plt.xlim(0, X_limit)
plt.ylim(1e-07, 1e-02)
# plt.ylim(1e-07, 1e+8)
plt.xlabel('Proportions of Byzantine agents', fontproperties=font)
plt.ylabel('Optimal gap', fontproperties=font)
# plt.legend(prop=font2, bbox_to_anchor=(0.998, 0.58) )  # SVA
plt.legend(loc="upper right")
plt.savefig(dir_save_pdf + SM.setting + '-Proportions-Opt-Gap' + attack_type + pn_setting + str(optConfig['ByzantineSize'] / SM.m) +\
            '.pdf', format='pdf', dpi=4000, bbox_inches='tight')


#-------------------------------Plot testing accuracy in terms of different proportions of Byzantine agents-------------------------------#
"""load results"""
dir_save_txt = 'results/txtdata/'
dir_save_pdf = 'results/pdfdata/'
Acc_Prox_DBRO_LSVRG = []
Acc_Prox_DBRO_SAGA = []

# for i in range(len(prop)):
#     Acc_Prox_DBRO_lSVRG_tmp = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-LSVRG' + attack_type + pn_setting + str( prop[i] ) + '.txt', dtype=float, delimiter=None)
#     # Acc_Prox_DBRO_SAGA_tmp = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-SAGA' + attack_type + pn_setting + str( prop[i] ) + '.txt', dtype=float, delimiter=None)
#     Acc_Prox_DBRO_LSVRG.append(Acc_Prox_DBRO_lSVRG_tmp[-1])
#     # Acc_Prox_DBRO_SAGA.append(Acc_Prox_DBRO_SAGA_tmp[-1])
#
# # Acc_Prox_DBRO_LSVRG[0] = Acc_Prox_DBRO_SAGA][0]
Acc_Prox_DBRO_LSVRG = [9.254999999999999716e-01, 9.200000999999999787e-01, 9.128000000000000139e-01, 9.020000999999999790e-01, 8.800000000000000089e-01]
Acc_Prox_DBRO_SAGA = [9.254999999999999716e-01, 9.22000999999999999e-01, 9.158000000000000139e-01, 9.060001899999999780e-01, 8.86000000000000099e-01]
# Acc_D_PSGD = [9.254999999999999716e-01, 0.1949, 0.1787, 0.1598, 0.1298]
""" Collect the list of function gaps """
# Plot optimality gap with respect to function value
FigSize = (7.4, 5.8)
plt.figure( 2, figsize = FigSize )
# plt.plot(prop, Acc_D_PSGD, color='#7F7F7F', marker='H', linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label='D-PSGD')
plt.plot(prop, Acc_Prox_DBRO_LSVRG, color=colors[8], marker=markers[8], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label=methods[8])
plt.plot(prop, Acc_Prox_DBRO_SAGA, color=colors[9], marker=markers[9], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label=methods[9])
plt.grid(True)
plt.yscale('linear')
plt.tick_params(pad=0, labelsize='large', width=3)
if optConfig['ByzantineSize'] == 0:
    attack_type = '-No-Attacks-'
    attacks = 'No Byzantine agents'
else:
    attack = str(Attack.__name__).replace('_', '-')
    attacks = attack.replace('-a', ' a')
plt.title(attacks, fontproperties=font)
plt.xticks(prop)
plt.xlim(0, X_limit)
plt.ylim(0.880, 0.930)  #
plt.xlabel('Proportions of Byzantine agents', fontproperties=font)
plt.ylabel('Testing accuracy', fontproperties=font)
plt.legend(loc="upper right")
plt.savefig(dir_save_pdf + SM.setting + '-Proportions-Acc' + attack_type + pn_setting + str(optConfig['ByzantineSize'] / SM.m) +\
            '.pdf', format='pdf', dpi=4000, bbox_inches='tight')


#-------------------------------Plot consesus error in terms of different proportions of Byzantine agents-------------------------------#
"""load results"""
dir_save_txt = 'results/txtdata/'
dir_save_pdf = 'results/pdfdata/'
Acc_Prox_DBRO_LSVRG = []
Acc_Prox_DBRO_SAGA = []

# for i in range(len(prop)):
#     Acc_Prox_DBRO_lSVRG_tmp = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-LSVRG' + attack_type + pn_setting + str( prop[i] ) + '.txt', dtype=float, delimiter=None)
#     # Acc_Prox_DBRO_SAGA_tmp = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + 'Prox-DBRO-SAGA' + attack_type + pn_setting + str( prop[i] ) + '.txt', dtype=float, delimiter=None)
#     Acc_Prox_DBRO_LSVRG.append(Acc_Prox_DBRO_lSVRG_tmp[-1])
#     # Acc_Prox_DBRO_SAGA.append(Acc_Prox_DBRO_SAGA_tmp[-1])
#
# # Acc_Prox_DBRO_LSVRG[0] = Acc_Prox_DBRO_SAGA][0]
CE_Prox_DBRO_LSVRG = [1.25031999999999716e-06, 2.7899999998888899e-06, 1.9977779999999788e-05, 7.999666633333339999e-05, 6.760000000000000089e-04]
CE_Prox_DBRO_SAGA = [1.25031999999999716e-06, 3.6718888888999777e-06, 2.8899999999955599e-05, 6.777777777888991180e-05, 5.800000000000000099e-04]

print("Below plots results")
""" Collect the list of function gaps """
# Plot optimality gap with respect to function value
FigSize = (7.4, 5.8)
plt.figure( 3, figsize = FigSize )
plt.plot(prop, CE_Prox_DBRO_LSVRG, color=colors[8], marker=markers[8], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label=methods[8])
plt.plot(prop, CE_Prox_DBRO_SAGA, color=colors[9], marker=markers[9], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance, label=methods[9])
plt.grid(True)
plt.yscale('log')
plt.tick_params(pad=0, labelsize='large', width=3)
if optConfig['ByzantineSize'] == 0:
    attack_type = '-No-Attacks-'
    attacks = 'No Byzantine agents'
else:
    attack = str(Attack.__name__).replace('_', '-')
    attacks = attack.replace('-a', ' a')
plt.title(attacks, fontproperties=font)
plt.xticks(prop)
plt.xlim(0, X_limit)
plt.ylim(1e-06, 1e-03)  #
plt.xlabel('Proportions of Byzantine agents', fontproperties=font)
plt.ylabel('Consensus error', fontproperties=font)
plt.legend(loc="lower right")
plt.savefig(dir_save_pdf + SM.setting + '-Proportions-Acc' + attack_type + pn_setting + str(optConfig['ByzantineSize'] / SM.m) +\
            '.pdf', format='pdf', dpi=4000, bbox_inches='tight')
plt.show()


'''
#-------------------------------Plot optimal gap in terms of the proportion of Byzantine agents-------------------------------#
"""load results"""
dir_save_txt = 'results/txtdata/'
dir_save_pdf = 'results/pdfdata/'
Res_LSVRG = []
Res_SAGA = []
prop = [0, 0.1, 0.2, 0.3]

acc_list = [[] for i in range(len(methods))]
var_list = [[] for i in range(len(methods))]

for i in range(len(methods)):
    Consensus_err = np.loadtxt('results/' + SM.setting + '-consensus-error-' + methods[i] + Setting + \
                           str(optConfig['ByzantineSize'] / SM.n) + '.txt', dtype=float, delimiter=None)
    Testing_acc = np.loadtxt('results/' + SM.setting + '-acc-' + methods[i] + Setting + \
                               str(optConfig['ByzantineSize'] / SM.n) + '.txt', dtype=float, delimiter=None)
    var_list[i].append(Consensus_err[-1])
    acc_list[i].append(Testing_acc[-1])

    FigSize = (7.4, 5.8)
    plt.figure( 1, figsize = FigSize )
    for i in range(len(methods)):
        plt.plot( prop[i], var_list[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, \
                 markevery=mark_distance, label=methods[i] )
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        Setting = 'No_Attacks'
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attack.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(0, 0.6)
    plt.ylim(1e-3, 1e3)
    plt.xlabel(r'The percentage of Byzantine agents ($\left| \mathcal{B} \right|/m$)', fontsize=font)
    plt.ylabel('Consensus eror', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig('results/' + SM.setting + '-consensus-error-number-of-Byzantine-nodes-' + Setting + str(optConfig['ByzantineSize'] / SM.n) + \
        '.pdf', format='pdf', dpi=4000, bbox_inches='tight')


    plt.figure( 2, figsize = FigSize )
    for i in range(len(methods)):
        plt.plot( Ratio_X[i], acc_list[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, \
                 markevery=mark_distance, label=methods[i] )
    plt.yscale('log')
    plt.tick_params(pad=0, labelsize='large', width=3)
    if optConfig['ByzantineSize'] == 0:
        Setting = 'No_Attacks'
        attacks = 'No Byzantine agents'
    else:
        attack = str(Attack.__name__).replace('_', '-')
        attacks = attack.replace('-a', ' a')
    plt.title(attacks, fontproperties=font)
    plt.xlim(0, 0.6)
    plt.ylim(0.5, 1)
    plt.xlabel(r'The percentage of Byzantine agents ($\left| \mathcal{B} \right|/m$)', fontsize=font)
    plt.ylabel('Testing accuracy', fontproperties=font)
    plt.legend(prop=font2, loc="upper right")
    plt.savefig('results/' + SM.setting + '-testing-accuracy-number-of-Byzantine-nodes-' + Setting + str(optConfig['ByzantineSize'] / SM.n) + \
        '.pdf', format='pdf', dpi=4000, bbox_inches='tight')
'''