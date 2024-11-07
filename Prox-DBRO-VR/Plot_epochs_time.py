########################################################################################################################
####-----------------------------------------------Geometric Network------------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Geometric directed graphs using logistic regression.
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
mark_distance = 15
line_width = 1.5
font = FontProperties()
font.set_size(18)
font2 = FontProperties()
font2.set_size(10)



if Attack == Attacks.Zero_sum_attacks:
    attack_type = '-ZSA-'
    pn_setting = 'LMax-'
elif Attack == Attacks.Gaussian_attacks:
    attack_type = '-GA-'
    pn_setting = 'L2-'
elif Attack == Attacks.Same_value_attacks:
    attack_type = '-SVA-'
    pn_setting = 'L1-'

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

"""load results"""
dir_save_txt = 'results/txtdata/Real data for SVA/'
dir_save_pdf = 'results/pdfdata/'
# dir_save_txt = 'results/txtdata/'
# dir_save_pdf = 'results/pdfdata/'
F_gap = []
time_list = []
for i in range(len(methods)):
    tmp_F_gap = np.loadtxt(dir_save_txt + SM.setting + '-F-gap-' + methods[i] + attack_type + pn_setting +\
                            str(optConfig['ByzantineSize'] / SM.m) + '.txt', dtype=float, delimiter=None)
    F_gap.append(tmp_F_gap)
    # # Save running time of all tested algorithms
    # temp_time_list = np.loadtxt(dir_save_txt + SM.setting + '-time-' + methods[i] + attack_type +\
    #                           str(optConfig['ByzantineSize'] / SM.m) + '.txt', dtype=float, delimiter=None)
    # time_list.append(temp_time_list)
#-------------------------------Plot optimality gap in terms of epochs-------------------------------#
print("Below plots results")
""" Collect the list of function gaps """
X_limit = 152
# Plot optimality gap with respect to function value
FigSize = (7.6, 5.8)
plt.figure( 1, figsize = FigSize )
for i in range( len(methods) ):
    plt.plot(F_gap[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size,\
             markevery=mark_distance, label=methods[i] )
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
plt.xlim(-2, X_limit)

# plt.ylim(0.008, 4.5)  # ZSA
# plt.ylim(1e-2, 1e2)  # GA
plt.ylim(0.05, 1e6)  # SVA

plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Optimal gap', fontproperties=font)
# plt.legend(prop=font2,  bbox_to_anchor=(0.67, 0.48) )  # ZSA
# plt.legend(prop=font2,  bbox_to_anchor=(0.67, 0.40) )  # GA
plt.legend(prop=font2, bbox_to_anchor=(0.998, 0.58) )  # SVA
plt.savefig(dir_save_pdf + SM.setting + '-Epochs-F-gap-Byzan' + attack_type + pn_setting + str(optConfig['ByzantineSize'] / SM.m) +\
            '.pdf', format='pdf', dpi=4000, bbox_inches='tight')
"""
# -------------------------------Plot optimality gap in terms of time-------------------------------#
plt.figure( 2, figsize = FigSize )
for i in range(len(methods)):
    plt.plot(time_list[i], F_gap[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, \
             markevery=mark_distance, label=methods[i])
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
plt.xlim(-1, X_limit)
plt.ylim(1e-3, 1e3)
plt.xlabel('Time (s)', fontproperties=font)
plt.ylabel('Optimality gap', fontproperties=font)
plt.legend(prop=font2, loc="upper right")
plt.savefig(dir_save_pdf + SM.setting + '-Time-F-gap-Byzan' + attack_type + str(optConfig['ByzantineSize'] / SM.m) + '.pdf', \
            format='pdf', dpi=4000, bbox_inches='tight')
# """

#-------------------------------Plot consensus errors in terms of epochs-------------------------------#
Consensus_error = []
for i in range(len(methods)):
    # Load the consensus errors of all tested algorithms
    tmp_consensus_error = np.loadtxt(dir_save_txt + SM.setting + '-consensus-error-' + methods[i] + attack_type + pn_setting+\
                                     str(optConfig['ByzantineSize'] / SM.m) + '.txt', dtype=float,
                                     delimiter=None)
    Consensus_error.append(tmp_consensus_error)

plt.figure(3, figsize=FigSize)
for i in range(len(methods)):
    plt.plot(Consensus_error[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, \
             markevery=mark_distance, label=methods[i])
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
plt.xlim(-2, X_limit)

# plt.ylim(1e-12, 1e3)  # ZSA
# plt.ylim(1e-6, 1e5)  # GA
plt.ylim(1e-4, 1e9)  # SVA

plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Consensus error', fontproperties=font)
# plt.legend( prop=font2, bbox_to_anchor=(0.67, 0.57) ) # ZSA
# plt.legend(prop=font2, bbox_to_anchor=(0.67, 0.46) ) # GA
plt.legend( prop=font2, bbox_to_anchor=(0.67, 0.3) )  # SVA
plt.savefig(dir_save_pdf + SM.setting + '-Epochs-consensus-error-Byzan' + attack_type
            + str(optConfig['ByzantineSize'] / SM.m) + '.pdf', format='pdf', dpi=4000, bbox_inches='tight')

'''
#-------------------------------Plot consensus error in terms of time-------------------------------#
plt.figure(4, figsize=FigSize)
# axes1 = plt.figure( figsize=FigSize )
""" Plot each line at a time:
plt.plot(time_Prox_Prox_DBRO_SAGA, F_gap_Prox_Prox_DBRO_SAGA, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
plt.plot(time_Prox_DBRO_LSVRG, F_gap_Prox_DBRO_LSVRG, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
"""
for i in range(len(methods)):
    plt.plot(time_list[i], Consensus_error[i], color=colors[i], marker=markers[i], linewidth=line_width,
             markersize=Marker_Size, markevery=mark_distance, label=methods[i])
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
plt.xlim(-1, X_limit)
plt.ylim(1e-7, 1e5)
plt.xlabel('Time (s)', fontproperties=font)
plt.ylabel('Consensus error', fontproperties=font)
plt.legend(prop=font2, loc="upper right")
plt.savefig(dir_save_pdf + SM.setting + '-Time-consensus-error-Byzan' + attack_type + str(optConfig['ByzantineSize'] / SM.m) + '.pdf', \
    format='pdf', dpi=4000, bbox_inches='tight')
#'''

#-------------------------------Plot testing accuracy in terms of epochs-------------------------------#
"""Load results of testing accuracy"""
Testing_accuracy = []
for i in range(len(methods)):
    # Load the testing accuracy of all tested algorithms
    tmp_testing_accuracy = np.loadtxt(dir_save_txt + SM.setting + '-acc-' + methods[i] + attack_type + pn_setting +
                                      str( optConfig['ByzantineSize']/SM.m ) + '.txt', dtype=float, delimiter=None)
    Testing_accuracy.append(tmp_testing_accuracy)

plt.figure(5, figsize=FigSize)
for i in range(len(methods)):
    plt.plot(Testing_accuracy[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size, markevery=mark_distance,\
             label=methods[i])
plt.grid(True)
plt.tick_params(pad=0, labelsize='large', width=3)
if optConfig['ByzantineSize'] == 0:
    attack_type = '-No-Attacks-'
    attacks = 'No Byzantine agents'
else:
    attack = str(Attack.__name__).replace('_', '-')
    attacks = attack.replace('-a', ' a')
plt.title(attacks, fontproperties=font)
plt.xlim(-2, X_limit)

# plt.ylim(0.040, 0.935)  # ZSA
# plt.ylim(0.05, 0.940)   # GA
plt.ylim(0.06, 0.935)    # SVA
# plt.yticks(list(plt.yticks()[0]) + [0.9])
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Testing accuracy', fontproperties=font)
# plt.legend(prop=font2, bbox_to_anchor=(0.67, 0.083))  # ZSA
# plt.legend(prop=font2, bbox_to_anchor=(1.0, 0.502))  # GA
plt.legend(prop=font2, bbox_to_anchor=(0.67, 0.071))  # SVA
plt.savefig(dir_save_pdf + SM.setting + '-Epochs-testing-accuracy-Byzan' + attack_type +
            str( optConfig['ByzantineSize']/SM.m ) + '.pdf', format='pdf', dpi=4000, bbox_inches='tight')

'''
#-------------------------------Plot testing accuracy in terms of time-------------------------------#
plt.figure(6, figsize=FigSize)
# axes1 = plt.figure( figsize=FigSize )
""" Plot each line at a time:
plt.plot(time_Prox_Prox_DBRO_SAGA, F_gap_Prox_Prox_DBRO_SAGA, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
plt.plot(time_Prox_DBRO_LSVRG, F_gap_Prox_DBRO_LSVRG, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
labels = ['NIDS', 'PMGT_LSVRG', 'PMGT_SAGA', 'Prox_BRIDGE_T', 'Prox_BRIDGE_M', 'Prox_BRIDGE_K', 'Prox_BRIDGE_B', \
          'Prox_GeoMed', 'Prox_DBRO_LSVRG', 'Prox_DBRO_SAGA']
"""
for i in range(len(methods)):
    # for i in []:
    plt.plot(time_list[i], Testing_accuracy[i], color=colors[i], marker=markers[i], linewidth=line_width, markersize=Marker_Size,\
            markevery=mark_distance, label=methods[i])
plt.grid(True)
plt.tick_params( pad = 0, labelsize='large', width=3 )
if optConfig['ByzantineSize'] == 0:
    attack_type = '-No-Attacks-'
    attacks = 'No Byzantine agents'
else:
    attack = str( Attack.__name__ ).replace('_', '-')
    attacks = attack.replace('-a', ' a')
plt.title(attacks, fontproperties=font)
plt.xlim(-1, X_limit)
plt.ylim(0.6, 1)
plt.xlabel('Time (s)', fontproperties=font)
plt.ylabel('Testing accuracy', fontproperties=font)
plt.legend(prop=font2, loc="lower right")
plt.savefig(dir_save_pdf + SM.setting + '-Time-testing-accuracy-Byzan' + attack_type + str( optConfig['ByzantineSize']/SM.m ) + '.pdf',\
            format = 'pdf', dpi = 4000, bbox_inches='tight')
# '''

# generate Byzantine and reliable  graphs
"""-------------------------------Plot the Byzantine graphs-------------------------------"""
pos = nx.random_layout(optConfig['ByzantineNetwork'])
labels = {x: x for x in optConfig['ByzantineNetwork'].nodes}
plt.figure(7, figsize=FigSize)
nx.draw_networkx_labels(optConfig['ByzantineNetwork'], pos, labels, font_size=8.5, font_color='w')
# a = '''
nodes = {
    '#FFA326': optConfig['ReliableSet'],
    '#FF0026': optConfig['ByzantineSet'],
}
for node_color, nodelist in nodes.items():
    nx.draw_networkx_nodes(optConfig['ByzantineNetwork'], pos, node_size=120, nodelist=nodelist, node_color=node_color)

edge_list = {x: x for x in optConfig['ByzantineNetwork'].edges}
# for i, j in edge_list:
#     G.add_edge(i, j)
nx.draw_networkx_edges(optConfig['ByzantineNetwork'], pos, edge_list, edge_color='#0073BD')
plt.axis('off')
plt.savefig(dir_save_pdf + SM.setting + '-Byzantine-Graph' + attack_type + pn_setting + str(optConfig['ByzantineSize'] / SM.m) + '.pdf',format='pdf', dpi=4000, bbox_inches='tight')

plt.figure(8, figsize=FigSize)
nx.draw(optConfig['ReliableNetwork'], pos, with_labels=True, font_size=8.5, font_color='w', node_size=120, node_color='#FFA326', width=1, node_shape=None, alpha=None, edge_color='#0073BD')
# nx.draw(G, pos, with_labels=False, node_size=120, node_color='#FFA326', node_shape=None, alpha=None, linewidths=8, edge_color = '#0073BD')
plt.axis('off')
plt.savefig(dir_save_pdf + SM.setting + '-Reliable-Graph' + attack_type +
            str(optConfig['ByzantineSize'] / SM.m) + '.pdf', format='pdf', dpi=4000, bbox_inches='tight')
plt.show()
plt.close()

# # 按照loss_an_epoch = computation_cost + storage_cost
# # epoch_iter = []
# # loss_x = []r
# # iter = []
# # for t in range(epochs+1):
# #     epoch_iter.append( t )
# #     loss_x.append( t*(b+1) )
# #
# # for k in range(epochs+1):
# #     iter.append( k )

# # Plot optimality gap in terms of decision variable
# # plt.figure(2)
# # plt.plot(x_res_Peng, color = colors[2], marker = markers[2], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
# # plt.plot(x_res_Prox_Prox_DBRO_SAGA, color = colors[3], marker = markers[3], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
# # plt.plot(x_res_Prox_DBRO_LSVRG, color = colors[4], marker = markers[4], linewidth=line_width, markersize=Marker_Size, markevery = mark_distance)
# # plt.grid(True)
# # plt.yscale('log')
# # plt.tick_params( pad = 0.8, labelsize='large', width=3 )
# # plt.title('Sign-flipping attacks', fontproperties=font)
# # plt.xlim(0, epochs)
# # plt.ylim(1e-10, 1e5)  #
# # plt.xlabel('Epochs', fontproperties=font)
# # plt.ylabel('Optimality Gap of Decision Variables', fontproperties=font)
# # plt.legend(('Peng', 'DBRO-SAGA', 'DBRO-LSVRG'), prop=font2)
# # plt.savefig('plots_ByzanMNIST.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')