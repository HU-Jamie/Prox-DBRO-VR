import numpy as np
from ByzantineNetwork import lr_0, optConfig, Attack, ByzantineSize
import pandas as pd


dir_save_txt = 'results/txtdata/'
method = 'PMGT-LSVRG'
method1 = 'PMGT-SAGA'
Setting = '-GA-'
pn_setting = 'L2-'
tmp_F_gap = np.loadtxt(dir_save_txt + lr_0.setting + '-F-gap-' + method1 + Setting + pn_setting +
                       str(optConfig['ByzantineSize'] / lr_0.m) + '.txt', dtype=float, delimiter=None)
tmp_F_gap = pd.DataFrame(tmp_F_gap)
tmp_F_gap.fillna(method = 'ffill', axis=0, inplace=True)
np.savetxt( dir_save_txt + lr_0.setting + '-F-gap-' + method1 + Setting + pn_setting + str(ByzantineSize / lr_0.m) + '.txt', tmp_F_gap )