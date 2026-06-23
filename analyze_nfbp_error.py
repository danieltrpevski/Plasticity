# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:48:41 2016

@author: daniel
"""
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def ferror(vector, window):
    werr = []
    for i in range(0, int(len(vector) / window)):
        werr.append(sum(vector[i*window: (i+1)*window])/window)
    return werr

sns.set(font_scale = 1.5)
sns.set_style("ticks")

a = [['r', 's'], ['r', 's', 'y'], ['r', 's', 'b'], ['r', 's', 'y', 'b']]
b = [['y', 'b'], ['y', 'b', 'r'], ['y', 'b', 's'], ['y', 'b', 'r', 's']]

indices_four = [3, 7, 11, 12, 13, 14, 18, 22, 26, 27, 28, 29, 30]
all_indices = (np.arange(0, 31, 1)).tolist()
indices_three = list(set(all_indices) - set(indices_four))

result = []
for element in itertools.product(a,b):
    result.append(list(element))
result = result[0:len(result)-1]
for element in itertools.product(b,a):
    result.append(list(element))
print(result)

filename = './results/metaplasticity/nfbp_eta_10.dat'
# filename = './results/metaplasticity/nfbp_no_upper.dat'
with open(filename, 'r', encoding = 'utf-8') as f:
    fload  = json.load(f)
    res_dict = json.loads(fload)

weights = res_dict['weights']
error = res_dict['error']
tw = res_dict['tw']
# thresh_LTP = res_dict['thresh_LTP']
# thresh_LTD = res_dict['thresh_LTD']
# t = res_dict['t']
rewards_delivered = res_dict['rewards_delivered']
trials = res_dict['trials']
tw = np.asarray(tw)*1e-3
num_stimuli = len(rewards_delivered[0])
window_error = 20 #12

indices_four = [3, 7, 11, 12, 13, 14, 18, 22, 26, 27, 28, 29, 30]
all_indices = (np.arange(0, 31, 1)).tolist()
indices_three = list(set(all_indices) - set(indices_four))

tw = np.asarray(tw)*1e-3
er = np.reshape(error, (len(all_indices), trials, int(num_stimuli/window_error)))
e1 = er
e1 = (1 - e1)*100
for e in error:
    print(e)

for i,ind in enumerate(indices_three):
    r = result[ind]
    print(r)
    print(error[i*trials])
    pal = sns.color_palette(); color_inh = pal[3]; color_exc = pal[-2]
    j = 0
    for stimulus in r:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 10), sharex=True)

        for feature in stimulus:
            if feature == 'r':
                color = '#be0119'# 'red'
            elif feature == 'y':
                color = '#fec615' #'yellow'
            elif feature == 's':
                color = '#fc5a50'# 'darkred'
            elif feature == 'b':
                color = '#b79400' # 'darkyellow'
            for p in np.arange(10*j, 10*(j+1), 1):
                ax[0].plot(tw, weights[ind*trials][p], color = color)
                # ax[1].plot(tw, thresh_LTP[i*trials][p], color = color)
                # ax[2].plot(tw, thresh_LTD[i*trials][p], color = color)
                # axt.plot(tcai[i][p], cai_max[i][0][p], color = color, linestyle = '', marker = 'o')
            # axt.plot(tw, 0.005*np.ones(tw.shape), color = color)
            ax[1].set_xlabel('t'); #axt.set_xlabel('t')
            ax[0].set_ylabel('weights'); #axt.set_ylabel('threshold')
            # ax[1].set_ylabel('$\theta_{LTP}$'); #axt.set_ylabel('threshold')
            # ax[2].set_ylabel('$\theta_{LTD}$'); #axt.set_ylabel('threshold')
            j = j+1

    me1 = np.mean(e1[i], axis = (0));
    se1 = np.std(e1[i], axis = (0))

    plt.show()

xval = np.arange(window_error, num_stimuli+window_error, window_error)

fige = plt.figure()
axe = fige.add_subplot(111)

axe.plot(xval, me1, color = color_inh, linewidth = 3.0)
axe.fill_between(xval, me1-se1, me1+se1, color = color_inh, alpha = 0.2)
axe.set_xlabel('# of training examples')
axe.set_ylabel('Performance (%)')
for e in e1[i]:
    axe.plot(xval, e, color = 'gray', linewidth = 1.0)

plt.show()

me = np.mean(e1, axis = (0,1));
se = np.std(e1, axis = (0,1))

fig_te = plt.figure()
ax_te = fig_te.add_subplot(111)
ax_te.plot(xval, me, color = color_inh, linewidth = 3.0)
ax_te.fill_between(xval, me-se, me+se, color = color_inh, alpha = 0.2)
ax_te.set_xlabel('t')
ax_te.set_ylabel('Performance (%)')

plt.show()

# figd = plt.figure()
# axd = figd.add_subplot(111)
# axd.plot( rewards_delivered[i])
# axd.set_xlabel('t')
# axd.set_ylabel('dopamine')

# # for i in range(0, len(verror)):
# #     fige = plt.figure()
# #     axe = fige.add_subplot(111)
# #     # axe.plot(np.arange(12, 960+12, 12), (1 - np.asarray(error[i]))*100)
# #     # er2 = ferror(verror[i], 24);
# #     # axe.plot(np.arange(24, 960+24, 24), (1 - np.asarray(er2))*100)
# #     # if i in indices_four:
# #     #     color = 'lightgreen'
# #     # else:
# #     #     color = 'pink'
# #     er3 = ferror(verror[i], 48);
# #     print(er3)
# #     axe.plot(np.arange(48, 960+48, 48), (1 - np.asarray(er3))*100, color = color)
# #     performance.append(((1 - np.asarray(er3))*100).tolist())
#
# axe.set_xlabel('t')
# axe.set_ylabel('Performance (%)')

# four_performance = [performance[i] for i in indices_four]
# four_performance = np.reshape(four_performance, (13, np.arange(48, 960+48, 48).size))
# three_performance = [performance[i] for i in indices_three]
# mean_four_performance = np.mean(four_performance, axis = 0)
# mean_three_performance = np.mean(three_performance, axis = 0)
# std_four_performance = np.std(four_performance, axis = 0)
# std_three_performance = np.std(three_performance, axis = 0)
#
# axe.errorbar(np.arange(48, 960+48, 48), mean_four_performance, yerr = std_four_performance, color = 'green')
# axe.errorbar(np.arange(48, 960+48, 48), mean_three_performance, yerr = std_three_performance, color = 'purple')
