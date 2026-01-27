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

three_feature_combinations = [[['r', 's'], ['y', 'b']],
[['r', 's'], ['y', 'b', 'r']],
[['r', 's'], ['y', 'b', 's']],
[['r', 's', 'y'], ['y', 'b']],
[['r', 's', 'y'], ['y', 'b', 'r']],
[['r', 's', 'y'], ['y', 'b', 's']],
[['r', 's', 'b'], ['y', 'b']],
[['r', 's', 'b'], ['y', 'b', 'r']],
[['r', 's', 'b'], ['y', 'b', 's']],
[['y', 'b'], ['r', 's']],
[['y', 'b'], ['r', 's', 'y']],
[['y', 'b'], ['r', 's', 'b']],
[['y', 'b', 'r'], ['r', 's']],
[['y', 'b', 'r'], ['r', 's', 'y']],
[['y', 'b', 'r'], ['r', 's', 'b']],
[['y', 'b', 's'], ['r', 's']],
[['y', 'b', 's'], ['r', 's', 'y']],
[['y', 'b', 's'], ['r', 's', 'b']]
]

inhibitory_combinations = [
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']],
[['r', 'y', 's', 'b'], ['r', 'y', 's', 'b']]
]
# indices_four = [3, 7, 11, 12, 13, 14, 18, 22, 26, 27, 28, 29, 30]
# all_indices = (np.arange(0, 31, 1)).tolist()
# indices_three = list(set(all_indices) - set(indices_four))
filename1 = './results/nfbp_upto3_diff.dat'
with open(filename1, 'r', encoding = 'utf-8') as f:
    fload  = json.load(f)
    res_dict = json.loads(fload)

weights = res_dict['weights']
weights_inh = res_dict['weights_inh']
# hthresh_LTP = res_dict['hthresh_LTP']
# lthresh_LTD = res_dict['lthresh_LTD']
# rewards_delivered = res_dict['rewards_delivered']
error = res_dict['window_error']
tw = res_dict['tw']
t = res_dict['t']
# trials = res_dict['trials']
trials = 13

filename2 = './results/nfbp_upto3_diff_only_exc.dat'
with open(filename2, 'r', encoding = 'utf-8') as f:
    fload  = json.load(f)
    res_dict = json.loads(fload)
weightse= res_dict['weights']
# weightse_inh = res_dict['weights_inh']
# hthreshe_LTP = res_dict['hthresh_LTP']
# lthreshe_LTD = res_dict['lthresh_LTD']
# rewards_delivered = res_dict['rewards_delivered']
errore = res_dict['window_error']

new_error = []
new_errore = []
for e in error:
    for i in range(0,int(1200/20)):
        new_error.append(sum(e[i*20: (i+1)*20])/20)
for e in errore:
    for i in range(0,int(1200/20)):
        new_errore.append(sum(e[i*20: (i+1)*20])/20)
error = new_error
errore = new_errore

tw = np.asarray(tw)*1e-3

e1 = np.reshape(error, (18, trials, int(1200/20)))
e2 = np.reshape(errore, (18, trials, int(1200/20)))
e1 = (1 - e1)*100
e2 = (1 - e2)*100
me1 = np.mean(e1, axis = (0,1)); se1 = np.std(e1, axis = (0,1))
me2 = np.mean(e2, axis = (0,1)); se2 = np.std(e2, axis = (0,1))

fig = plt.figure()
ax = fig.add_subplot(111)
xval = np.arange(20, 1200+20, 20)

pal = sns.color_palette(); color_inh = pal[3]; color_exc = pal[-2]
ax.plot(xval, np.ones(len(xval))*87.5, color = 'gray', linestyle = '--')
ax.plot(xval, me1, color = color_inh, linewidth = 3.0)
ax.fill_between(xval, me1-se1, me1+se1, color = color_inh, alpha = 0.2)
ax.plot(xval, me2, color = color_exc, linewidth = 3.0)
ax.fill_between(xval, me2-se2, me2+se2, color = color_exc, alpha = 0.2)
ax.set_xlabel('Number of patterns')
ax.set_ylabel('Performance (%)')
plt.show()

for i,r in enumerate(three_feature_combinations):
    print(r)
    j = 0
    # fig_ba, axes_ba = plt.subplots(3, 1, sharex = True)
    # axes_ba[0].plot(vdlist[i][0]);
    # axes_ba[1].plot(vdlist[i][1]);
    # axes_ba[2].plot(vs[i]);

    for stimulus in r:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # figt = plt.figure()
        # axt = figt.add_subplot(111)

        for feature in stimulus:
            if feature == 'r':
                color = 'red'
            elif feature == 'y':
                color = 'gold'
            elif feature == 's':
                color = 'darkred'
            elif feature == 'b':
                color = 'orange'
            r = np.arange(10*j, 10*(j+1), 1)

            for p in r:
                # print("%s, %s, %s" % (feature, w_source[i][p], color))
                ax.plot(tw, weights[i*trials][p], color = color)
                # axt.plot(tw, hthresh_LTP[i][p], color = color)
                # axt.plot(tcai[i][p], cai_max[i][0][p], color = color, linestyle = '', marker = 'o')
            # axt.plot(tw, 0.005*np.ones(tw.shape), color = color)
            ax.set_xlabel('t');
            ax.set_ylabel('weights');
            j = j+1

    jj = 0
    for si in inhibitory_combinations[i]:

        figi = plt.figure()
        axi = figi.add_subplot(111)
        axi.set_xlabel('t');
        axi.set_ylabel('$w_{inh}$');
        axi.set_title('jj = %d' % jj);
        for feature in si:
            if feature == 'r':
                color = 'red'
            elif feature == 'y':
                color = 'gold'
            elif feature == 's':
                color = 'darkred'
            elif feature == 'b':
                color = 'orange'

            ri = np.arange(5*jj, 5*(jj+1), 1)

            for p in ri:
                axi.plot(tw, weights_inh[i*trials][p], color = color)
            jj = jj+1

    fige = plt.figure()
    axe = fige.add_subplot(111)
    axe.plot(xval_l, np.asarray(error[i])*100)
    axe.set_xlabel("number of patterns");
    axe.set_ylabel("Performance (%)")
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
