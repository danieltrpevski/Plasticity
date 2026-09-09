# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:48:41 2016

@author: daniel
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import parameters as p

def ferror(vector, window):
    werr = []
    for i in range(0, int(len(vector) / window)):
        werr.append(sum(vector[i*window: (i+1)*window])/window)
    return werr

sns.set(font_scale = 2.0)
sns.set_style("ticks")

filename = './results/metaplasticity/Erev_NMDA_0/fbp_dist_st.dat'
with open(filename, 'r', encoding = 'utf-8') as f:
    fload  = json.load(f)
    res_dict = json.loads(fload)

weights = res_dict['weights']
error = res_dict['error']
verror = res_dict['verror']
tw = res_dict['tw']
trials = res_dict['trials']
# thresh_LTP = res_dict['thresh_LTP']
# thresh_LTD = res_dict['thresh_LTD']
num_syns_per_feature = int(len(weights[0])/3)
tw = np.asarray(tw)*1e-3

error = []
window = 20
training_set_size = len(verror[0])
performance = []
figet = plt.figure(figsize = (p.fig_width, p.fig_height))
axet = figet.add_subplot(111)

x_values = np.arange(window, training_set_size + 20, window)
for i in range(0, len(verror)):
    er3 = ferror(verror[i], window);
    print(er3)
    error.append(er3)
    performance.append(((1 - np.asarray(er3))*100).tolist())

performance = np.reshape(performance, (trials, x_values.size))
mean_performance = np.mean(performance, axis = 0)
std_performance = np.std(performance, axis = 0)

color = "#4C72B0"
axet.plot(x_values, mean_performance)
axet.fill_between(x_values, mean_performance - std_performance, mean_performance + std_performance, color=color, alpha = 0.5)
axet.plot(x_values, np.ones(x_values.size)*75, linestyle = '--', linewidth = 2 ,color = 'gray')

axet.set_xlabel('# of patterns')
axet.set_xlim(0,training_set_size);
xlim = int(training_set_size/100)*100
axet.set_xticks([0,int(xlim/2),xlim])
axet.set_ylabel('Score (%)')
axet.set_ylim(0,105);
axet.set_yticks([0, 50,100]);
#
# filename = './results/metaplasticity/Erev_NMDA_0/fbp_rl_etam4.dat'
# with open(filename, 'r', encoding = 'utf-8') as f:
#     fload  = json.load(f)
#     res_dict = json.loads(fload)
#
# weights = res_dict['weights']
# error = res_dict['error']
# verror = res_dict['verror']
# tw = res_dict['tw']
# trials = res_dict['trials']
# thresh_LTP = res_dict['thresh_LTP']
# thresh_LTD = res_dict['thresh_LTD']
# num_syns_per_feature = int(len(weights[0])/3)
# tw = np.asarray(tw)*1e-3
# num_stimuli = len(verror[0])
#
# color = '#937860'
# er = np.reshape(error, (trials, int(num_stimuli/20)))
# er =  (1 - er)*100
# me1 = np.mean(er, axis = (0)); se1 = np.std(er, axis = (0))
# axet.plot(x_values, me1, color = color)
# axet.fill_between(x_values, me1 - se1, me1 + se1, color=color, alpha = 0.5)

plt.show()

# r = ['r', 's', 'y']
# for i in range(0, trials):
#     print(error[i])
#     j = 0
#
#     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True)
#
#     for feature in r:
#         if feature == 'r':
#             color = '#be0119'# 'red'
#         elif feature == 'y':
#             color = '#fec615' #'yellow'
#         elif feature == 's':
#             color = '#fc5a50'# 'darkred'
#         elif feature == 'b':
#             color = '#b79400' # 'darkyellow'
#         for p in np.arange(num_syns_per_feature*j, num_syns_per_feature*(j+1), 1):
#             ax[0].plot(tw, weights[i][p], color = color)
#             ax[1].plot(tw, thresh_LTP[i][p], color = color)
#             ax[2].plot(tw, thresh_LTD[i][p], color = color)
#         j = j+1
#     ax[2].set_xlabel('t'); #axt.set_xlabel('t')
#     ax[0].set_ylabel('weights'); #axt.set_ylabel('threshold')
#     ax[1].set_ylabel('$\theta_{LTP}$'); #axt.set_ylabel('threshold')
#     ax[2].set_ylabel('$\theta_{LTD}$'); #axt.set_ylabel('threshold')
#
#
#     figp = plt.figure()
#     axp = figp.add_subplot(111)
#     axp.plot(x_values, performance[i])
#     axp.set_xlabel("# of patterns")
#     axp.set_ylabel("Performance (%)")
#
#     plt.show()

# for i, wa in enumerate(weights_agh):
#     print(error[i])
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     figt = plt.figure()
#     axt = figt.add_subplot(111)
#
#     figtt = plt.figure()
#     axtt = figtt.add_subplot(111)
#
#     for j in range(0, 45):
#         if j < 15:
#             color = '#be0119' #RED
#         elif j < 30:
#             color = 'gold'
#         elif j < 45:
#             color = '#fc5a50' #DARK RED
#
#         ax.plot(tw, wa[j], color = color)
#         # axt.plot(tw, thresh_LTD_agh[i][j], color = color)
#         axt.plot(tw, thresh_LTP_agh[i][j], color = color)
#         axtt.plot(tw, thresh_LTD_agh[i][j], color = color)
#
#         ax.set_xlabel('t (s)'); axt.set_xlabel('t (s)')
#         ax.set_ylabel('weights'); axt.set_ylabel('LTP threshold')
#         axtt.set_xlabel('t (s)'); axtt.set_ylabel('LTD threshold')
#
#     fige = plt.figure()
#     axe = fige.add_subplot(111)
#     axe.plot(x_values, (1 - np.asarray(error[i]))*100, color = color)
#     axe.plot(x_values, np.ones(x_values.size)*75, linestyle = '--', linewidth = 2 ,color = 'grey')
#     axe.set_ylim(0, 102)
#     plt.show()

mean_dict = {
              'xval': x_values.tolist()[0:20],
              'me': mean_performance.tolist()[0:20],
              'se': std_performance.tolist()[0:20]}
to_save = json.dumps(mean_dict)

# filename = './results/metaplasticity/performance_fbp_rate17.dat'
# with open(filename, 'w', encoding = 'utf-8') as f:
#     json.dump(to_save, f)
