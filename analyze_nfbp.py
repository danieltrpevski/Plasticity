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
import parameters as p

def ferror(vector, window):
    werr = []
    for i in range(0, int(len(vector) / window)):
        werr.append(sum(vector[i*window: (i+1)*window])/window)
    return werr

sns.set(font_scale = 2.0)
sns.set_style("ticks")

a = [['r', 's'], ['r', 's', 'y'], ['r', 's', 'b'], ['r', 's', 'y', 'b']]
b = [['y', 'b'], ['y', 'b', 'r'], ['y', 'b', 's'], ['y', 'b', 'r', 's']]

# indices_four = [3, 7, 11, 12, 13, 14, 18, 22, 26, 27, 28, 29, 30]
# all_indices = (np.arange(0, 31, 1)).tolist()
# indices_three = list(set(all_indices) - set(indices_four))

result = []
for element in itertools.product(a,b):
    result.append(list(element))
result = result[0:len(result)-1]
for element in itertools.product(b,a):
    result.append(list(element))
print(result)

print("Loading data for analysis...")
filename = './results/metaplasticity/nfbp.dat'
with open(filename, 'r', encoding = 'utf-8') as f:
    fload  = json.load(f)
    res_dict = json.loads(fload)
print("Loading done!")

weights = res_dict['weights']
weights_agh = res_dict['weights_agh']
lthresh_LTP = res_dict['lthresh_LTP']
# rewards_delivered = res_dict['rewards_delivered']
error = res_dict['error']
tw = res_dict['tw']
# t = res_dict['t']
trials = res_dict['trials']
tw = np.asarray(tw)*1e-3

indices_four = [3, 7, 11, 12, 13, 14, 18, 22, 26, 27, 28, 29, 30]
all_indices = (np.arange(0, 31, 1)).tolist()
indices_three = list(set(all_indices) - set(indices_four))

er = np.reshape(error, (len(result), trials, int(400/20))) #int(640/12)
e1 = er[indices_three]
e1 = (1 - e1)*100
me1 = np.mean(e1, axis = (0,1)); se1 = np.std(e1, axis = (0,1))

e2 = er[indices_four]
e2 = (1 - e2)*100
me2 = np.mean(e2, axis = (0,1)); se2 = np.std(e2, axis = (0,1))

fig = plt.figure(figsize = (p.fig_width, p.fig_height))
ax = fig.add_subplot(111)
# xval = np.arange(12, 640+12, 12)
xval = np.arange(20, 420, 20)

pal = sns.color_palette(); #color_inh = pal[3]; color_exc = pal[-2]
ax.plot(xval[0:xval.size], np.ones((xval.size,))*87.5, color = 'gray', linewidth = 2.0, linestyle = '--')
# ax.plot(xval[0:xval.size-1], np.ones((xval.size-1,))*100, color = 'gray', linewidth = 1.5, linestyle = '--')
ax.plot(xval[0:xval.size], me1, color = '#DD8452', linewidth = 3.0)
ax.fill_between(xval[0:xval.size], me1-se1, me1+se1, color = '#DD8452', alpha = 0.2)
ax.plot(xval[0:xval.size], me2, color = '#55A868', linewidth = 3.0)
ax.fill_between(xval[0:xval.size], me2-se2, me2+se2, color = '#55A868', alpha = 0.2)
ax.set_xlabel('# of patterns')
ax.set_xlim(0,420);
ax.set_xticks([0,200,400])

ax.set_ylabel('Score (%)')
ax.set_ylim([30, 105])
ax.set_yticks([50, 100])
plt.show()

for e in error:
    print(e)

mean_dict = {
              'xval': xval.tolist()[0:20],
              'me1': me1.tolist()[0:20],
              'se1': se1.tolist()[0:20],
              'me2': me2.tolist()[0:20],
              'se2': se2.tolist()[0:20]}
to_save = json.dumps(mean_dict)

# filename = './results/metaplasticity/performance_nfbp_rate_085.dat'
# with open(filename, 'w', encoding = 'utf-8') as f:
#     json.dump(to_save, f)
