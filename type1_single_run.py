# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:48:41 2016

@author: daniel
"""

from neuron import h
import d1msn as msn
import inhibitory_experiment as pe
import pickle
import parameters_inh as p
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import efel
# --- 1. Create a cell and other useful stuff

params  = "./params_dMSN.json"

with open('D1_71bestFit_updRheob.pkl', 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")

with open(params) as file:
    par = json.load(file)

cell_index = 34
variables = model_sets[cell_index]['variables']
cell = msn.MSN(params = params, variables = variables)

independent_dends = p.independent_dends
cell.increase_dend_res(independent_dends, 5)

for sec in cell.dendlist:
    print(sec.name(), "End = %f, Start = %f, L =%f, dist(%.2f) = %f, d = %.2f" % (h.distance(1.0, sec = sec),
                                      h.distance(0, sec = sec),
                                      h.distance(1.0, sec = sec) - h.distance(0, sec = sec),
                                      0.05,
                                      h.distance(0.05, sec = sec),
                                      sec.diam))

for sec in cell.somalist:
    print(sec.name(), "End = %f, Start = %f, L =%f, d = %.2f" % (h.distance(1, sec = sec),
                                      h.distance(0, sec = sec),
                                      h.distance(1, sec = sec) - h.distance(0, sec = sec),
                                      sec.diam))
#
#
# --- 2. Insert stimulation to cell

#independent_dends = [3, 5, 8, 12, 15, 22, 26, 35, 41, 47, 53, 57]
dend_record_list = [12, 5, 26, 22] # [1, 16, 19, 30, 31, 42, 49, 55]
dend_stim_list = []
plateau_cluster_list = [12]
inhibitory_cluster_dict = {'loc': [53],
                        'pos': [0.85],
                        'start': [p.inhibitory_burst_start ],
                        'end': [p.inhibitory_burst_end ] }

ex = pe.Inhibitory_Experiment('inhibitory_plasticity', cell)
ex.insert_synapses('inhibitory_plasticity')
ex.set_up_recording(dend_record_list)
ex.simulate()
ex.plot_results()

trace = {}
trace['T'] = ex.tv.to_python()
trace['V'] = ex.vs.to_python()
traces = [trace]
f = []

for t in np.arange(3000, p.simtime+1000, 1000):
    trace['stim_start'] = [t-1000]
    trace['stim_end'] = [t]
    freq = efel.get_feature_values(traces, ['mean_frequency'])[0]['mean_frequency']
    if freq == None:
        freq = 0
    f.append(freq[0])
print(f)

plt.show()
