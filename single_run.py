# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:48:41 2016

@author: daniel
"""

from neuron import h, rxd
import d1msn as msn
import plasticity_experiment as pe
import pickle
import parameters as p
import numpy as np
import json
import matplotlib.pyplot as plt

# --- 1. Create a cell and other useful stuff

params  = "./params_dMSN.json"

with open('D1_71bestFit_updRheob.pkl', 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")

with open(params) as file:
    par = json.load(file)

cell_index = 34
variables = model_sets[cell_index]['variables']
cell = msn.MSN(params = params, variables = variables)
all_dendrites = [i for i in range(0,len(cell.dendlist))]
cell.increase_dend_res(all_dendrites, 1)

# dist_dends = list(set(range(1, len(self.cell.dendlist)-1)) - set([0,1,6,7,11,16,19, 23, 25, 30, 31, 32, 39, 42, 43, 49, 54, 55]))
#
#
# --- 2. Insert stimulation to cell

#independent_dends = [3, 5, 8, 12, 15, 22, 26, 52]
dend_record_list = [3, 12, 22] #p.independent_dends
dend_stim_list = []

ex = pe.Plasticity_Experiment('record_ca', cell)
ex.insert_synapses('MSN')
ex.create_dopamine()

ex.set_up_recording(dend_record_list)
ex.simulate()
#ex.plot_cell()
ex.plot_results()
plt.show()
