# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:53:21 2019

@author: daniel
"""

from neuron import h
import d1msn as msn
#import iMSN
import inhibitory_experiment as pe
import parameters_inh as p
import pickle
import dendstat as ds

# --- 1. Create a cell and other useful stuff
dMSN_library = 'D1_71bestFit_updRheob.pkl'
iMSN_library = 'D2_34bestFit_updRheob.pkl'
with open(dMSN_library, 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")

cell_ID = 34
variables = model_sets[cell_ID]['variables']
cell = msn.MSN(variables = variables)
#cell = iMSN.iMSN(variables = variables)
for d in p.input_dends:
    cell.dendlist[d].nseg *=5

for sec in cell.dendlist:
    print(sec.name(), "%f, %f, %f, d = %.2f" % (h.distance(1, sec = sec),
                                      h.distance(0, sec = sec),
                                      h.distance(1, sec = sec) - h.distance(0, sec = sec),
                                      sec.diam))

for sec in cell.somalist:
    print(sec.name(), "%f, %f, %f, d = %.2f" % (h.distance(1, sec = sec),
                                      h.distance(0, sec = sec),
                                      h.distance(1, sec = sec) - h.distance(0, sec = sec),
                                      sec.diam))


# --- 2. Insert stimulation to cell
dendstatobj = ds.DendStat()
dendstatobj.dends = [3, 15]
dendstatobj.dend_inputs = [['r', 'y', 's', 'b' ], ['r', 'y', 's', 'b']]
dendstatobj.dend_syns = [[10, 10, 10, 10], [10, 10, 10, 10]]
dendstatobj.dend_inh_inputs = [['y', 'b'], ['r', 's']]
dendstatobj.dend_inh_syns = [[6, 6], [6, 6]]

ex = pe.Inhibitory_Experiment('nfbp_inh', cell, dendstatobj = dendstatobj)
ex.set_up_experiment()
ex.set_up_recording(dendstatobj.dends)

ex.simulate()
#ex.gmax_derivs()
ex.plot_results()
#ex.write_results()
#ex.save_syn_weights(p.save_weights_file)
