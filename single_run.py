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
# --- 1. Create a cell and other useful stuff

params  = "./params_dMSN.json"

with open('D1_71bestFit_updRheob.pkl', 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")

with open(params) as file:
    par = json.load(file)

cell_index = 34
variables = model_sets[cell_index]['variables']
cell = msn.MSN(params = params, variables = variables)

for d in p.input_dends:
    cell.dendlist[d].nseg *=5

for sec in cell.dendlist:
    print(sec.name(), "%f, %f, %f, %f, d = %.2f" % (h.distance(1.0, sec = sec),
                                      h.distance(0, sec = sec),
                                      h.distance(0.35, sec = sec) - h.distance(0.25, sec = sec),
                                      h.distance(0.25, sec = sec),
                                      sec.diam))

for sec in cell.somalist:
    print(sec.name(), "%f, %f, %f, d = %.2f" % (h.distance(1, sec = sec),
                                      h.distance(0, sec = sec),
                                      h.distance(1, sec = sec) - h.distance(0, sec = sec),
                                      sec.diam))

# dist_dends = list(set(range(1, len(self.cell.dendlist)-1)) - set([0,1,6,7,11,16,19, 23, 25, 30, 31, 32, 39, 42, 43, 49, 54, 55]))
#
#
# --- 2. Insert stimulation to cell

#independent_dends = [3, 5, 8, 12, 15, 22, 26, 35, 41, 47, 53, 57]
dend_record_list = p.independent_dends
dend_stim_list = []
plateau_cluster_list = [52]
inhibitory_cluster_dict = {'loc': [53],
                        'pos': [0.85],
                        'start': [p.inhibitory_burst_start ],
                        'end': [p.inhibitory_burst_end ] }


# istim = h.IClamp(cell.somalist[0](0.5))
# istim.dur = 25
# istim.amp = 0.9
# istim.delay = 100

#istim = h.IClamp(cell.somalist[0](0.5))
#istim.dur = 5
#istim.amp = 0.5
#istim.delay = 1000
#
# istim2 = h.IClamp(cell.somalist[0](0.5))
# istim2.dur = 2000
# istim2.amp = 0.15

#heads = [h.head for h in cell.spines]
#necks = [h.neck for h in cell.spines]
#allsecslist = []
#allsecslist.extend(cell.dendlist)
#allsecslist.extend(heads)
#allsecslist.extend(necks)
#allsecslist.extend(cell.somalist)
#
#Dca = 200
#exc = rxd.Region(allsecslist, name='exc', nrn_region='o', geometry=rxd.Shell(1, 2))
#membrane = rxd.Region(allsecslist, name='mem', geometry=rxd.membrane())
#regions = rxd.Region(allsecslist, nrn_region= 'i')
#ca_nmda = rxd.Species(regions, name='ca_nmda', d = Dca, charge=2, initial = 5e-5, atolscale=1e-6)
#ca_nmda_exc = rxd.Species(exc, name='ca_nmda', d = Dca, charge=2, initial = 5e-5, atolscale=1e-6)
#calbindin = rxd.Species(regions, name='calbindin', d = 66, initial = .15, atolscale=1e-6)
#CaMN = rxd.Species(regions, name='CaMN', d = 66, initial = .055, atolscale=1e-6)
#CaMC = rxd.Species(regions, name='CaMC', d = 66, initial = .055, atolscale=1e-6)
#fixed = rxd.Species(regions, name='fixed', d = 0, initial = 2.5, atolscale=1e-6)
#
#ca_nmda_calbindin = rxd.Species(regions, name='ca_nmda_calbindin', d = 66, initial = 0, atolscale=1e-6)
#ca_nmda_CaMN = rxd.Species(regions, name='ca_nmda_CaMN', d = 66, initial = 0, atolscale=1e-6)
#ca_nmda_CaMC = rxd.Species(regions, name='ca_nmda_CaMC', d = 66, initial = 0, atolscale=1e-6)
#ca_nmda_fixed = rxd.Species(regions, name='ca_nmda_fixed', d = 0, initial = 0, atolscale=1e-6)
#
#kf_ca_nmda_calbindin = 28; kr_ca_nmda_calbindin = 0.7e-6 *28
#kf_ca_nmda_CaMN = 100; kr_ca_nmda_CaMN = 1e-3
#kf_ca_nmda_CaMC = 6; kr_ca_nmda_CaMC = 1.5e-6*6
#kf_ca_nmda_fixed = 400; kr_ca_nmda_fixed = 100*400e-3
#
#R_ca_nmda_calbindin = rxd.Reaction(ca_nmda + calbindin, ca_nmda_calbindin, kf_ca_nmda_calbindin, kr_ca_nmda_calbindin)
#R_ca_nmda_CaMN = rxd.Reaction(ca_nmda + CaMN, ca_nmda_CaMN, kf_ca_nmda_CaMN, kr_ca_nmda_CaMN )
#R_ca_nmda_CaMC = rxd.Reaction(ca_nmda + CaMC, ca_nmda_CaMC, kf_ca_nmda_CaMC, kr_ca_nmda_CaMC)
#R_ca_nmda_fixed = rxd.Reaction(ca_nmda + fixed, ca_nmda_fixed, kf_ca_nmda_fixed, kr_ca_nmda_fixed)
#
#gserca = 1.9565
#Kserca = 0.1
#serca = rxd.MultiCompartmentReaction(ca_nmda[regions], ca_nmda_exc[exc],
#                                     gserca*(1e3*ca_nmda[regions])**2/(Kserca**2+(1e3*ca_nmda[regions])**2),
#                                     membrane=membrane,
#
#cell.insert_spines(plateau_cluster_list, p.cluster_start_pos,
#                   p.cluster_end_pos, num_spines = p.plateau_cluster_size)

ex = pe.Plasticity_Experiment('record_ca', cell)
ex.insert_synapses('MSN')
ex.create_dopamine()

ex.insert_synapses('my_spillover', plateau_cluster_list, deterministic = 0,
                   num_syns = 20, add_spine = 1, on_spine = 0)
#ex.insert_synapses('inhexpsyn_plateau', plateau_cluster_list, deterministic = 1,
#                   num_syns = p.inhibitory_cluster_size)
#cell.insert_spines(plateau_cluster_list, 0.3, 0.45, num_spines = 10)
#
#ex.insert_synapses('pf', plateau_cluster_list, deterministic = 0,
#                   num_syns = p.pf_input_size, add_spine = 0)
#
#ex.insert_synapses('input_syn', deterministic = 1, num_syns = 40,
#                   add_spine = 1)
cell.set_up_diffusion()
ex.set_up_diffusion()
ex.set_up_recording(dend_record_list)
ex.simulate()
#ex.plot_cell()
ex.plot_results()
import matplotlib.pyplot as plt
plt.show()
