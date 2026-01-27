# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:14:41 2020

@author: daniel
"""

from mpi4py import MPI
import d1msn as msn
import inhibitory_experiment as pe
import dendstat as ds
import parameters_inh as p
import numpy as np
import pickle
import json
import itertools
import random as rnd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
print("Number of processes = %d" % nprocs)
print("Before opening model sets")

params  = "./params_dMSN.json"

with open('D1_71bestFit_updRheob.pkl', 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")
print("Before creating a cell")

cell_index = 34

variables = model_sets[cell_index]['variables']
cell = msn.MSN(variables = variables)
cell.increase_dend_res(p.independent_dends, 3)

weights = []
w_source = []
hthresh_LTP = []
lthresh_LTD = []
window_error = []
full_error = []
dopamine = []
weights_inh = []

symmetric_combinations = [[['r', 's'], ['y', 'b', 'r', 's']],
[['r', 's', 'y'], ['y', 'b', 'r', 's']],
[['r', 's', 'b'], ['y', 'b', 'r', 's']],
[['r', 's', 'y', 'b'], ['y', 'b']],
[['r', 's', 'y', 'b'], ['y', 'b', 'r']],
[['r', 's', 'y', 'b'], ['y', 'b', 's']],
[['y', 'b'], ['r', 's', 'y', 'b']],
[['y', 'b', 'r'], ['r', 's', 'y', 'b']],
[['y', 'b', 's'], ['r', 's', 'y', 'b']],
[['y', 'b', 'r', 's'], ['r', 's']],
[['y', 'b', 'r', 's'], ['r', 's', 'y']],
[['y', 'b', 'r', 's'], ['r', 's', 'b']],
[['y', 'b', 'r', 's'], ['r', 's', 'y', 'b']],
]

inhibitory_combinations = [[[], ['r', 's']],
[['y'], ['r', 's']],
[['b'], ['r', 's']],
[['y', 'b'], []],
[['y', 'b'], ['r']],
[['y', 'b'], ['s']],
[[], ['y', 'b']],
[['r'], ['y', 'b']],
[['s'], ['y', 'b']],
[['r', 's'], []],
[['r', 's'], ['y']],
[['r', 's'], ['b']],
[['r', 's'], ['y', 'b']],
]

if rank == 0:
    # 2. Create tasks for the queue of tasks for parallel execution
    tasks = []
    trials = 19

    for c, ci in zip(symmetric_combinations, inhibitory_combinations):
        for trial in range(1, trials + 1):
              tasks.append([c, ci, trial])

    div, res = divmod(len(tasks), nprocs)
    counts = [div + 1 if p < res else div for p in range(nprocs)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]
    tasks = [tasks[starts[p]:ends[p]] for p in range(nprocs)]
else:
    tasks = None

tasks = comm.scatter(tasks, root=0)
for t in tasks:
    di = t[0]; dii = t[1]; trial = t[2];
    input_dends = rnd.sample(p.independent_dends, 2)
    print("Running trial %d for input = %s" % (trial, input_dends))

    dendstatobj = ds.DendStat()
    dendstatobj.dends = input_dends
    dendstatobj.dend_inputs = di
    dendstatobj.dend_syns = [[10]*len(di[0]), [10]*len(di[1])]
    dendstatobj.dend_inh_inputs = dii
    dendstatobj.dend_inh_syns = [[6]*len(dii[0]), [6]*len(dii[1])]

    ex = pe.Inhibitory_Experiment('nfbp_inh', cell, dendstatobj = dendstatobj)
    ex.set_up_experiment()
    ex.set_up_recording(dendstatobj.dends)
    ex.simulate()

    synlist = ex.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
    weights.append([s.ref_var_nmda.to_python() for s in synlist])
    e1, e2, e3, e4, e5 = ex.error(p.window_error)
    full_error.append(e4); window_error.append(e5)
    synlist_i = ex.get_synapse_list('adaptive2_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
    weights_inh.append([s.ref_var_w_inh.to_python() for s in synlist_i])

weights = comm.gather(weights, root = 0)
full_error = comm.gather(full_error, root = 0)
window_error = comm.gather(window_error, root = 0)
weights_inh = comm.gather(weights_inh, root = 0)

# 5. Calculate and plot results
if rank == 0:

    res1 = []; res2 = []; res3 = []; res4 = [];

    for w,ef,ew,wi in zip(weights, full_error, window_error, weights_inh):
        res1.extend(w); res2.extend(ef); res3.extend(ew); res4.extend(wi)
    weights = res1; full_error = res2; window_error = res3; weights_inh = res4

    res_dict = {'weights': weights,
                'weights_inh': weights_inh,
                'full_error': full_error,
                'window_error': window_error,
                'tw': ex.tthresh.to_python(),
                't': ex.tout.tolist(),
               }

    to_save = json.dumps(res_dict)
    with open('./results/nfbp_sym_diff_long.dat', 'w', encoding = 'utf-8') as f:
        json.dump(to_save, f)
