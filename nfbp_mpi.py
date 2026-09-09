# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:14:41 2020

@author: daniel
"""

from mpi4py import MPI
import d1msn as msn
import plasticity_experiment as pe
import dendstat as ds
import parameters as p
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
cell.increase_dend_res(p.independent_dends, 5)

a = [['r', 's'], ['r', 's', 'y'], ['r', 's', 'b'], ['r', 's', 'y', 'b']]
b = [['y', 'b'], ['y', 'b', 'r'], ['y', 'b', 's'], ['y', 'b', 'r', 's']]

a = [ ['r', 's', 'y']]; b = [['y', 'b', 'r']]

all_combinations = []
for element in itertools.product(a,b):
    all_combinations.append(list(element))
print(all_combinations)

weights = []
error = []
verror = []
training_set = []
rewards_delivered = []

if rank == 0:
    # 2. Create tasks for the queue of tasks for parallel execution
    tasks = []
    trials = 1
    for c in all_combinations:
        for trial in range(1, trials + 1):
              tasks.append([c, trial])

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
    input_comb = t[0]; trial = t[1];
    print("Running trial %d for input = %s" % (trial, input_comb))

    dendstatobj = ds.DendStat()
    input_dends = rnd.sample(p.independent_dends, 2)
    dendstatobj.dends = input_dends
    dendstatobj.dend_inputs = input_comb
    dendstatobj.dend_syns = [ [10]*len(input_comb[0]), [10]*len(input_comb[1]) ]

    ex = pe.Plasticity_Experiment('xor_hom_spillover', cell, dendstatobj = dendstatobj)
    ex.set_up_experiment()
    ex.set_up_recording(input_dends)
    ex.simulate()

    synlist = ex.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
    weights.append([s.ref_var_nmda.to_python() for s in synlist])
    error.append(ex.error(window = p.window_error)[-1])
    verror.append(ex.verror(window = p.window_error)[-2])
    training_set.append(ex.training_set)
    rewards_delivered.append(ex.rewards_delivered)

weights = comm.gather(weights, root = 0)
error = comm.gather(error, root = 0)
verror = comm.gather(verror, root = 0)
training_set = comm.gather(training_set, root = 0)
rewards_delivered = comm.gather(rewards_delivered, root = 0)

# 5. Calculate and plot results
if rank == 0:

    res1 = []; res2 = []; res3 = []; res4 = []; res5 = []
    for w,e,ve,d,h in zip(weights, error, verror, rewards_delivered, training_set):
        res1.extend(w); res2.extend(e); res3.extend(d); res4.extend(ts); res5.extend(ve);
    weights = res1; error = res2; rewards_delivered = res3; training_set = res4; verror = res5


    res_dict = {'weights': weights,
                'error': error,
                'verror': verror,
                'rewards_delivered': rewards_delivered,
                'training_set': training_set,
                'trials': trials,
                'tw': ex.tthresh.to_python(),
                't': ex.t.to_python()
               }
    to_save = json.dumps(res_dict)
    with open('./results/nfbp_test.dat', 'w', encoding = 'utf-8') as f:
        json.dump(to_save, f)
