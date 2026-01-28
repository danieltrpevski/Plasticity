# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:14:41 2020

@author: daniel
"""

from mpi4py import MPI
import d1msn as msn
import inhibitory_experiment as pe
import parameters_inh as p
import numpy as np
import pickle
import json
import efel
import random as rnd
import dendstat as ds

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

independent_dends = p.independent_dends
cell.increase_dend_res(independent_dends, 5)

rd1 = []; rd2 = []; rd3 = []
weights = []; theta_inh = []; theta_min_inh = []
tw = []; input_dends_list = []
num_inh_syns = 6; nis = num_inh_syns

if rank == 0:
    # 2. Create tasks for the queue of tasks for parallel execution
    tasks = []
    #i_weights = [0.75e-3, 1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3]
    trials = 128
    for trial in range(1, trials + 1):
        tasks.append([trial])

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
    trial = t[0];
    print("Running trial %d" % (trial))
    input_dends = rnd.sample(p.independent_dends, 3)
    dendstatobj = ds.DendStat()
    dendstatobj.dends = input_dends
    dendstatobj.dend_inputs = [['a', 'b', 'c' ], ['a', 'b', 'c' ], ['a', 'b', 'c' ]]
    dendstatobj.dend_syns = [[20, 15, 10], [10, 20, 15], [15, 10, 20]]
    dendstatobj.dend_inh_inputs = [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']]
    dendstatobj.dend_inh_syns = [[nis, nis, nis],
                                 [nis, nis, nis],
                                 [nis, nis, nis]]

    ex = pe.Inhibitory_Experiment('pattern_homo_bcm', cell, dendstatobj = dendstatobj)
    ex.set_up_experiment()
    ex.set_up_recording(input_dends)

    ex.simulate()

    input_dends_list.append(input_dends)
    tw = ex.tthresh.to_python()

    d1, d2, d3 = ex.peak_dend_voltage()
    rd1.append([d1['a'], d1['b'], d1['c']])
    rd2.append([d2['a'], d2['b'], d2['c']])
    rd3.append([d3['a'], d3['b'], d3['c']])

    synlist = ex.get_synapse_list('adaptive2_homo_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
    #weights.append([s.ref_var_w_inh.to_python() for s in synlist])
    #theta_inh.append([s.ref_var_theta_inh.to_python() for s in synlist])

    cell.esyn = []
    ex.estim = []
    ex.enc = []

    cell.isyn = []
    ex.istim = []
    ex.inc = []
    ex.delete_everything()

rd1 = comm.gather(rd1, root = 0)
rd2 = comm.gather(rd2, root = 0)
rd3 = comm.gather(rd3, root = 0)
#weights = comm.gather(weights, root = 0)
#theta_inh = comm.gather(theta_inh, root = 0)
input_dends_list = comm.gather(input_dends_list, root = 0)
# 5. Calculate and plot results
if rank == 0:
    res1 = []; res2 = []; res3 = []; res4 = []
    for d1, d2, d3, inpd in zip(rd1, rd2, rd3, input_dends_list):
        res1.extend(d1); res2.extend(d2); res3.extend(d3); res4.extend(inpd)
    rd1 = res1; rd2 = res2; rd3 = res3; input_dends_list = res4

    res_dict = {'rd1': rd1,
                'rd2': rd2,
                'rd3': rd3,
 #               'weights': weights,
#                'theta_inh': theta_inh,
                'num_inh_syns': num_inh_syns,
                'trials': trials,
#                'tw': tw,
                'input_dends_list': input_dends_list
               }
    to_save = json.dumps(res_dict)

    with open('./results/pattern_homo_bcm_new.dat', 'w', encoding = 'utf-8') as f:
        json.dump(to_save, f)
