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

input_dends = p.input_dends
cell.increase_dend_res(input_dends, 5)

a = [['r', 's', 'y']]
b = [[]]

all_combinations = []
for element in itertools.product(a,b):
    all_combinations.append(list(element))

dend_record_list = input_dends
weights = []
thresh_LTP = []
thresh_LTD = []
error = []
verror = []
weights_agh = []
thresh_LTP_agh = []
thresh_LTD_agh = []

if rank == 0:
    # 2. Create tasks for the queue of tasks for parallel execution
    tasks = []
    trials = 50
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
    dendstatobj.dends = rnd.sample(p.independent_dends, 2)
    dendstatobj.dend_inputs = input_comb
    dendstatobj.dend_syns = [[10]*len(input_comb[0]), [10]*len(input_comb[1])]

    ex = pe.Plasticity_Experiment('fbp', cell, dendstatobj = dendstatobj)
    ex.set_up_experiment()
    ex.set_up_recording(dendstatobj.dends)
    ex.simulate()

    synlist = ex.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
    weights.append([s.ref_var_nmda.to_python() for s in synlist])
    thresh_LTP.append([s.ref_var_lthresh_LTP.to_python() for s in synlist])
    thresh_LTD.append([s.ref_var_lthresh_LTD.to_python() for s in synlist])
    error.append(ex.error(window = 20)[-1])
    verror.append(ex.error(window = 20)[-2])
    synlist_agh = ex.get_synapse_list('adaptive_hom_NMDA', clustered_flag = False)
    weights_agh.append([s.ref_var_nmda.to_python() for s in synlist_agh])
    thresh_LTP_agh.append([s.ref_var_lthresh_LTP.to_python() for s in synlist_agh])
    thresh_LTD_agh.append([s.ref_var_lthresh_LTD.to_python() for s in synlist_agh])

weights = comm.gather(weights, root = 0)
thresh_LTP = comm.gather(thresh_LTP, root = 0)
thresh_LTD = comm.gather(thresh_LTD, root = 0)
error = comm.gather(error, root = 0)
verror = comm.gather(verror, root = 0)
weights_agh = comm.gather(weights_agh, root = 0)
thresh_LTP_agh = comm.gather(thresh_LTP_agh, root = 0)
thresh_LTD_agh = comm.gather(thresh_LTD_agh, root = 0)

# 5. Calculate and plot results
if rank == 0:

    res1 = []; res2 = []; res3 = []; res4 = [];
    res5 = []; res6 = []; res7 = []; res8 = [];
    for w,e,ve,tp,td,wa,tpa,tda in zip(weights, error, verror,
                                       thresh_LTP, thresh_LTD,
                                       weights_agh, thresh_LTP_agh, thresh_LTD_agh):
        res1.extend(w); res2.extend(e); res3.extend(ve); res4.extend(tp);
        res5.extend(td); res6.extend(wa); res7.extend(tpa); res8.extend(tda);
    weights = res1; error = res2; verror = res3;
    thresh_LTP = res4; thresh_LTD = res5;
    weights_agh = res6; thresh_LTP_agh = res7; thresh_LTD_agh = res8

    res_dict = {'weights': weights,
                'weights_agh': weights_agh,
                'thresh_LTP': thresh_LTP,
                'thresh_LTD': thresh_LTD,
                'error': error,
                'verror': verror,
                'trials': trials,
                'thresh_LTP_agh': thresh_LTP_agh,
                'thresh_LTD_agh': thresh_LTD_agh,
                'tw': ex.tthresh.to_python(),
               }
    to_save = json.dumps(res_dict)
    with open('./results/fbp.dat', 'w', encoding = 'utf-8') as f:
        json.dump(to_save, f)
