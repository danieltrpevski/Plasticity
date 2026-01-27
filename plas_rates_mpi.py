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
from neuron import h

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

dend_record_list = [12]
output_freqs = []
caint = []
w_inh = []
vs = []
ipos = []

if rank == 0:
    # 2. Create tasks for the queue of tasks for parallel execution
    tasks = []
    e_rates = np.arange(1.5, 3.1, 0.1)
    trials = 50
    for r in e_rates:
        for trial in range(1, trials + 1):
            tasks.append([r, trial])

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
    r = t[0]; trial = t[1];
    print("Running trial %d for erate = %.1f." % (trial, r))
    p.new_erate = r

    ex = pe.Inhibitory_Experiment('inhibitory_plasticity', cell)
    ex.insert_synapses('inhibitory_plasticity')

    ex.set_up_recording(dend_record_list)
    ex.simulate()

    trace = {}
    trace['T'] = ex.tv.to_python()
    trace['V'] = ex.vs.to_python()
    traces = [trace]
    freqs = []
    for t in np.arange(2000, p.simtime+1000, 1000):
        trace['stim_start'] = [t-1000]
        trace['stim_end'] = [t]
        res = efel.getFeatureValues(traces, ['mean_frequency'])
        print(res)
        try:
            f = res[0]['mean_frequency'][0]
        except TypeError as e:
            f = 0
        freqs.append(f)
    output_freqs.append(freqs)
    caint.append(ex.caint[0].to_python())

    isyn = ex.get_synapse_list('adaptive_inhexp2syn', drive_type = 'i')
    ipos.append([h.distance(s.pos, sec = s.sec) for s in isyn])
    w_inh.append([s.obj.weight for s in isyn])
 #   vs.append(ex.vs.to_python())

    cell.esyn = []
    ex.estim = []
    ex.enc = []

    cell.isyn = []
    ex.istim = []
    ex.inc = []
    ex.delete_everything()

output_freqs = comm.gather(output_freqs, root = 0)
caint = comm.gather(caint, root = 0)
#vs = comm.gather(vs, root = 0)
ipos = comm.gather(ipos, root = 0)
w_inh = comm.gather(w_inh, root = 0)
# 5. Calculate and plot results
if rank == 0:
    res1 = []; res2 = []; res3 = []; res4 = []
    for o, c, i, w in zip(output_freqs, caint, ipos, w_inh):
        res1.extend(o); res2.extend(c); res3.extend(i); res4.extend(w)
    output_freqs = res1; caint = res2; ipos = res3; w_inh = res4

    res_dict = {'e_rates': e_rates.tolist(),
                'trials': trials,
                'output_freqs': output_freqs,
                'caint' : caint,
                'w_inh': w_inh,
                'ipos': ipos,
                't': p.simtime
               }
    to_save = json.dumps(res_dict)

    with open('./results/plas_rates.dat', 'w', encoding = 'utf-8') as f:
        json.dump(to_save, f)
