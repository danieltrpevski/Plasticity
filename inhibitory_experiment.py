# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:36:50 2017

@author: daniel
"""

from neuron import h
import plasticity_experiment as pe
import parameters_inh as p
import time
import random as rnd
import numpy as np
import json
import re
import dendstat as dsd
import sys
import pickle
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

class Inhibitory_Experiment(pe.Plasticity_Experiment):

    def __init__(self, exptype, cell, dendstatobj = [], training_mode = p.training_mode):
        super(Inhibitory_Experiment, self).__init__(exptype, cell, dendstatobj = dendstatobj, training_mode = training_mode)

    def insert_synapses(self, syntype, syn_loc = [], deterministic = 0,
                        num_syns = p.distributed_input_size, add_spine = 0, on_spine = 0):
        if syntype in ['expsyn',  'inhexpsyn', 'inhexp2syn_caint']:
            if syntype in ['expsyn']:
                num_syns = p.distributed_input_size
            elif syntype == 'inhexpsyn':
                num_syns = p.inhibitory_cluster_size

            if syn_loc == []:
                for i in range (0,num_syns):
                    syn_loc.append(rnd.randint(0,len(self.cell.dendlist)-1))

            for loc in syn_loc:
                if syntype == 'expsyn_hom':
                    self.cell.dendlist[loc].insert('Hom_Cai')
                syn = self.cell.insert_synapse(syntype, self.cell.dendlist[loc], p.pos, add_spine = add_spine, on_spine = on_spine)
                self.add_input_generator(syn, syntype)

        elif syntype == 'input_syn':
            syn_loc = []
            for i in range(0, num_syns):
                syn_loc.append([rnd.randint(0, len(self.cell.dendlist)-1), rnd.uniform(0,1)])

            if deterministic == 1:
                spike_time = []
                for i in range(0, len(syn_loc)):
                    spike_time.append(rnd.uniform(p.distributed_input_start, p.distributed_input_end))

            counter = 0
            for loc in syn_loc:
                counter += 1
                syn1 = self.cell.insert_synapse('AMPA', self.cell.dendlist[loc[0]], loc[1],
                           add_spine = 0, on_spine = 0)
                syn2 = self.cell.insert_synapse('NMDA', self.cell.dendlist[loc[0]], loc[1],
                                           add_spine = 0, on_spine = 0)
#                 syn = self.cell.insert_synapse('glutamate', self.cell.dendlist[loc[0]], loc[1], add_spine = add_spine, on_spine = on_spine)
                if deterministic == 1:
#                    self.add_input_generator(syn, syntype, deterministic = deterministic, numsyn = counter, tstart = spike_time[counter-1])
                    self.add_input_generator(syn1, 'AMPA', deterministic = deterministic, numsyn = counter, tstart = spike_time[counter-1])
                    self.connect_input_generator(syn2, 'NMDA', syn1.stim[-1])
                else:
#                    self.add_input_generator(syn, syntype, deterministic = deterministic, numsyn = counter)
                    self.add_input_generator(syn1, 'AMPA', deterministic = deterministic, numsyn = counter)
                    self.connect_input_generator(syn2, 'NMDA', syn1.stim[-1])

        elif syntype == 'ramp':
            for loc in syn_loc:
                syn = self.cell.insert_synapse('expsyn',self.cell.dendlist[loc],  p.pos, add_spine = add_spine, on_spine = on_spine)
                self.add_input_generator(syn, syntype)

        elif syntype in ['noise_SPN', 'inhibitory_plasticity']:
            for dend in self.cell.dendlist:
                # Synapses according to Cheng et al. Experimental Neurobiology, 147:287-298 (1997)
                unit_length = 20.0 # Values reported in units per 20 microns in the study
                [exc_mean, exc_sem, inh_mean, inh_sem] = self.synapse_distribution(self.celltype, dend)
                # Insert excitatory synapses in this section,
                # This contains both an exponential and an NMDA synapse.
                stype = 'glutamate'
                if dend.nseg <= exc_mean:
                    freq_multiplier = exc_mean/dend.nseg
                    step = 1/dend.nseg
                    for i in range(0, dend.nseg):
                        pos = (i + (i+1))*step/2
                        self.helper_insert(stype, pos, dend, freq_multiplier)
                else:
                    num_exc_syn = int(dend.L/unit_length * rnd.gauss(exc_mean,exc_sem))
                    freq_multiplier = 1.0
                    for i in range(0,num_exc_syn):
                        pos = rnd.uniform(0,1)
                        self.helper_insert(stype, pos, dend, freq_multiplier)

                # Insert inhibitory synapses in this section
                if syntype == 'noise_SPN':
                    stype = 'inhexp2syn'
                elif syntype == 'inhibitory_plasticity':
                    stype = 'adaptive_inhexp2syn'

                if dend.nseg <= inh_mean:
                    freq_multiplier = inh_mean/dend.nseg
                    step = 1/dend.nseg
                    for i in range(0, dend.nseg):
                        pos = (i + (i+1))*step/2
                        self.helper_insert(stype, pos, dend, freq_multiplier)
                else:
                    num_inh_syn = int(dend.L/unit_length * rnd.gauss(inh_mean,inh_sem))
                    freq_multiplier = 1.0
                    for i in range(0,num_inh_syn):
                        pos = rnd.uniform(0,1)
                        self.helper_insert(stype, pos, dend, freq_multiplier)

        elif syntype in ['plateau_cluster', 'inhexpsyn_plateau', 'generalized_rule',
                         'spillover', 'spillover_test', 'no_spillover', 'my_spillover',
                          'no_spillover_stp', 'my_spillover_stp', 'inhexp2syn_plateau']:
            if syntype == 'plateau_cluster':
                syntype = 'tmGlut'
            elif syntype == 'spillover':
                syntype = 'adaptive_glutamate_test'
            elif syntype == 'my_spillover' or syntype == 'my_spillover_stp':
                self.exglusec.append(h.Section(name = 'exglusec%d' % len(self.exglusec)))
                self.exglu.append(h.IntFire1(self.exglusec[-1](0.5)))
                self.exglu[-1].tau = p.exglu_tau
                self.exglu[-1].refrac = p.session_length
            elif syntype == 'inhexp2syn_plateau':
                syntype = 'inhexp2syn_caint'
#                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', self.exglusec[-1].exglu)
            for num,loc in enumerate(syn_loc):
                syn_step = 1.0/num_syns
                cluster_start_pos = p.cluster_start_poss[p.independent_dends.index(loc)]
                cluster_end_pos = p.cluster_end_poss[p.independent_dends.index(loc)]
                for i in range(0, num_syns):
#                    pos = cluster_start_pos + (cluster_end_pos - cluster_start_pos)*i*syn_step
                    pos = cluster_end_pos - (cluster_end_pos - cluster_start_pos)*i*syn_step
                    if syntype == 'inhexpsyn_plateau':
                        if deterministic == 0:
                            start = rnd.uniform(p.inhibitory_burst_start, p.inhibitory_burst_end)
                        elif deterministic == 1:
                            start = p.inhibitory_burst_start + i*p.deterministic_interval
                        pos = p.inh_cluster_start_pos + (p.inh_cluster_end_pos - p.inh_cluster_start_pos)*i*syn_step
                    else:
                        if deterministic == 0:
                            start = rnd.uniform(p.plateau_burst_start, p.plateau_burst_end)
                        elif deterministic == 1:
                            start = p.plateau_burst_start + i*p.deterministic_interval
                    if not (syntype in ['spillover_test', 'no_spillover', 'my_spillover', 'no_spillover_stp',
                                        'my_spillover_stp']):
                        syn = self.cell.insert_synapse(syntype, self.cell.dendlist[loc],
                                                       pos, add_spine = add_spine, on_spine = on_spine)
                        self.add_input_generator(syn, syntype, deterministic = deterministic, numsyn = i, tstart = start)
                        syn.clustered_flag = True
                        if syntype == 'generalized_rule':
                            h.setpointer(h._ref_dopamine, 'dopamine', syn.obj)
                            h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn.obj)
                        elif syntype == 'adaptive_glutamate_test':
                            h.setpointer(h._ref_dopamine, 'dopamine', syn.obj)

                    elif syntype in ['spillover_test', 'my_spillover']:
                        syn1 = self.cell.insert_synapse('AMPA', self.cell.dendlist[loc], pos,
                                                   add_spine = add_spine, on_spine = on_spine)
                        self.add_input_generator(syn1, 'AMPA', deterministic = deterministic, numsyn = i, tstart = start)
                        if self.cell.spines != []:
                            spines = [s for s in self.cell.spines if s.parent == self.cell.dendlist[loc]]
                            spines[i].syn_on = 0
                        syn2 = self.cell.insert_synapse('NMDA', self.cell.dendlist[loc], pos,
                                                   add_spine = 0, on_spine = 1)
                        self.connect_input_generator(syn2, 'NMDA', syn1.stim[-1])
                        syn3 = self.cell.insert_synapse('NMDAe', self.cell.dendlist[loc], pos,
                                                               add_spine = 0, on_spine = 0)
                        syn1.clustered_flag = syn2.clustered_flag = syn3.clustered_flag = True

                        if syntype == 'spillover_test':
                            self.connect_input_generator(syn3, 'NMDAe', syn2.stim[-1], delay = p.delay_exnmda)
                        elif syntype == 'my_spillover':
                            self.connect_input_generator(self.exglu[-1], 't_exglu', syn1.stim[-1])
                            self.connect_input_generator(syn3, 's_exglu', self.exglu[-1])

                    elif syntype in ['my_spillover_stp']:
                        syn1 = self.cell.insert_synapse('AMPA_stp', self.cell.dendlist[loc], pos,
                                                   add_spine = add_spine, on_spine = on_spine)
                        self.add_input_generator(syn1, 'AMPA_stp', deterministic = deterministic, numsyn = i, tstart = start)
                        if self.cell.spines != []:
                            spines = [s for s in self.cell.spines if s.parent == self.cell.dendlist[loc]]
                            spines[i].syn_on = 0
                        syn2 = self.cell.insert_synapse('NMDA_stp', self.cell.dendlist[loc], pos,
                                                   add_spine = 0, on_spine = 1)
                        self.connect_input_generator(syn2, 'NMDA_stp', syn1.stim[-1])
                        syn3 = self.cell.insert_synapse('NMDAe', self.cell.dendlist[loc], pos,
                                                               add_spine = 0, on_spine = 0)
                        syn1.clustered_flag = syn2.clustered_flag = syn3.clustered_flag = True

                        self.connect_input_generator(self.exglu[-1], 't_exglu', syn1.stim[-1])
                        self.connect_input_generator(syn3, 's_exglu', self.exglu[-1])

                    elif syntype == 'no_spillover':
                        syn1 = self.cell.insert_synapse('AMPA', self.cell.dendlist[loc], pos,
                                                   add_spine = add_spine, on_spine = on_spine)
                        self.add_input_generator(syn1, 'AMPA', deterministic = deterministic, numsyn = i, tstart = start)
                        if self.cell.spines != []:
                            spines = [s for s in self.cell.spines if s.parent == self.cell.dendlist[loc]]
                            spines[i].syn_on = 0
                        syn2 = self.cell.insert_synapse('NMDA', self.cell.dendlist[loc], pos,
                                                   add_spine = 0, on_spine = 1)
                        self.connect_input_generator(syn2, 'NMDA', syn1.stim[-1])
                        syn1.clustered_flag = syn2.clustered_flag = True

                    elif syntype == 'no_spillover_stp':
                        syn1 = self.cell.insert_synapse('AMPA_stp', self.cell.dendlist[loc], pos,
                                                   add_spine = add_spine, on_spine = on_spine)
                        self.add_input_generator(syn1, 'AMPA_stp', deterministic = deterministic, numsyn = i, tstart = start)
                        if self.cell.spines != []:
                            spines = [s for s in self.cell.spines if s.parent == self.cell.dendlist[loc]]
                            spines[i].syn_on = 0
                        syn2 = self.cell.insert_synapse('NMDA_stp', self.cell.dendlist[loc], pos,
                                                   add_spine = 0, on_spine = 1)
                        self.connect_input_generator(syn2, 'NMDA_stp', syn1.stim[-1])
                        syn1.clustered_flag = syn2.clustered_flag = True

                if syntype == 'my_spillover':
                    self.set_exglu_weights()

    def add_input_generator(self, syn, syntype, freq_multiplier = 1,
                            tstart = p.plateau_burst_start, tend = p.plateau_burst_end, deterministic = 0, numsyn = 1):

        if deterministic == 1:
            noise = 0
            start = tstart
            number = p.num_spikes
            interval = p.deterministic_interval
            weight = p.gAMPAmax_plateau

            if syntype in ['input_syn']:
                weight = p.gAMPAmax
            elif syntype in ['NMDA', 'AMPA', 'NMDAe', 'AMPA_stp', 'NMDA_stp']:
                weight = p.weight
            elif syntype in ['inhexpsyn_plateau']:
                weight = p.gGABAmax_plateau
                number = 1
            elif syntype in ['inhexp2syn_caint']:
                weight = p.gGABAmax_plateau
                number = p.num_inh_spikes
            elif syntype in [ 'inhexp2syn', 'adaptive_inhexp2syn', 'adaptive2_inhexp2syn',
                              'adaptive2_homo_inhexp2syn', 'adaptive2_homo_sz_inhexp2syn',
                              'adaptive2_hetero_inhexp2syn',
                              'adaptive2_homo_bcm_inhexp2syn', 'adaptive2_hetero_bcm_inhexp2syn']:
                weight = p.gGABA_max

        elif deterministic == 0:
            noise = 1
            if syntype in ['ramp']:
                start = p.ramp_burst_start
                end = p.ramp_burst_end
                number = (end-start) * p.ramp_syn_rate
                interval = p.ramp_syn_interval
                weight = p.g_ramp_max

            elif syntype in ['inhexpsyn', 'inhexp2syn', 'adaptive_inhexp2syn']:
                start = 0;
                end = p.simtime
                number = (end-start) * p.irate * freq_multiplier
                interval = p.i_interval/freq_multiplier
                weight = p.g_inhexpsyn_max

            elif syntype in ['inhexpsyn_plateau', 'inhexp2syn_caint']:
                start = p.inhibitory_burst_start
                end = p.inhibitory_burst_end
                number = p.num_inh_spikes
                interval = 0
                weight = p.gGABAmax_plateau

            elif syntype in ['expsyn_plateau', 'plateau_cluster', 'nmda_plateau', 'tmGlut',
                             'glutamate_ica_nmda',
                            'glutamate_plateau',
                            'adaptive_glutamate_hom',
                            'glutamate_xor_test',
                            'AMPA', 'NMDA', 'NMDAe',
                            'AMPA_stp', 'NMDA_stp',
                            'adaptive_AMPA', 'adaptive_NMDA',
                            'adaptive_sAMPA', 'adaptive_sNMDA',
                            'adaptive_hom_AMPA', 'adaptive_hom_NMDA',
                            'AMPA_test', 'NMDA_test','adaptive_NMDAe',
                            'adaptive_shom_AMPA', 'adaptive_shom_NMDA', 'adaptive_my_shom_NMDA',
                            'adaptive_shom_AMPA_stp', 'adaptive_shom_NMDA_stp',
                            'adaptive_glutamate_shom', 'adaptive2_inhexp2syn',
                            'adaptive2_homo_inhexp2syn', 'adaptive2_homo_sz_inhexp2syn',
                            'adaptive2_homo_bcm_inhexp2syn', 'adaptive2_hetero_inhexp2syn',
                            'adaptive2_hetero_bcm_inhexp2syn',
                            'adaptive_cshom_AMPA', 'adaptive_cshom_NMDA',
                            'adaptive_glutamate_cshom', 'adaptive_sglutamate', 'NMDAe',
                            'adaptive_zahra_NMDA','adaptive_pf_AMPA', 'adaptive_pf_NMDA']:
                start = tstart; end = tend
                number = p.num_spikes#(end-start) * p.plateau_syn_rate
                interval = p.deterministic_interval#p.plateau_syn_interval
                weight = 1.0
                if syntype in ['expsyn_plateau', 'tmGlut',
                'glutamate_plateau']:
                    weight = p.gAMPAmax_plateau
                elif syntype in ['plateau_cluster', 'nmda_plateau', 'generalized_rule',
                                 'generalized_rule_dist']:
                    weight = p.gNMDAmax_plateau
                elif syntype in ['nmda']:
                    weight = p.gNMDAmax
                elif syntype in ['NMDA', 'AMPA', 'NMDA_stp', 'AMPA_stp']:
                    weight = p.weight
                elif syntype in ['adaptive2_inhexp2syn', 'adaptive2_homo_inhexp2syn', 'adaptive2_homo_sz_inhexp2syn',
                                 'adaptive2_hetero_inhexp2syn', 'adaptive2_homo_bcm_inhexp2syn',
                                 'adaptive2_hetero_bcm_inhexp2syn']:
                    number = p.num_inh_spikes

            elif syntype in ['AMPA_pf', 'NMDA_pf', 'adaptive_pf_AMPA', 'adaptive_pf_NMDA']:
                start = p.pf_input_start
                end = p.pf_input_end
                number = p.pf_num_spikes
                interval = p.pf_input_interval
                weight = p.weight

            elif syntype in ['input_syn']:
                start = p.distributed_input_start
                end = p.distributed_input_end
                number = (end-start) * p.distributed_input_rate
                interval = p.distributed_input_interval
                weight = p.gAMPAmax

            elif syntype in ['adaptive_glutamate']:
                start = tstart; end = tend
                number = (end-start) * p.low_rate
                interval = p.low_interval
                weight = 1.0

            elif syntype in ['expsyn', 'exp2syn', 'glutamate']:
                start = 0;
                end = p.simtime
                number = (end-start) * p.erate * freq_multiplier
                interval = p.e_interval/freq_multiplier
                weight = p.g_expsyn_max

        gen = h.NetStim(0.5, sec = self.presyn)
        gen.seed(int(time.time() + rnd.randint(1,10**7)))
        gen.start = start
        gen.noise = noise
        gen.number = number
        gen.interval = interval

        nc = h.NetCon(gen, syn.obj)
        nc.delay = 0
        nc.weight[0] = weight

        if syntype in ['ramp']:
            self.ramp_estim.append(gen)
            self.ramp_enc.append(nc)

        elif syntype in ['inhexpsyn', 'inhexp2syn', 'adaptive_inhexp2syn', 'inhexp2syn_caint',
                         'adaptive2_inhexp2syn','inhexpsyn_plateau', 'adaptive2_homo_inhexp2syn',
                         'adaptive2_homo_sz_inhexp2syn',
                         'adaptive2_hetero_inhexp2syn', 'adaptive2_homo_bcm_inhexp2syn',
                         'adaptive2_hetero_bcm_inhexp2syn',
                         'expsyn', 'exp2syn', 'plateau_cluster', 'input_syn',
                         'expsyn_plateau', 'nmda_plateau', 'tmGlut', 'glutamate',
                         'adaptive_glutamate', 'glutamate_ica_nmda', 'glutamate_plateau',
                         'pf','AMPA', 'NMDA', 'NMDAe', 'adaptive_AMPA', 'NMDA_stp', 'AMPA_stp',
                         'adaptive_NMDA', 'adaptive_hom_AMPA', 'adaptive_hom_NMDA',
                         'AMPA_test', 'NMDA_test','adaptive_shom_AMPA', 'adaptive_shom_NMDA',
                         'adaptive_shom_AMPA_stp', 'adaptive_shom_NMDA_stp', 'adaptive_my_shom_NMDA',
                         'adaptive_NMDAe', 'adaptive_glutamate_shom','adaptive_cshom_AMPA',
                         'adaptive_cshom_NMDA','adaptive_glutamate_cshom', 'adaptive_sAMPA', 'adaptive_sNMDA',
                         'adaptive_sglutamate', 'adaptive_zahra_NMDA', 'AMPA_pf', 'NMDA_pf']:
            syn.stim.append(gen)
            syn.nc.append(nc)

    def set_up_recording(self, dend_record_list = [], record_step = p.record_step):
        self.dend_record_list = dend_record_list
        self.vdlist = []
        self.t = h.Vector()
        self.t.record(h._ref_t, record_step)
        self.tv = h.Vector()
        self.tv.record(h._ref_t, p.record_step_v)
        self.tthresh = h.Vector()
        self.tthresh.record(h._ref_t, p.record_step_thresh)

        self.cali = []
        self.cali_dend = []
        self.cai_nmda = []
        self.cai = []
        self.cati = []
        self.cai_nmda_spine = []
        self.cai_spine = []
        self.cali_spine = []
        self.cati_spine = []
        self.vspine = []
        self.kernel = []
        self.kernel_LTD = []
        self.cao = []

        self.vs = []
        self.vs = h.Vector()
        self.vs.record(self.cell.somalist[0](0.5)._ref_v, p.record_step_v)
        self.Cdur = []
        if self.exptype == 'voltage statistics':
            if dend_record_list != []:
                for sec in dend_record_list:
                    self.vdlist.append(h.Vector())
                    self.vdlist[-1].record(self.cell.dendlist[sec](p.pos)._ref_v, p.record_step_v)
                    self.record_spinelist = [s for s in self.cell.spines if s.parent == self.cell.dendlist[sec]]
                    for spine in self.record_spinelist:
                        self.vspine.append(h.Vector())
                        self.vspine[-1].record(spine.head(0.5)._ref_v, p.record_step_v)

            else:
                for loc in range(0,len(self.cell.dendlist)):
                    self.dend_record_list.append(loc)
                    recording_points = self.get_recording_points(self.cell.dendlist[loc], p.step)
                    for pos in recording_points:
                        self.vdlist.append({'sec': loc, 'pos': pos, 'data': h.Vector()})
                        self.vdlist[-1]['data'].record(self.cell.dendlist[loc](pos)._ref_v, p.record_step_v)

        if self.exptype in ['xor', 'xor_test_set', 'xor_hom', 'xor_gen',
                            'xor_spillover', 'xor_sspillover', 'xor_hom_spillover',
                            'xor_spillover_test', 'xor_shom_my_spillover', 'xor_shom_my_spillover_stp',
                            'xor_shom_spillover','xor_cshom_spillover', 'xor_zahra_spillover',
                            'inhibitory_plasticity', 'pattern', 'pattern_homo', 'pattern_homo_sz',
                            'pattern_homo_bcm', 'pattern_hetero_bcm',
                            'pattern_hetero', 'pattern_hetero_amp',
                            'pattern_cont_hetero_amp', 'nfbp_inh']:

            self.cali = []
            if type(dend_record_list) == list:
                if dend_record_list != []:
                    for sec in dend_record_list:
                        self.dend_record_list = dend_record_list
                        self.vdlist.append(h.Vector())
                        self.vdlist[-1].record(self.cell.dendlist[sec](p.pos)._ref_v, p.record_step_v)
                        self.cali.append(h.Vector())
                        pos = 0.5
                        self.cali[-1].record(self.cell.dendlist[sec](pos)._ref_cali, record_step)
                        self.record_spinelist = [s for s in self.cell.spines if s.parent == self.cell.dendlist[sec]]
                        # print("RECORDING FROM %d SPINES" % len(self.record_spinelist))
                        # for spine in self.record_spinelist:
                        #     self.vspine.append(h.Vector())
                        #     self.vspine[-1].record(spine.head(0.5)._ref_v, p.record_step_v)
                        #     self.cao.append(h.Vector())
                        #     self.cao[-1].record(spine.head(0.5)._ref_cao, record_step)
#                        self.cali.append(h.Vector())
#                        self.cali[-1].record(self.cell.dendlist[sec](p.pos)._ref_cali, record_step)

            if p.mpi:
                pass
            else:
                if self.exptype in ['inhibitory_plasticity', 'nfbp_inh', 'pattern',
                                    'pattern_homo', 'pattern_homo_sz',
                                    'pattern_homo_bcm', 'pattern_hetero_bcm',
                                    'pattern_hetero', 'pattern_hetero_amp',
                                    'pattern_cont_hetero_amp']:
                    self.m = []

                    if self.exptype not in ['inhibitory_plasticity']:
                        self.stimulus_flag = h.Vector()
                        self.stimulus_flag.record(h._ref_stimulus_flag, record_step)
                        self.weight_update_flag = h.Vector()
                        self.weight_update_flag.record(h._ref_weight_update_flag, record_step)

                        for intfire in self.exglu:
                            self.m.append(h.Vector())
                            self.m[-1].record(intfire._ref_m, record_step)

                if self.exptype in ['xor']:
                    synlist = self.get_synapse_list('adaptive_glutamate2', clustered_flag = True)
                elif self.exptype == 'xor_hom':
                    synlist = self.get_synapse_list('adaptive_glutamate_hom', clustered_flag = True)
                elif self.exptype == 'xor_gen':
                    synlist = self.get_synapse_list('generalized_rule', clustered_flag = True)
                elif self.exptype in ['xor_spillover']:
                    synlist = self.get_synapse_list('adaptive_NMDA', clustered_flag = True)
                elif self.exptype == 'xor_sspillover':
                    synlist = self.get_synapse_list('adaptive_sAMPA', clustered_flag = True)
                elif self.exptype in ['xor_hom_spillover', 'nfbp_inh']:
                    synlist = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
                elif self.exptype in ['xor_shom_spillover']:
                    synlist = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
                elif self.exptype in ['xor_shom_my_spillover']:
                    synlist = self.get_synapse_list('adaptive_my_shom_NMDA', clustered_flag = True)
                elif self.exptype == 'xor_shom_my_spillover_stp':
                    synlist = self.get_synapse_list('adaptive_shom_NMDA_stp', clustered_flag = True)
                elif self.exptype == 'xor_zahra_spillover':
                    synlist = self.get_synapse_list('adaptive_zahra_NMDA', clustered_flag = True)
                elif self.exptype == 'xor_cshom_spillover':
                    synlist = self.get_synapse_list('adaptive_cshom_NMDA', clustered_flag = True)
                elif self.exptype == 'xor_test_set':
                    synlist = self.get_synapse_list('glutamate_xor_test', clustered_flag = True)
                elif self.exptype == 'xor_spillover_test':
                    synlist = self.get_synapse_list('glutamate_xor_test', clustered_flag = True)
                elif self.exptype == 'inhibitory_plasticity':
                    synlist = self.get_synapse_list('adaptive_inhexp2syn', clustered_flag = False, drive_type = 'i')
                elif self.exptype == 'pattern':
                    synlist = self.get_synapse_list('adaptive2_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_homo':
                    synlist = self.get_synapse_list('adaptive2_homo_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_homo_sz':
                    synlist = self.get_synapse_list('adaptive2_homo_sz_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_homo_bcm':
                    synlist = self.get_synapse_list('adaptive2_homo_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_hetero_bcm':
                    synlist = self.get_synapse_list('adaptive2_hetero_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_hetero':
                    synlist = self.get_synapse_list('adaptive2_hetero_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_hetero_amp':
                    synlist = self.get_synapse_list('adaptive2_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_cont_hetero_amp':
                    synlist = self.get_synapse_list('adaptive2_cont_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
                synlist.sort(key = lambda f: f.sec.name())

                # if self.dendstatobj.pf_inputs != [[], []]:
                #     synlist_pf = self.get_synapse_list('adaptive_pf_NMDA', clustered_flag = True)
                #     self.weight_pf = []
                #     for s in synlist_pf:
                #         self.weight_pf.append(h.Vector())
                #         self.weight_pf[-1].record(s.obj._ref_weight, record_step)
                #         s.ref_var_nmda = self.weight_pf[-1]

                self.cai_nmda_in_syns = []
                self.cai_in_syns = []
                self.cali_in_syns = []
                self.cati_in_syns = []
                self.cai_nmda_max = []
                self.cali_max = []
                self.ica_in_syns = []
                for s in synlist:
                    if not p.with_diffusion:
                        self.cai_nmda_in_syns.append(h.Vector())
                        self.cai_nmda_in_syns[-1].record(s.sec(s.pos)._ref_ca_nmdai, record_step)
                        s.ref_var_cai_nmda = self.cai_nmda_in_syns[-1]
                    self.cali_in_syns.append(h.Vector())
                    self.cali_in_syns[-1].record(s.sec(s.pos)._ref_cali, record_step)
    #                self.cati_in_syns.append(h.Vector())
    #                self.cati_in_syns[-1].record(s.sec(s.pos)._ref_cati, record_step)
                    self.cai_in_syns.append(h.Vector())
                    self.cai_in_syns[-1].record(s.sec(s.pos)._ref_cai, record_step)
                    # self.ica_in_syns.append(h.Vector())
                    # self.ica_in_syns[-1].record(s.sec(s.pos)._ref_ica, record_step)
                    s.ref_var_cali = self.cali_in_syns[-1]
                    s.ref_var_cai = self.cai_in_syns[-1]
                    if self.exptype not in ['inhibitory_plasticity', 'pattern', 'pattern_homo',
                                            'pattern_homo_sz', 'pattern_homo_bcm', 'pattern_hetero_bcm',
                                            'pattern_hetero', 'pattern_hetero_amp', 'pattern_cont_hetero_amp']:
                        self.cai_nmda_max.append(h.Vector())
                        self.cai_nmda_max[-1].record(s.obj._ref_ca_nmdai_max, record_step)
                        self.cali_max.append(h.Vector())
                        self.cali_max[-1].record(s.obj._ref_cali_max, record_step)

                if self.exptype not in ['inhibitory_plasticity', 'pattern', 'pattern_homo',
                                        'pattern_homo_sz', 'pattern_homo_bcm', 'pattern_hetero_bcm',
                                        'pattern_hetero', 'pattern_hetero_amp', 'pattern_cont_hetero_amp']:

                    self.dopamine_vec = h.Vector()
                    self.dopamine_vec.record(h._ref_dopamine, record_step)

                if self.exptype in ['xor_spillover','xor_hom_spillover', 'xor_gen','xor_shom_spillover',
                                    'xor_cshom_spillover', 'xor_sspillover', 'xor_zahra_spillover',
                                    'xor_shom_my_spillover', 'xor_shom_my_spillover_stp']:
                    self.cali_dend = []
                    self.cai_dend = []
                    self.cai_nmda_dend = []
                    self.cati_dend = []

                    if not p.long_simulation:
                        for d in dend_record_list:
                            self.cai_nmda_dend.append(h.Vector())
                            self.cali_dend.append(h.Vector())
    #                        self.cati_dend.append(h.Vector())
                            self.cai_dend.append(h.Vector())

                            self.cai_nmda_dend[-1].record(self.cell.dendlist[d](p.pos)._ref_ca_nmdai, record_step)
                            self.cali_dend[-1].record(self.cell.dendlist[d](p.pos)._ref_cali, record_step)
    #                        self.cati_dend[-1].record(self.cell.dendlist[d](p.pos)._ref_cati, record_step)
                            self.cai_dend[-1].record(self.cell.dendlist[d](p.pos)._ref_cai, record_step)
                        for s in synlist:
                            self.kernel.append(h.Vector())
                            self.kernel[-1].record(s.obj._ref_kernel, record_step)
                            self.kernel_LTD.append(h.Vector())
                            self.kernel_LTD[-1].record(s.obj._ref_kernel_LTD, record_step)
                    self.m = []
                    for intfire in self.exglu:
                        self.m.append(h.Vector())
                        self.m[-1].record(intfire._ref_m, record_step)

        if self.exptype in ['record_ca']:
            self.m = []
            self.ica_nmda = []
            self.caint = []

            # self.dopamine_vec = h.Vector()
            # self.dopamine_vec.record(h._ref_dopamine, record_step)
            # self.stimulus_flag = h.Vector()
            # self.stimulus_flag.record(h._ref_stimulus_flag, record_step)

            self.cai_soma = h.Vector()
            self.cai_soma.record(self.cell.somalist[0](0.5)._ref_cai, record_step)
            for intfire in self.exglu:
                self.m.append(h.Vector())
                self.m[-1].record(intfire._ref_m, record_step)

            for d in dend_record_list:
                self.vdlist.append(h.Vector())
                self.vdlist[-1].record(self.cell.dendlist[d](p.pos)._ref_v, p.record_step_v)

                self.cai.append(h.Vector())
                self.cali.append(h.Vector())
                self.cati.append(h.Vector())
                self.cai_nmda.append(h.Vector())
                self.cali_dend.append(h.Vector())
                pos = p.cluster_start_poss[p.independent_dends.index(d)]
                self.cai[-1].record(self.cell.dendlist[d](pos)._ref_cai, record_step)
                self.cali[-1].record(self.cell.dendlist[d](pos)._ref_cali, record_step)
                self.cati[-1].record(self.cell.dendlist[d](pos)._ref_cati, record_step)
                self.cai_nmda[-1].record(self.cell.dendlist[d](pos)._ref_ca_nmdai, record_step)
                self.cali[-1].record(self.cell.dendlist[d](pos)._ref_cali, record_step)
                self.cati[-1].record(self.cell.dendlist[d](pos)._ref_cati, record_step)

                if self.cell.spines != []:
                    self.record_spinelist = [s for s in self.cell.spines if s.parent == self.cell.dendlist[d] and s.syn_on == 1]
                    if p.include_empty_spines:
                        self.record_spinelist.extend([s for s in self.cell.spines if s.parent == self.cell.dendlist[d] and s.syn_on == 0])
                    print("In record_ca: recording from %d spines" % len(self.record_spinelist))
                    for spine in self.record_spinelist:
#                    spine = self.record_spinelist[0]
                        self.vspine.append(h.Vector())
                        self.vspine[-1].record(spine.head(0.5)._ref_v, p.record_step_v)
                        self.cali_spine.append(h.Vector())
                        self.cali_spine[-1].record(spine.head(0.5)._ref_cali, record_step)
                        self.cati_spine.append(h.Vector())
                        self.cati_spine[-1].record(spine.head(0.5)._ref_cati, record_step)
                        self.cao.append(h.Vector())
                        self.cao[-1].record(spine.head(0.5)._ref_cao, record_step)
                        self.cai_spine.append(h.Vector())
                        self.cai_spine[-1].record(spine.head(0.5)._ref_cai, record_step)
                        self.cai_nmda_spine.append(h.Vector())
                        self.cai_nmda_spine[-1].record(spine.head(0.5)._ref_ca_nmdai, record_step)

#                        self.ical = []
#                        self.ical.append(h.Vector())
#                        self.ical[-1].record(spine.head(0.5)._ref_ical, record_step)
                        self.ica_nmda = []
                        self.ica_nmda.append(h.Vector())
                        self.ica_nmda[-1].record(spine.head(0.5)._ref_ica_nmda, record_step)

                synlist = self.get_synapse_list('inhexp2syn_caint', clustered_flag = True, drive_type = 'i')
                self.caint.append(h.Vector())
                self.caint[-1].record(synlist[0].obj._ref_caint, record_step)

        if self.exptype in ['inhibitory_plasticity','record_ca', 'nfbp_inh',
                            'pattern', 'pattern_homo', 'pattern_homo_sz',
                            'pattern_homo_bcm', 'pattern_hetero_bcm',
                            'pattern_hetero', 'pattern_hetero_amp',
                            'pattern_cont_hetero_amp'] and not p.mpi:

            self.w_inh = []
            self.theta_inh = []
            self.kernel_theta_min_inh = []
            self.theta_min_inh = []
            self.kernel_inh = []
            self.caint_syns = []
            self.caint_max = []
            self.caint = []
            self.tlast = []
            self.delta_t = []
            self.asf = []
            for s in self.cell.isyn:
                if s.type in ['adaptive_inhexp2syn']:
                    self.w_inh.append(h.Vector())
                    self.w_inh[-1].record(s.obj._ref_weight, record_step)
                    s.ref_var_w_inh = self.w_inh[-1]
                    # self.kernel_inh.append(h.Vector())
                    # self.kernel_inh[-1].record(s.obj._ref_kernel, record_step)
                if s.type in ['adaptive2_inhexp2syn']:
                    self.w_inh.append(h.Vector())
                    self.w_inh[-1].record(s.obj._ref_weight, p.record_step_thresh)
                    s.ref_var_w_inh = self.w_inh[-1]
                    # self.kernel_inh.append(h.Vector())
                    # self.kernel_inh[-1].record(s.obj._ref_kernel, record_step)
                    self.theta_inh.append(h.Vector())
                    self.theta_inh[-1].record(s.obj._ref_theta, p.record_step_thresh)
                    # # self.theta_min_inh.append(h.Vector())
                    # # self.theta_min_inh[-1].record(s.obj._ref_theta_min, p.record_step_thresh)
                    # self.kernel_theta_min_inh.append(h.Vector())
                    # self.kernel_theta_min_inh[-1].record(s.obj._ref_kernel_theta_min, p.record_step_thresh)
                    self.caint_syns.append(h.Vector())
                    self.caint_syns[-1].record(s.obj._ref_caint, record_step)
                    self.caint_max.append(h.Vector())
                    self.caint_max[-1].record(s.obj._ref_calcium_max, record_step)
                    s.ref_var_theta_inh = self.theta_inh[-1]
                    # s.ref_var_theta_min_inh = self.theta_min_inh[-1]
                    # s.ref_var_kernel_theta_min_inh = self.kernel_theta_min_inh[-1]
                    s.ref_var_caint = self.caint_syns[-1]
                    s.ref_var_caint_max = self.caint_max[-1]
                    # self.tlast.append(h.Vector())
                    # self.tlast[-1].record(s.obj._ref_tlast, record_step)
                    # self.delta_t.append(h.Vector())
                    # self.delta_t[-1].record(s.obj._ref_delta_t, record_step)
                    # self.asf.append(h.Vector())
                    # self.asf[-1].record(s.obj._ref_active_syn_flag, record_step)
                if s.type in ['adaptive2_homo_inhexp2syn', 'adaptive2_homo_sz_inhexp2syn',
                              'adaptive2_homo_bcm_inhexp2syn', 'adaptive2_hetero_bcm_inhexp2syn']:
                    self.w_inh.append(h.Vector())
                    self.w_inh[-1].record(s.obj._ref_weight,  p.record_step_thresh)
                    s.ref_var_w_inh = self.w_inh[-1]
                    # self.kernel_inh.append(h.Vector())
                    # self.kernel_inh[-1].record(s.obj._ref_kernel, record_step)
                    self.theta_inh.append(h.Vector())
                    self.theta_inh[-1].record(s.obj._ref_theta, p.record_step_thresh)
                    self.caint_syns.append(h.Vector())
                    self.caint_syns[-1].record(s.obj._ref_caint, record_step)
                    self.caint_max.append(h.Vector())
                    self.caint_max[-1].record(s.obj._ref_calcium_max, record_step)
                    s.ref_var_theta_inh = self.theta_inh[-1]
                    s.ref_var_caint = self.caint_syns[-1]
                    s.ref_var_caint_max = self.caint_max[-1]
                    # self.asf.append(h.Vector())
                    # self.asf[-1].record(s.obj._ref_active_syn_flag, record_step)
                if s.type in ['adaptive2_hetero_inhexp2syn', 'adaptive2_hetero_amp_inhexp2syn',
                    'adaptive2_cont_hetero_amp_inhexp2syn']:
                    self.w_inh.append(h.Vector())
                    self.w_inh[-1].record(s.obj._ref_weight,  p.record_step_thresh)
                    s.ref_var_w_inh = self.w_inh[-1]
                    self.theta_inh.append(h.Vector())
                    self.theta_inh[-1].record(s.obj._ref_theta, p.record_step_thresh)
                    self.caint_syns.append(h.Vector())
                    self.caint_syns[-1].record(s.obj._ref_caint, record_step)
                    self.caint_max.append(h.Vector())
                    self.caint_max[-1].record(s.obj._ref_calcium_max, record_step)
                    s.ref_var_theta_inh = self.theta_inh[-1]
                    s.ref_var_caint = self.caint_syns[-1]
                    s.ref_var_caint_max = self.caint_max[-1]
                    # self.asf.append(h.Vector())
                    # self.asf[-1].record(s.obj._ref_active_syn_flag, record_step)

            if self.exptype in ['inhibitory_plasticity', 'nfbp_inh', 'pattern',
                                'pattern_homo', 'pattern_homo_sz', 'pattern_homo_bcm',
                                'pattern_hetero_bcm', 'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp']:
                for d in dend_record_list:
                    pos = p.cluster_start_poss[p.independent_dends.index(d)]
                    self.caint.append(h.Vector())
                    self.caint[-1].record(self.cell.dendlist[d](pos)._ref_caint_caint, record_step)

        if self.exptype == 'record_i':
            self.ina = []
            self.ik = []
            self.iampa = []
            self.iNMDA = []
            self.ica_nmda = []

            for d in dend_record_list:
                self.vdlist.append(h.Vector())
                self.vdlist[-1].record(self.cell.dendlist[d](p.pos)._ref_v, p.record_step_v)
                self.ik.append(h.Vector())
                self.ik[-1].record(self.cell.dendlist[d](p.pos)._ref_ik, record_step)
                self.ina.append(h.Vector())
                self.ina[-1].record(self.cell.dendlist[d](p.pos)._ref_ina, record_step)
                ampa_syns = [s for s in self.cell.esyn if s.sec == self.cell.dendlist[d] and s.type == 'AMPA']
                nmda_syns = [s for s in self.cell.esyn if s.sec == self.cell.dendlist[d] and s.type == 'NMDA']
                for s in ampa_syns:
                    self.iampa.append(h.Vector())
                    self.iampa[-1].record(s.obj._ref_iAMPA, record_step)
                for s in nmda_syns:
                    self.iNMDA.append(h.Vector())
                    self.iNMDA[-1].record(s.obj._ref_iNMDA, record_step)
                    self.ica_nmda.append(h.Vector())
                    self.ica_nmda[-1].record(s.obj._ref_ica_nmda, record_step)

        self.gsyn = []
        self.inmda = []
        self.A = []
        self.B = []

        self.w_ampa = []
        self.w_nmda = []
        self.lthresh_LTP = []
        self.hthresh_LTP = []
        self.lthresh_LTD = []
        self.cali_agh = []
        self.cati_agh = []
        self.cai_agh = []
        self.cai_nmda_agh = []
        self.v_agh = []
        self.kernel_agh = []
        self.kernel_LTD_agh = []
        self.delta_LTP = []
#
        for syn in self.cell.esyn:
            if syn.type in ['glutamate_mod', 'glutamate_test',
                'adaptive_glutamate_test'] :
                    self.w_nmda.append(h.Vector())
                    self.w_nmda[-1].record(syn.obj._ref_w_nmda, p.record_step_thresh)
            if syn.type in ['adaptive_NMDA', 'adaptive_hom_NMDA', 'adaptive_hom_NMDA2',
            'adaptive_NMDAe', 'adaptive_shom_NMDA', 'adaptive_cshom_NMDA', 'adaptive_sNMDA',
            'adaptive_zahra_NMDA', 'adaptive_shom_NMDA_stp', 'adaptive_my_shom_NMDA', 'adaptive_NMDAe'] :
                    self.w_nmda.append(h.Vector())
                    self.w_nmda[-1].record(syn.obj._ref_weight, p.record_step_thresh)
                    syn.ref_var_nmda = self.w_nmda[-1]

            if (syn.type in ['adaptive_shom_NMDA','adaptive_shom_NMDA_stp', 'adaptive_my_shom_NMDA'] and syn.clustered_flag == False):
                self.v_agh.append(h.Vector())
                self.v_agh[-1].record(syn.sec(syn.pos)._ref_v, p.record_step_v)

                self.cali_agh.append(h.Vector())
                self.cali_agh[-1].record(syn.sec(syn.pos)._ref_cali, record_step)
                syn.ref_var_cali = self.cali_agh[-1]

#                self.cati_agh.append(h.Vector())
#                self.cati_agh[-1].record(syn.sec(syn.pos)._ref_cati, record_step)

                self.cai_agh.append(h.Vector())
                self.cai_agh[-1].record(syn.sec(syn.pos)._ref_cai, record_step)
                syn.ref_var_cai = self.cai_agh[-1]

                self.cai_nmda_agh.append(h.Vector())
                self.cai_nmda_agh[-1].record(syn.sec(syn.pos)._ref_ca_nmdai, record_step)
                syn.ref_var_cai_nmda = self.cai_nmda_agh[-1]

                # self.kernel_agh.append(h.Vector())
                # self.kernel_agh[-1].record(syn.obj._ref_kernel, record_step)
                #
                # self.kernel_LTD_agh.append(h.Vector())
                # self.kernel_LTD_agh[-1].record(syn.obj._ref_kernel_LTD, record_step)

            if syn.type in ['adaptive_glutamate_test','adaptive_glutamate', 'adaptive_glutamate_hom2']:
                self.w_ampa.append(h.Vector())
                self.w_ampa[-1].record(syn.obj._ref_w_ampa, p.record_step_thresh)
                syn.ref_var_ampa = self.w_ampa[-1]

                self.w_nmda.append(h.Vector())
                self.w_nmda[-1].record(syn.obj._ref_w_nmda, p.record_step_thresh)
                syn.ref_var_nmda = self.w_nmda[-1]

                self.cali_agh.append(h.Vector())
                self.cali_agh[-1].record(syn.sec(syn.pos)._ref_cali, record_step)

                self.cai_nmda_agh.append(h.Vector())
                self.cai_nmda_agh[-1].record(syn.sec(syn.pos)._ref_ca_nmdai, record_step)

            if syn.type in ['adaptive_glutamate2', 'adaptive_glutamate_hom']:
                self.w_ampa.append(h.Vector())
                self.w_ampa[-1].record(syn.obj._ref_weight, p.record_step_thresh)
                syn.ref_var_ampa = self.w_ampa[-1]

                self.cali_agh.append(h.Vector())
                self.cali_agh[-1].record(syn.sec(syn.pos)._ref_cali, record_step)

                self.cai_nmda_agh.append(h.Vector())
                self.cai_nmda_agh[-1].record(syn.sec(syn.pos)._ref_ca_nmdai, record_step)

                if syn.type in ['adaptive_glutamate2']:
                    syn.ref_var_lthresh_LTD = h.Vector(np.ones(( int(p.simtime/p.record_step_thresh)))*syn.obj.thresh_LTD )
                    syn.ref_var_lthresh_LTp = h.Vector(np.ones(( int(p.simtime/p.record_step_thresh)))*syn.obj.thresh_LTP )

                elif syn.type in ['adaptive_glutamate_hom']:
                    self.lthresh_LTD.append(h.Vector())
                    self.lthresh_LTD[-1].record(syn.obj._ref_thresh_LTD, p.record_step_thresh)
                    syn.ref_var_lthresh_LTD = self.lthresh_LTD[-1]

                    self.lthresh_LTP.append(h.Vector())
                    self.lthresh_LTP[-1].record(syn.obj._ref_thresh_LTP, p.record_step_thresh)
                    syn.ref_var_lthresh_LTP = self.lthresh_LTP[-1]

            if syn.type in ['adaptive_glutamate_shom', 'adaptive_glutamate_cshom', 'adaptive_sglutamate'] or \
            (syn.type in ['adaptive_cshom_NMDA'] and syn.clustered_flag == False):
                if not p.long_simulation:
                    self.w_ampa.append(h.Vector())
                    self.w_ampa[-1].record(syn.obj._ref_weight, p.record_step_thresh)
                    syn.ref_var_ampa = self.w_ampa[-1]

                    self.lthresh_LTP.append(h.Vector())
                    self.lthresh_LTP[-1].record(syn.obj._ref_lthresh_LTP, p.record_step_thresh)
                    syn.ref_var_lthresh_LTP = self.lthresh_LTP[-1]

                    self.hthresh_LTP.append(h.Vector())
                    self.hthresh_LTP[-1].record(syn.obj._ref_hthresh_LTP, p.record_step_thresh)
                    syn.ref_var_hthresh_LTP = self.hthresh_LTP[-1]

                    self.lthresh_LTD.append(h.Vector())
                    self.lthresh_LTD[-1].record(syn.obj._ref_lthresh_LTD, p.record_step_thresh)
                    syn.ref_var_lthresh_LTD = self.lthresh_LTD[-1]
                    if not p.long_simulation:
                        self.v_agh.append(h.Vector())
                        self.v_agh[-1].record(syn.sec(syn.pos)._ref_v, p.record_step_v)

                        self.cali_agh.append(h.Vector())
                        self.cali_agh[-1].record(syn.sec(syn.pos)._ref_cali, record_step)

                        self.cai_nmda_agh.append(h.Vector())
                        self.cai_nmda_agh[-1].record(syn.sec(syn.pos)._ref_ca_nmdai, record_step)


            if syn.type == 'adaptive_glutamate_hom' or syn.type == 'adaptive_hom_NMDA':
                self.lthresh_LTP.append(h.Vector())
                self.lthresh_LTP[-1].record(syn.obj._ref_thresh_LTP, p.record_step_thresh)
                syn.ref_var_lthresh_LTP = self.lthresh_LTP[-1]

                self.hthresh_LTP.append(h.Vector())
                self.hthresh_LTP[-1].record(syn.obj._ref_thresh_LTP, p.record_step_thresh)
                syn.ref_var_hthresh_LTP = self.hthresh_LTP[-1]

                self.lthresh_LTD.append(h.Vector())
                self.lthresh_LTD[-1].record(syn.obj._ref_thresh_LTD, p.record_step_thresh)
                syn.ref_var_lthresh_LTD = self.lthresh_LTD[-1]

            if syn.type in ['adaptive_sNMDA']:
                self.hthresh_LTP.append(h.Vector())
                self.hthresh_LTP[-1].record(syn.obj._ref_hthresh_LTP, p.record_step_thresh)
                syn.ref_var_hthresh_LTP = self.hthresh_LTP[-1]

                self.lthresh_LTP.append(h.Vector())
                self.lthresh_LTP[-1].record(syn.obj._ref_lthresh_LTP, p.record_step_thresh)
                syn.ref_var_lthresh_LTP = self.lthresh_LTP[-1]

            if syn.type in ['adaptive_shom_NMDA', 'adaptive_cshom_NMDA',
                            'adaptive_my_shom_NMDA', 'adaptive_shom_NMDA_stp']:
                if not p.long_simulation:
                    self.lthresh_LTP.append(h.Vector())
                    self.lthresh_LTP[-1].record(syn.obj._ref_lthresh_LTP, p.record_step_thresh)
                    syn.ref_var_lthresh_LTP = self.lthresh_LTP[-1]

                    self.hthresh_LTP.append(h.Vector())
                    self.hthresh_LTP[-1].record(syn.obj._ref_hthresh_LTP, p.record_step_thresh)
                    syn.ref_var_hthresh_LTP = self.hthresh_LTP[-1]

    #                self.delta_LTP.append(h.Vector())
    #                self.delta_LTP[-1].record(syn.obj._ref_delta_LTP, p.record_step_thresh)
                    self.lthresh_LTD.append(h.Vector())
                    self.lthresh_LTD[-1].record(syn.obj._ref_lthresh_LTD, p.record_step_thresh)
                    syn.ref_var_lthresh_LTD = self.lthresh_LTD[-1]

#                    self.cati_max.append(h.Vector())
#                    self.cati_max[-1].record(syn.obj._ref_cati_max, p.record_step_thresh)
#                    syn.ref_var_cati_max = self.cati_max[-1]


    def plot_ba(self):
        fig_ba, axes_ba = plt.subplots(3, 1, sharex = True)
        legends = [];
        stimuli = 3
        multiplier = int(p.session_length*p.nrn_dots_per_1ms)
        for j in range(0, len(self.vdlist)):
            for i in range(0,stimuli):
                color = self.set_color(i+1); color = color[0]
                axes_ba[j].plot(self.tv.to_python()[p.first_training_input_start+i*multiplier : p.first_training_input_start + (i+1)*multiplier],
                                self.vdlist[j].to_python()[p.first_training_input_start+i*multiplier: p.first_training_input_start + (i+1)*multiplier], color = color, alpha = 0.5);
                if i == stimuli -1 :
                    axes_ba[j].plot(self.tv.to_python()[p.first_training_input_start+i*multiplier:  p.first_training_input_start + (i+1)*multiplier],
                                self.vdlist[j].to_python()[-(stimuli-i)*multiplier:], color = color);
                else:
                    axes_ba[j].plot(self.tv.to_python()[p.first_training_input_start+i*multiplier:  p.first_training_input_start + (i+1)*multiplier],
                                self.vdlist[j].to_python()[-(stimuli-i)*multiplier:-(stimuli-i-1)*multiplier], color = color);

            axes_ba[j].set_ylabel('v$_{\mathrm{{d%d}}}$ (mV)' % (j+1))
            axes_ba[j].set_yticks([-80, -50, -20])
        axes_ba[-1].set_xlabel('t (ms)')

        return fig_ba, axes_ba

    def plot_voltage(self):

        rows = len(self.dend_record_list)//p.dends_per_plot
        if (len(self.dend_record_list) % p.dends_per_plot) == 0:
            rows = rows + 1
        else:
            rows = rows + 2

        plot_num = 1;
        fig, axes = plt.subplots(rows, 1, sharex = True)
        fig_ba, axes_ba = plt.subplots(rows, 1, sharex = True)
        legends = [];
        if self.exptype == 'nfbp_inh':
            stimuli = 4
        elif self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                              'pattern_homo_bcm', 'pattern_hetero_bcm',
                              'pattern_hetero', 'pattern_hetero_amp',
                              'pattern_cont_hetero_amp']:
            stimuli = 3

        axes[0].plot(self.tv, self.vdlist[0]); plot_num += 1;
        multiplier = int(p.session_length*p.nrn_dots_per_1ms)
        axes_ba[0].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start],
                        self.vdlist[0].to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start], color = 'gray');
        axes_ba[0].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start],
                        self.vdlist[0].to_python()[-stimuli*multiplier:], color = 'black');
        plot_num += 1;

        if type(self.dend_record_list) == list:
            legends.append(['Dend 1'])

            for i in range(1,len(self.vdlist)):
                if i//p.dends_per_plot == (i-1)//p.dends_per_plot:
                    idx = (i-1)//p.dends_per_plot
                    axes[idx].plot(self.tv, self.vdlist[i])
                    axes_ba[idx].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start],
                                      self.vdlist[idx].to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start], color = 'gray')
                    axes_ba[idx].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start],
                                      self.vdlist[idx].to_python()[-stimuli*multiplier:], color = 'black')
                    legends.append(['Dend %d' % (i+1)] )
                else:
                    plot_num += 1;
                    idx = i//p.dends_per_plot
                    axes[idx].plot(self.tv, self.vdlist[i])
                    axes_ba[idx].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier + p.first_training_input_start],
                                      self.vdlist[idx].to_python()[p.first_training_input_start:stimuli*multiplier+p.first_training_input_start], color = 'gray')
                    axes_ba[idx].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier + p.first_training_input_start],
                                      self.vdlist[idx].to_python()[-stimuli*multiplier:], color = 'black')
                    legends.append(['Dend %d' % (i+1)] )

        for i in range(0,rows-1):
            axes[i].set_ylabel('v$_{\mathrm{d%d}}$ (mV)' % (i+1))
            axes[i].set_yticks([-80, -50, -20])
            axes[i].legend(legends[i], frameon = False)
            axes_ba[i].set_ylabel('v$_{\mathrm{{d%d}}}$ (mV)' % (i+1))
            axes_ba[i].set_yticks([-80, -50, -20])
            # axes[i].set_xlabel('t (ms)')
            # axes_ba[i].set_xlabel('t (ms)')
            # axes_ba[i].legend(legends[i], frameon = False)

        #-----------------------------------------#
        #           Plot voltage in soma          #
        #-----------------------------------------#

        plot_num += 1;
        axes[-1].plot(self.tv, self.vs)
        axes[-1].set_ylabel('v$_{\mathrm{soma}}$ (mV)')
        axes[-1].set_yticks([-80, -20, 20])
        axes[-1].set_xlabel('t (ms)')
        axes_ba[-1].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier + p.first_training_input_start],
                         self.vs.to_python()[p.first_training_input_start:stimuli*multiplier + p.first_training_input_start], color = 'gray')
        axes_ba[-1].plot(self.tv.to_python()[p.first_training_input_start:stimuli*multiplier + p.first_training_input_start],
                         self.vs.to_python()[-stimuli*multiplier:], color = 'black')
        axes_ba[-1].set_ylabel('v$_{\mathrm{soma}}$ (mV)')
        axes_ba[-1].set_yticks([-80, -20, 40])
        axes_ba[-1].set_xlabel('t (ms)')
        # sns.despine()
        return fig, fig_ba, axes, axes_ba

    def plot_results(self):
        sns.set(font_scale = 1.5)
        sns.set_style("ticks")

        if self.exptype in ['xor', 'xor_hom', 'xor_test_set', 'xor_gen', 'xor_shom_spillover',
                              'xor_spillover', 'xor_sspillover', 'xor_hom_spillover', 'xor_spillover_test',
                              'xor_cshom_spillover', 'xor_zahra_spillover','xor_shom_my_spillover',
                              'xor_shom_my_spillover_stp', 'nfbp_inh', 'pattern', 'pattern_homo',
                              'pattern_homo_sz', 'pattern_homo_bcm', 'pattern_hetero_bcm',
                              'pattern_hetero', 'pattern_hetero_amp', 'pattern_cont_hetero_amp']:
            #-----------------------------------------#
            #        Plot voltage in dendrites        #
            #-----------------------------------------#

            fig_, fig_ba, axes, axes_ba = self.plot_voltage()
            #------------------------------------#
            #           Plot thresholds          #
            #------------------------------------#
#            axes_LTP, axes_LTD = self.plot_thresholds()

#            sns.despine()

            #------------------------------------#
            #           Plot delta_LTP           #
            #------------------------------------#

            # fig_delta_LTP = plt.figure();
            # ax_delta_LTP = fig_delta_LTP.add_subplot(111);
            # ax_delta_LTP.set_ylabel('delta_LTP')
            # ax_delta_LTP.set_xlabel('t')
            # for i in range(0,len(self.delta_LTP)):
            #     ax_delta_LTP.plot(self.tthresh, self.delta_LTP[i])



            #------------------------------------------------#
            #           Plot weigths, calcium, voltage       #
            #                 during learning                #
            #------------------------------------------------#

            if self.exptype in ['xor', 'xor_hom', 'xor_gen', 'xor_spillover', 'xor_sspillover',
                                'xor_hom_spillover', 'xor_shom_spillover', 'xor_cshom_spillover',
                                'xor_zahra_spillover', 'xor_shom_my_spillover', 'xor_shom_my_spillover_stp',
                                'nfbp_inh', 'pattern', 'pattern_homo', 'pattern_homo_bcm', 'pattern_homo_sz',
                                'pattern_hetero_bcm', 'pattern_hetero', 'pattern_hetero_amp', 'pattern_cont_hetero_amp']:
                if self.exptype == 'xor':
                    self.synlist = self.get_synapse_list('adaptive_glutamate2', clustered_flag = True)
                elif self.exptype == 'xor_hom':
                    self.synlist = self.get_synapse_list('adaptive_glutamate_hom', clustered_flag = True)
                elif self.exptype == 'xor_gen':
                    self.synlist = self.get_synapse_list('adaptive_AMPA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_glutamate2', clustered_flag = False)
                elif self.exptype == 'xor_sspillover':
                    self.synlist = self.get_synapse_list('adaptive_sNMDA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_sNMDA', clustered_flag = False)
                elif self.exptype == 'xor_hom_spillover':
                    self.synlist = self.get_synapse_list('adaptive_hom_AMPA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_glutamate_hom', clustered_flag = False)
                elif self.exptype == 'xor_shom_spillover':
                    self.synlist = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = False)
                elif self.exptype == 'xor_shom_my_spillover':
                    self.synlist = self.get_synapse_list('adaptive_my_shom_NMDA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_my_shom_NMDA', clustered_flag = False)
                elif self.exptype == 'xor_shom_my_spillover_stp':
                    self.synlist = self.get_synapse_list('adaptive_shom_NMDA_stp', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_shom_NMDA_stp', clustered_flag = False)
                elif self.exptype == 'xor_zahra_spillover':
                    self.synlist = self.get_synapse_list('adaptive_zahra_NMDA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_zahra_NMDA', clustered_flag = False)
                elif self.exptype == 'xor_cshom_spillover':
                    self.synlist = self.get_synapse_list('adaptive_cshom_AMPA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_cshom_NMDA', clustered_flag = False)
                elif self.exptype == 'xor_test_set':
                    self.synlist = self.get_synapse_list('glutamate_xor_test', clustered_flag = True)
                elif self.exptype == 'nfbp_inh':
                    self.synlist = self.get_synapse_list('adaptive2_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern':
                    self.synlist = self.get_synapse_list('adaptive2_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_homo':
                    self.synlist = self.get_synapse_list('adaptive2_homo_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_homo_sz':
                    self.synlist = self.get_synapse_list('adaptive2_homo_sz_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_homo_bcm':
                    self.synlist = self.get_synapse_list('adaptive2_homo_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_hetero_bcm':
                    self.synlist = self.get_synapse_list('adaptive2_hetero_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_hetero':
                    self.synlist = self.get_synapse_list('adaptive2_hetero_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_hetero_amp':
                    self.synlist = self.get_synapse_list('adaptive2_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
                elif self.exptype == 'pattern_cont_hetero_amp':
                    self.synlist = self.get_synapse_list('adaptive2_cont_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
                self.synlist.sort(key = lambda f: f.sec.name())

                # syn_figs_wampa, syn_axes_wampa = self.plot_helper('wnmda'); #sns.despine()
                #if not self.exptype == 'xor_gen':
#                    syn_figs_wnmda, syn_axes_wnmda = self.plot_helper('wnmda'); sns.despine()
#                    syn_figs_wnmda, syn_axes_wnmda = self.plot_helper('wnmda_e'); sns.despine()
                # syn_figs_ca_nmda, syn_axes_ca_nmda = self.plot_helper('ca_nmda'); sns.despine()
#                syn_figs_ca_nmda_dend, syn_axes_ca_nmda_dend = self.plot_helper('ca_nmda dend'); sns.despine()
                # syn_figs_cali, syn_axes_cali = self.plot_helper('cali')
                # syn_figs_cali, syn_axes_cali = self.plot_helper('cai')
                # syn_figs_cali, syn_axes_cali = self.plot_helper('cao')
                # syn_figs_cali, syn_axes_cali = self.plot_helper('ica')
#                syn_figs_ca_nmda_cali, syn_axes_ca_nmda_cali = self.plot_helper('ca_nmda + cali'); sns.despine()
                if self.exptype in ['xor_hom_spillover', 'nfbp_inh']:
                    syn_figs_thp, syn_axes_thp = self.plot_helper('wnmda');
                    syn_figs_ca_nmda, syn_axes_ca_nmda = self.plot_helper('cai');
                    syn_figs_cali, syn_axes_cali = self.plot_helper('cali')
                    syn_figs_thp, syn_axes_thp = self.plot_helper('thresh_LTP');
                    syn_figs_thd, syn_axes_thd = self.plot_helper('thresh_LTD');
                    # syn_figs_ktm, syn_axes_ktm = self.plot_helper('kernel_theta_min_inh');
                if self.exptype in ['xor_shom_spillover', 'xor_cshom_spillover',
                                    'xor_shom_my_spillover', 'xor_shom_my_spillover_stp'] and not p.long_simulation:
                    syn_figs_lthp, syn_axes_thp = self.plot_helper('lthresh_LTP');
                    syn_figs_hthp, syn_axes_thp = self.plot_helper('hthresh_LTP');
                    syn_figs_thd, syn_axes_thd = self.plot_helper('thresh_LTD');
                    syn_figs_kernel, syn_axes_kernel = self.plot_helper('kernel');
                    syn_figs_kernel, syn_axes_kernel = self.plot_helper('kernel_LTD');
                if self.exptype == 'xor_sspillover':
                    syn_figs_thp, syn_axes_thp = self.plot_helper('lthresh_LTP');
                    syn_figs_thd, syn_axes_thd = self.plot_helper('hthresh_LTP');
                if self.exptype in ['nfbp_inh', 'pattern', 'pattern_homo', 'pattern_homo_bcm',
                                    'pattern_hetero_bcm', 'pattern_hetero', 'pattern_hetero_amp',
                                    'pattern_homo_sz', 'pattern_cont_hetero_amp']:
                    syn_figs_wampa, syn_axes_wampa = self.plot_helper('winh');
                    syn_figs_wampa, syn_axes_wampa = self.plot_helper('theta_inh');
                    syn_figs_wampa, syn_axes_wampa = self.plot_helper('caint_max');
#                if self.exptype in ['xor_shom_spillover', 'xor_cshom_spillover'] and p.distributed_input_size != 0:

                # self.plot_distributed_inputs();
                #------------------------------------------------#
                #           Plot dopamine, error, and            #
                #          calcium in dendrites (commented)      #
                #------------------------------------------------#

                fig_tlast = plt.figure()
                ax_tlast = fig_tlast.add_subplot(111)
                ax_tlast.set_xlabel('t'); ax_tlast.set_ylabel('tlast');
                for i in range(0, len(self.tlast)):
                    ax_tlast.plot(self.tout, self.tlast[i])
                sns.despine()

                fig_asf = plt.figure()
                ax_asf = fig_asf.add_subplot(111)
                ax_asf.set_xlabel('t'); ax_asf.set_ylabel('asf');
                # synlist = self.get_synapse_list('adaptive2_inhexp2syn', clustered_flag = True, drive_type = 'i')
                for i in range(0, len(self.asf)):
                    color, linestyle, marker = self.set_color(self.synlist[i].source)
                    ax_asf.plot(self.tout, self.asf[i], color = color)
                sns.despine()

                fig_dt = plt.figure()
                ax_dt = fig_dt.add_subplot(111)
                ax_dt.set_xlabel('t'); ax_dt.set_ylabel('delta_t');
                for i in range(0, len(self.delta_t)):
                    ax_dt.plot(self.tout, self.delta_t[i])
                sns.despine()

                fig_wuf = plt.figure()
                ax_wuf = fig_wuf.add_subplot(111)
                ax_wuf.set_xlabel('t'); ax_wuf.set_ylabel('Weight update flag');
                ax_wuf.plot(self.tout, self.weight_update_flag)
                sns.despine()

                fig_sf = plt.figure()
                ax_sf = fig_sf.add_subplot(111)
                ax_sf.set_xlabel('t'); ax_sf.set_ylabel('Stimulus flag');
                ax_sf.plot(self.tout, self.stimulus_flag)
                sns.despine()

                if self.exptype not in ['pattern', 'pattern_homo', 'pattern_homo_bcm', 'pattern_hetero_bcm',
                                        'pattern_homo_sz', 'pattern_hetero', 'pattern_hetero_amp',
                                        'pattern_cont_hetero_amp']:
                    fig_d = plt.figure()
                    ax_d = fig_d.add_subplot(111)
                    ax_d.set_xlabel('t'); ax_d.set_ylabel('dopamine');
                    ax_d.plot(self.tout, self.dopamine_vec)
                    sns.despine()

                fig_cali = plt.figure();
                ax_cali = fig_cali.add_subplot(1,1,1);
                ax_cali.set_xlabel('t'); ax_cali.set_ylabel('cali');
                legend = []
                for i in range(0,len(self.cali)):
                    ax_cali.plot(self.tout, self.cali[i])
                    if type(self.dend_record_list) == list:
                        legend.append('dend[%d]' % self.dend_record_list[i])
                    elif type(self.dend_record_list) == dict:
                        legend.append('dend[%d]' % self.dend_record_list['loc'][i])
                ax_cali.legend(legend)
                sns.despine()

                if self.exptype in ['xor_spillover','xor_hom_spillover', 'xor_gen',
                                    'xor_shom_spillover', 'xor_cshom_spillover','xor_sspillover',
                                    'xor_shom_my_spillover', 'xor_shom_my_spillover_stp']:
                    fig_cai_nmda_dend = plt.figure();
                    ax_cai_nmda_dend = fig_cai_nmda_dend.add_subplot(1,1,1);
                    ax_cai_nmda_dend.set_xlabel('t'); ax_cai_nmda_dend.set_ylabel('ca_nmda dend');
                    legend = []
                    for i in range(0,len(self.cai_nmda_dend)):
                        ax_cai_nmda_dend.plot(self.tout, self.cai_nmda_dend[i])
                        if type(self.dend_record_list) == list:
                            legend.append('dend[%d]' % self.dend_record_list[i])
                        elif type(self.dend_record_list) == dict:
                            legend.append('dend[%d]' % self.dend_record_list['loc'][i])
                    ax_cai_nmda_dend.legend(legend)
                    sns.despine()

                # fig_cali_dend = plt.figure()
                # ax_cali_dend = fig_cali_dend.add_subplot(111)
                # ax_cali_dend.set_ylabel('[Cal]_i dend')
                # ax_cali_dend.set_xlabel('t')
                # for c in self.cali_dend:
                #     ax_cali_dend.plot(self.tout, c)

                legend_list = []
                for i in self.dend_record_list:
                    legend_list.append('dend[%d](%.2f) = %.2f um' % (i, p.pos, h.distance(p.pos, sec = self.cell.dendlist[i]) ))
                ax_cali.legend(legend_list);

                fig_m = plt.figure();
                ax_m = fig_m.add_subplot(111);
                ax_m.set_ylabel('Intfire m')
                ax_m.set_xlabel('t')
                for i in range(0,len(self.m)):
                    ax_m.plot(self.t, self.m[i])

                # plt.show()
                return fig_tlast, fig_dt, fig_wuf, fig_sf, fig_cali, fig_m


#                for i in range(0,len(self.synlist)):
#                    if i==0 or (self.synlist[i].sec.name() != self.synlist[i-1].sec.name()):
#                        syn_figs_wampa.append(plt.figure())
#                        syn_axes_wampa.append(syn_figs_wampa[-1].add_subplot(111))
#                        syn_figs_ca_nmda.append(plt.figure())
#                        syn_axes_ca_nmda.append(syn_figs_ca_nmda[-1].add_subplot(111))
#                        syn_figs_cali.append(plt.figure())
#                        syn_axes_cali.append(syn_figs_cali[-1].add_subplot(111))

#
#                        syn_axes_wampa[-1].set_xlabel('t'); syn_axes_wampa[-1].set_ylabel('w_ampa');
#                        syn_axes_wampa[-1].set_ylim(p.LTD_factor*p.gAMPAmax_plateau, p.LTP_factor*p.gAMPAmax_plateau)
#                        syn_axes_ca_nmda[-1].set_xlabel('t'); syn_axes_ca_nmda[-1].set_ylabel('cai_nmda');
#                        syn_axes_cali[-1].set_xlabel('t'); syn_axes_cali[-1].set_ylabel('cali');
#
#                        if not (i == 0):
#                            syn_axes_wampa[-2].legend(legend)
#                            syn_axes_ca_nmda[-2].legend(legend)
#                            syn_axes_cali[-2].legend(legend)
#                            legend = []
#
#                    color, linestyle = self.set_color(self.synlist[i].source)
#                    syn_axes_wampa[-1].plot(self.tout, self.synlist[i].ref_var, color = color, linestyle = linestyle)
#                    syn_axes_ca_nmda[-1].plot(self.tout, self.cai_nmda_in_syns[i])
#                    syn_axes_cali[-1].plot(self.tout, self.cali_in_syns[i])
#
#                    string = '%s(%.2f) = %.2f um' % (self.synlist[i].sec.name(), self.synlist[i].pos,
#                            h.distance(self.synlist[i].pos, sec = self.synlist[i].sec) )
#                    legend.append(string)


        elif self.exptype in ['record_ca', 'inhibitory_plasticity']:
            fig_vs = plt.figure()
            ax_vs = fig_vs.add_subplot(111)
            ax_vs.set_ylabel('Vs')
            if p.simtime < 10000:
                ax_vs.plot(self.tv, self.vs)
                ax_vs.set_xlabel('t (ms)')
            else:
                t = self.tv.to_python()[int(p.skip_first_x_ms*p.nrn_dots_per_1ms):]
                vs =self.vs.to_python()[int(p.skip_first_x_ms*p.nrn_dots_per_1ms):]
                ax_vs.plot(np.asarray(t)*1e-3,vs)
                ax_vs.set_xlabel('t (s)')
            if self.exptype == 'inhibitory_plasticity' and p.simtime >= 10000:
                import efel
                ax_vs2 = ax_vs.twinx()
                ax_vs2.set_ylabel('$\mathrm{f_{out}}$ (Hz)')
                trace = {}
                trace['T'] = self.tv.to_python()
                trace['V'] = self.vs.to_python()
                traces = [trace]
                freqs = []
                trange = np.arange(1000 + p.skip_first_x_ms, p.simtime+1000, 1000)
                for t in np.arange(1000 + p.skip_first_x_ms, p.simtime+1000, 1000):
                    trace['stim_start'] = [t-1000]
                    trace['stim_end'] = [t]
                    res = efel.getFeatureValues(traces, ['mean_frequency'])
                    print(res)
                    try:
                        f = res[0]['mean_frequency'][0]
                    except TypeError as e:
                        f = 0
                    freqs.append(f)
                t = np.divide(np.arange(1000 + p.skip_first_x_ms, p.simtime+1000, 1000), 1000)
                colors = sns.color_palette("Blues", 20)
                ax_vs2.plot(t, freqs, color = colors[4], marker = 'o')

            if self.exptype == 'record_ca':
                fig_cai_soma = plt.figure()
                ax_cai_soma = fig_cai_soma.add_subplot(111)
                ax_cai_soma.set_ylabel('soma cai')
                ax_cai_soma.set_xlabel('t')
                ax_cai_soma.plot(self.tout, self.cai_soma)

            fig_vd = plt.figure();
            ax_vd = fig_vd.add_subplot(111);
            ax_vd.set_ylabel('Vd')
            ax_vd.set_xlabel('t')
            for i in range(0,len(self.vdlist)):
                ax_vd.plot(self.tv, self.vdlist[i])

            # fig_m = plt.figure();
            # ax_m = fig_m.add_subplot(111);
            # ax_m.set_ylabel('Intfire m')
            # ax_m.set_xlabel('t')
            # for i in range(0,len(self.m)):
            #     ax_m.plot(self.tout, self.m[i])


            legend_list = []
            for i in self.dend_record_list:
                legend_list.append('dend[%d](%.2f) = %.2f um' % (i,p.pos, h.distance(p.pos, sec = self.cell.dendlist[i]) ))

#            fig_gk = plt.figure()
#            ax_gk = fig_gk.add_subplot(111)
#            ax_gk.set_ylabel('gk')
#            ax_gk.set_xlabel('t')

#            for i in range(0,len(self.gk)):
#                ax_gk.plot(self.tout, self.gk[i])

            fig_cai = plt.figure();
            ax_cai = fig_cai.add_subplot(111);
            ax_cai.set_ylabel('[Ca]_i')
            ax_cai.set_xlabel('t')
            for i in range(0,len(self.cai)):
                ax_cai.plot(self.tout, self.cai[i])

            fig_cao = plt.figure();
            ax_cao = fig_cao.add_subplot(111);
            ax_cao.set_ylabel('[Ca]_o')
            ax_cao.set_xlabel('t')
            for i in range(0,len(self.cao)):
                ax_cao.plot(self.tout, self.cao[i])

            fig_cali = plt.figure();
            ax_cali = fig_cali.add_subplot(111);
            ax_cali.set_ylabel('[Cal]_i')
            ax_cali.set_xlabel('t')
            for i in range(0,len(self.cali)):
                ax_cali.plot(self.tout, self.cali[i])

            fig_cati = plt.figure();
            ax_cati = fig_cati.add_subplot(111);
            ax_cati.set_ylabel('[Cat]_i')
            ax_cati.set_xlabel('t')
            for i in range(0,len(self.cati)):
                ax_cati.plot(self.tout, self.cati[i])

            fig_cai_nmda = plt.figure();
            ax_cai_nmda = fig_cai_nmda.add_subplot(111);
            ax_cai_nmda.set_ylabel('[Ca]_NMDA')
            ax_cai_nmda.set_xlabel('t')
            for i in range(0,len(self.cai_nmda)):
                ax_cai_nmda.plot(self.tout, self.cai_nmda[i])

            if self.cell.spines != []:
                fig_vspine = plt.figure()
                ax_vspine = fig_vspine.add_subplot(111)
                ax_vspine.set_ylabel('Vspine')
                ax_vspine.set_xlabel('t')
                for v in self.vspine:
                    ax_vspine.plot(self.tv, v)

                fig_cai_nmda_spine = plt.figure();
                ax_cai_nmda_spine = fig_cai_nmda_spine.add_subplot(111);
                ax_cai_nmda_spine.set_ylabel('spine [Ca]_NMDA')
                ax_cai_nmda_spine.set_xlabel('t')
                for i in range(0,len(self.cai_nmda_spine)):
                    ax_cai_nmda_spine.plot(self.tout, self.cai_nmda_spine[i])

                fig_cai_spine = plt.figure();
                ax_cai_spine = fig_cai_spine.add_subplot(111);
                ax_cai_spine.set_ylabel('spine [Ca]')
                ax_cai_spine.set_xlabel('t')
                for i in range(0,len(self.cai_spine)):
                    ax_cai_spine.plot(self.tout, self.cai_spine[i])

                fig_cati_spine = plt.figure();
                ax_cati_spine = fig_cati_spine.add_subplot(111);
                ax_cati_spine.set_ylabel('spine [Cat]')
                ax_cati_spine.set_xlabel('t')
                for i in range(0,len(self.cai_spine)):
                    ax_cati_spine.plot(self.tout, self.cati_spine[i])

                fig_cali_spine = plt.figure();
                ax_cali_spine = fig_cali_spine.add_subplot(111);
                ax_cali_spine.set_ylabel('spine [Cal]')
                ax_cali_spine.set_xlabel('t')
                for i in range(0,len(self.cali_spine)):
                    ax_cali_spine.plot(self.tout, self.cali_spine[i])

                fig_cali_dend = plt.figure()
                ax_cali_dend = fig_cali_dend.add_subplot(111)
                ax_cali_dend.set_ylabel('[Cal]_i dend')
                ax_cali_dend.set_xlabel('t')
                for c in self.cali_dend:
                    ax_cali_dend.plot(self.tout, c)

            legend_list = []
            for i in self.dend_record_list:
                legend_list.append('dend[%d](%.2f) = %.2f um' % (i, p.pos, h.distance(p.pos, sec = self.cell.dendlist[i]) ))

#            ax_vd.legend(legend_list); ax_cai.legend(legend_list);
            ax_cali.legend(legend_list); ax_cai_nmda.legend(legend_list);
            if self.cell.spines != []:
                ax_cali_dend.legend(legend_list)

#            plt.show()
#            return fig_vs, fig_vd, fig_cali, fig_cai_nmda
            if self.exptype == 'inhibitory_plasticity':
                synlist = self.get_synapse_list('adaptive_inhexp2syn', clustered_flag = False, drive_type = 'i')
                max_dist = self.cell.max_dist()
                cmap = plt.cm.Blues
                norm = plt.Normalize(vmin=0, vmax=max_dist)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                fig_w_inh = plt.figure()
                ax_w_inh = fig_w_inh.add_subplot(111)
                ax_w_inh.set_ylabel('$\mathrm{w_{inh}}$')
                ax_w_inh.set_xlabel('t')
                for s in synlist:
                    color = cmap(norm(h.distance(s.pos, sec = s.sec)))
                    t = self.tout[int(p.skip_first_x_ms*p.nrn_dots_per_1ms):]
                    w = s.ref_var_w_inh.to_python()[int(p.skip_first_x_ms*p.nrn_dots_per_1ms):]
                    ax_w_inh.plot(t, w, color = color)
                cbar = fig_w_inh.colorbar(sm, ax=ax_w_inh)
                cbar.set_label('Distance')

                fig_kernel_inh = plt.figure()
                ax_kernel_inh = fig_kernel_inh.add_subplot(111)
                ax_kernel_inh.set_ylabel('kernel_inh')
                ax_kernel_inh.set_xlabel('t')
                for k in self.kernel_inh:
                    ax_kernel_inh.plot(self.tout, k)

            fig_caint = plt.figure();
            ax_caint = fig_caint.add_subplot(111);
            ax_caint.set_ylabel('F ($\mathrm{\mu}$M ms)')
            ax_caint.set_xlabel('t')
            t = self.tout[int(p.skip_first_x_ms*p.nrn_dots_per_1ms):]
            for i in range(0,len(self.caint)):
                c = (self.caint[i]).to_python()[int(p.skip_first_x_ms*p.nrn_dots_per_1ms):]
                ax_caint.plot(t, np.multiply(c,1000))

            legend_list = []
            for i in self.dend_record_list:
                legend_list.append('dend[%d](%.2f) = %.2f um' % (i,p.pos, h.distance(p.pos, sec = self.cell.dendlist[i]) ))

        if self.exptype in ['record_i']:
#            fig_vs = plt.figure()
#            ax_vs = fig_vs.add_subplot(111)
#            ax_vs.set_ylabel('Vs')
#            ax_vs.set_xlabel('t')
#            ax_vs.plot(self.tv, self.vs)
#
#            fig_vd = plt.figure();
#            ax_vd = fig_vd.add_subplot(111);
#            ax_vd.set_ylabel('Vd')
#            ax_vd.set_xlabel('t')

#            for i in range(0,len(self.vdlist)):
#                ax_vd.plot(self.tv, self.vdlist[i])
#
#            legend_list = []
#            for i in self.dend_record_list:
#                legend_list.append('dend[%d](%.2f) = %.2f um' % (i,p.pos, h.distance(p.pos, sec = self.cell.dendlist[i]) ))

            fig_ik = plt.figure()
            ax_ik = fig_ik.add_subplot(111)
            ax_ik.set_ylabel('ik')
            ax_ik.set_xlabel('t')
            for i in range(0,len(self.ik)):
                ax_ik.plot(self.tout, self.ik[i])

            fig_ica_nmda = plt.figure();
            ax_ica_nmda = fig_ica_nmda.add_subplot(111);
            ax_ica_nmda.set_ylabel('ica_NMDA')
            ax_ica_nmda.set_xlabel('t')
            ax_ica_nmda.plot(self.tout, sum(self.ica_nmda))

            fig_ina = plt.figure()
            ax_ina = fig_ina.add_subplot(111)
            ax_ina.set_ylabel('ina')
            ax_ina.set_xlabel('t')
            for i in range(0,len(self.ina)):
                ax_ina.plot(self.tout, self.ina[i])

#            fig_iampa = plt.figure()
#            ax_iampa = fig_iampa.add_subplot(111)
#            ax_iampa.set_ylabel('iampa')
#            ax_iampa.set_xlabel('t')
#            ax_iampa.plot(self.tout, sum(self.iampa))

            fig_inmda = plt.figure()
            ax_inmda = fig_inmda.add_subplot(111)
            ax_inmda.set_ylabel('inmda')
            ax_inmda.set_xlabel('t')
            ax_inmda.plot(self.tout, sum(self.iNMDA))

            fig_i = plt.figure()
            ax_i = fig_i.add_subplot(111)
            ax_i.set_ylabel('i')
            ax_i.set_xlabel('t')
            for i in range(0,len(self.ik)):
                ax_i.plot(self.tout, self.ik[i]+self.ina[i]+sum(self.ica_nmda[i])+sum(self.iampa)+sum(self.iNMDA))

            # plt.show()
            return fig_i, fig_ik, fig_inmda, fig_ica_nmda, fig_ina

    def plot_distributed_inputs(self):
        figs_agh = []; ax_agh = []
        figs_v_agh = []; ax_v_agh = []
        figs_cali_agh = []; ax_cali_agh = []
        figs_cai_agh = []; ax_cai_agh = []
        figs_cai_nmda_agh = []; ax_cai_nmda_agh = []
        figs_kernel_agh = []; ax_kernel_agh = []
        figs_kernel_LTD_agh = []; ax_kernel_LTD_agh = []
        figs_lthresh_LTD_agh = []; ax_lthresh_LTD_agh = []
        figs_lthresh_LTP_agh = []; ax_lthresh_LTP_agh = []
        figs_cai_nmda_cali_agh = []; ax_cai_nmda_cali_agh = []
        if p.adaptive_distributed_inputs == True:
            if not p.correlated_distributed_inputs:
                figs_agh.append(plt.figure())
                ax_agh.append(figs_agh[-1].add_subplot(111))
                ax_agh[-1].set_xlabel('t'); ax_agh[-1].set_ylabel('w_ampa');

                figs_cali_agh.append(plt.figure())
                ax_cali_agh.append(figs_cali_agh[-1].add_subplot(111))
                ax_cali_agh[-1].set_xlabel('t'); ax_cali_agh[-1].set_ylabel('cali agh');

                figs_cai_agh.append(plt.figure())
                ax_cai_agh.append(figs_cai_agh[-1].add_subplot(111))
                ax_cai_agh[-1].set_xlabel('t'); ax_cai_agh[-1].set_ylabel('cai agh');

                figs_lthresh_LTP_agh.append(plt.figure())
                ax_lthresh_LTP_agh.append(figs_lthresh_LTP_agh[-1].add_subplot(111))
                ax_lthresh_LTP_agh[-1].set_xlabel('t'); ax_lthresh_LTP_agh[-1].set_ylabel('lthresh_LTP agh');
#
                figs_cai_nmda_agh.append(plt.figure())
                ax_cai_nmda_agh.append(figs_cai_nmda_agh[-1].add_subplot(111))
                ax_cai_nmda_agh[-1].set_xlabel('t'); ax_cai_nmda_agh[-1].set_ylabel('ca_nmda agh');

                figs_cai_nmda_cali_agh.append(plt.figure())
                ax_cai_nmda_cali_agh.append(figs_cai_nmda_cali_agh[-1].add_subplot(111))
                ax_cai_nmda_cali_agh[-1].set_xlabel('t'); ax_cai_nmda_cali_agh[-1].set_ylabel('ca_nmda x cali agh');

                figs_v_agh.append(plt.figure())
                ax_v_agh.append(figs_v_agh[-1].add_subplot(111))
                ax_v_agh[-1].set_xlabel('t'); ax_v_agh[-1].set_ylabel('V agh');

                figs_kernel_agh.append(plt.figure())
                ax_kernel_agh.append(figs_kernel_agh[-1].add_subplot(111))
                ax_kernel_agh[-1].set_xlabel('t'); ax_kernel_agh[-1].set_ylabel('kernel agh');

                figs_kernel_LTD_agh.append(plt.figure())
                ax_kernel_LTD_agh.append(figs_kernel_agh[-1].add_subplot(111))
                ax_kernel_LTD_agh[-1].set_xlabel('t'); ax_kernel_LTD_agh[-1].set_ylabel('kernel LTD agh');

                for i in range(0,p.distributed_input_size):

                    nmda = np.asarray(self.synlist_agh[i].ref_var_nmda.to_python())
                    ax_agh[-1].plot(self.tthresh, nmda)
                    if not p.long_simulation:
                        ax_cali_agh[-1].plot(self.tout, self.cali_agh[i])
                        ax_cai_agh[-1].plot(self.tout, self.cai_agh[i])
                        ax_cai_nmda_agh[-1].plot(self.tout, self.cai_nmda_agh[i])
                        ax_lthresh_LTP_agh[-1].plot(self.tthresh, self.synlist_agh[i].ref_var_lthresh_LTP.to_python())
                        ax_v_agh[-1].plot(self.tv, self.v_agh[i])
                        ax_kernel_agh[-1].plot(self.tout, self.kernel_agh[i])
                        ax_kernel_LTD_agh[-1].plot(self.tout, self.kernel_LTD_agh[i])
                        ax_cai_nmda_cali_agh[-1].plot(self.tout, np.add(np.add(self.cai_nmda_agh[i], self.cali_agh[i]), self.cai_agh[i]))

            else:
                num_groups = 4
                for i in range(0, num_groups):
                    figs_agh.append(plt.figure())
                    ax_agh.append(figs_agh[-1].add_subplot(111))
                    ax_agh[-1].set_xlabel('t'); ax_agh[-1].set_ylabel('w_ampa');
                    ax_agh[-1].set_title('Group %d' % i);
#
                    figs_cali_agh.append(plt.figure())
                    ax_cali_agh.append(figs_cali_agh[-1].add_subplot(111))
                    ax_cali_agh[-1].set_xlabel('t'); ax_cali_agh[-1].set_ylabel('cali agh');
                    ax_cali_agh[-1].set_title('Group %d' % i);
#
                    figs_lthresh_LTD_agh.append(plt.figure())
                    ax_lthresh_LTD_agh.append(figs_lthresh_LTD_agh[-1].add_subplot(111))
                    ax_lthresh_LTD_agh[-1].set_xlabel('t'); ax_lthresh_LTD_agh[-1].set_ylabel('lthresh_LTD agh');
                    ax_lthresh_LTD_agh[-1].set_title("Group %d" % i)
#
                    figs_cai_nmda_agh.append(plt.figure())
                    ax_cai_nmda_agh.append(figs_cai_nmda_agh[-1].add_subplot(111))
                    ax_cai_nmda_agh[-1].set_xlabel('t'); ax_cai_nmda_agh[-1].set_ylabel('ca_nmda agh');
                    ax_cai_nmda_agh[-1].set_title('Group %d' %i);
#                        figs_v_agh.append(plt.figure())
#                        ax_v_agh.append(figs_v_agh[-1].add_subplot(111))
#                        ax_v_agh[-1].set_xlabel('t'); ax_v_agh[-1].set_ylabel('V agh');
#                        ax_v_agh[-1].set_title("Group %d" % n)
#
                    for n in range(0,p.distributed_input_size):
                        idx = i*p.distributed_input_size+n
                        nmda = np.asarray(self.synlist_agh[idx].ref_var_nmda.to_python())
                        ax_agh[-1].plot(self.tthresh, nmda)
                        ax_cali_agh[-1].plot(self.tout, self.cali_agh[idx])
                        ax_cai_nmda_agh[-1].plot(self.tout, self.cai_nmda_agh[idx])

    def plot_helper(self, plot_what):
        if self.exptype == 'xor':
            self.synlist = self.get_synapse_list('adaptive_glutamate2', clustered_flag = True)
            l = self.synlist
        elif self.exptype == 'xor_hom':
            self.synlist = self.get_synapse_list('adaptive_glutamate_hom', clustered_flag = True)
            l = self.synlist
        elif self.exptype == 'xor_gen':
            self.synlist = self.get_synapse_list('generalized_rule', clustered_flag = True)
            l = self.synlist
        elif self.exptype == 'xor_spillover':
            self.synlist = self.get_synapse_list('adaptive_AMPA', clustered_flag = True)
            self.synlist_n = self.get_synapse_list('adaptive_NMDA', clustered_flag = True)
            self.synlist_nc = [s for s in self.synlist_n if 'spine' in s.sec.name()]
            self.synlist_ne = [s for s in self.synlist_n if not('spine' in s.sec.name())]
            l = self.synlist
            if plot_what in ['ca_nmda dend', 'wnmda_e']:
                l = self.synlist_ne
            elif plot_what == 'ca_nmda':
                l = self.synlist_nc
            elif plot_what == 'wnmda':
                l = self.synlist_nc
        elif self.exptype == 'xor_sspillover':
            self.synlist = self.get_synapse_list('adaptive_sAMPA', clustered_flag = True)
            self.synlist_n = self.get_synapse_list('adaptive_sNMDA', clustered_flag = True)
            self.synlist_nc = [s for s in self.synlist_n if 'spine' in s.sec.name()]
            self.synlist_ne = [s for s in self.synlist_n if not('spine' in s.sec.name())]
            l = self.synlist
            if plot_what in ['ca_nmda dend', 'wnmda_e']:
                l = self.synlist_ne
            elif plot_what in ['ca_nmda', 'lthresh_LTP', 'hthresh_LTP']:
                l = self.synlist_nc
            elif plot_what == 'wnmda':
                l = self.synlist_nc
        elif self.exptype in ['xor_hom_spillover', 'xor_shom_spillover', 'xor_cshom_spillover',
                              'xor_zahra_spillover', 'xor_shom_my_spillover', 'xor_shom_my_spillover_stp']:
            if self.exptype == 'xor_hom_spillover':
                self.synlist = self.get_synapse_list('adaptive_hom_AMPA', clustered_flag = True)
                self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
                self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
    #            self.synlist_nc = [s for s in self.synlist_n if 'spine' in s.sec.name()]
    #            self.synlist_ne = [s for s in self.synlist_n if not('spine' in s.sec.name())]
            elif self.exptype == 'xor_shom_spillover':
                self.synlist = self.get_synapse_list('adaptive_shom_AMPA', clustered_flag = True)
                self.synlist_nc = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
                self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
                self.synlist_pf = self.get_synapse_list('adaptive_pf_NMDA', clustered_flag = True)

            elif self.exptype == 'xor_shom_my_spillover':
                self.synlist = self.get_synapse_list('adaptive_shom_AMPA', clustered_flag = True)
                self.synlist_nc = self.get_synapse_list('adaptive_my_shom_NMDA', clustered_flag = True)
#                self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
                self.synlist_pf = self.get_synapse_list('adaptive_pf_NMDA', clustered_flag = True)

            elif self.exptype == 'xor_shom_my_spillover_stp':
                self.synlist = self.get_synapse_list('adaptive_shom_AMPA_stp', clustered_flag = True)
                self.synlist_nc = self.get_synapse_list('adaptive_shom_NMDA_stp', clustered_flag = True)
#                self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
                self.synlist_pf = self.get_synapse_list('adaptive_pf_NMDA', clustered_flag = True)

            elif self.exptype == 'xor_cshom_spillover':
                self.synlist = self.get_synapse_list('adaptive_cshom_AMPA', clustered_flag = True)
                self.synlist_nc = self.get_synapse_list('adaptive_cshom_NMDA', clustered_flag = True)
                self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)

            elif self.exptype == 'xor_zahra_spillover':
                self.synlist = self.get_synapse_list('adaptive_zahra_AMPA', clustered_flag = True)
                self.synlist_nc = self.get_synapse_list('adaptive_zahra_NMDA', clustered_flag = True)
                self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
            l = self.synlist_nc

        elif self.exptype in ['nfbp_inh']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            self.synlist = self.get_synapse_list('adaptive2_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

        elif self.exptype in ['pattern']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            self.synlist = self.get_synapse_list('adaptive2_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

        elif self.exptype in ['pattern_homo']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            self.synlist = self.get_synapse_list('adaptive2_homo_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

        elif self.exptype in ['pattern_homo_sz']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            self.synlist = self.get_synapse_list('adaptive2_homo_sz_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

        elif self.exptype in ['pattern_homo_bcm']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            self.synlist = self.get_synapse_list('adaptive2_homo_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

        elif self.exptype in ['pattern_hetero_bcm']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            self.synlist = self.get_synapse_list('adaptive2_hetero_bcm_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

        elif self.exptype in ['pattern_hetero', 'pattern_hetero_amp', 'pattern_cont_hetero_amp']:
            self.synlist_nc = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            if self.exptype == 'pattern_hetero':
                self.synlist = self.get_synapse_list('adaptive2_hetero_inhexp2syn', clustered_flag = True, drive_type = 'i')
            elif self.synlist == 'pattern_hetero_amp':
                self.synlist = self.get_synapse_list('adaptive2_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
            elif self.synlist == 'pattern_cont_hetero_amp':
                self.synlist = self.get_synapse_list('adaptive2_cont_hetero_amp_inhexp2syn', clustered_flag = True, drive_type = 'i')
            l = self.synlist_nc
            l_inh = self.synlist

            if plot_what in ['ca_nmda dend', 'wnmda_e']:
                l = self.synlist_ne
            elif plot_what in ['ca_nmda', 'cali']:
                l = self.synlist_nc
            elif plot_what == 'wnmda' or plot_what in ['thresh_LTP', 'thresh_LTD',
                                                       'lthresh_LTP', 'hthresh_LTP']:
                l = self.synlist_nc

        elif self.exptype == 'xor_test_set':
            self.synlist = self.get_synapse_list('glutamate_xor_test', clustered_flag = True)
            l = self.synlist
        self.synlist.sort(key = lambda f: f.sec.name())
        l.sort(key = lambda f: f.sec.name())

        xlabel = 't (s)'
        if self.exptype in ['xor_spillover','xor_hom_spillover','xor_shom_spillover',
                            'xor_cshom_spillover','xor_sspillover', 'xor_zahra_spillover',
                            'xor_shom_my_spillover', 'xor_shom_my_spillover_stp']:
            syn_strength = p.weight
        else:
            if plot_what == 'wampa':
                syn_strength = p.gAMPAmax_plateau
            elif plot_what == 'wnmda':
                syn_strength = 0.5*(p.start_weight + p.end_weight)
            elif plot_what == 'wnmda_e':
                syn_strength = p.gNMDAmax_plateau

        if plot_what == 'wampa':
            yval = np.asarray([np.asarray(s.ref_var_ampa.to_python())/syn_strength for s in l])*100
            ylabel = '  AMPA conductance\n(% of initial value)'
            yliml = yval.min()#p.gAMPAmax_plateau*p.scale_conductance
            ylimu = yval.max()#p.gAMPAmax_plateau*p.scale_conductance
            if self.exptype == 'xor_gen':
                yliml = yval.min()
                ylimu = yval.max()
        elif plot_what == 'wnmda':
            yval = np.asarray([np.asarray(s.ref_var_nmda.to_python())/syn_strength for s in l])*100
            ylabel = '  $w_{exc}$\n(% of initial value)'
            yliml = yval.min()#p.gNMDAmax_plateau*p.scale_conductance
            ylimu = yval.max()#p.gNMDAmax_plateau*p.scale_conductance
        elif plot_what == 'wnmda_e':
            yval = np.asarray([np.asarray(s.ref_var_nmda.to_python())/syn_strength for s in l])*100
            ylabel = ' eNMDA conductance\n(% of initial value)'
            yliml = yval.min()#p.gNMDAmax_plateau*p.scale_conductance
            ylimu = yval.max()#p.gNMDAmax_plateau*p.scale_conductance
        elif plot_what == 'winh':
            yval = np.asarray([np.asarray(s.ref_var_w_inh.to_python())/p.weight_inh for s in l_inh])*100
            ylabel = ' $w_{inh}$ \n (% of initial value)'
            yliml = yval.min()#p.gNMDAmax_plateau*p.scale_conductance
            ylimu = yval.max()#p.gNMDAmax_plateau*p.scale_conductance
        elif plot_what == 'theta_inh':
            yval = np.asarray([np.asarray(s.ref_var_theta_inh.to_python()) for s in l_inh])
            ylabel = r'$\theta_{inh}$'
            yliml = yval.min()#p.gNMDAmax_plateau*p.scale_conductance
            ylimu = yval.max()#p.gNMDAmax_plateau*p.scale_conductance
        elif plot_what == 'caint_max':
            yval = np.asarray([np.asarray(s.ref_var_theta_inh.to_python()) for s in l_inh])
            ylabel = 'max(F)'
            yliml = yval.min()#p.gNMDAmax_plateau*p.scale_conductance
            ylimu = yval.max()#p.gNMDAmax_plateau*p.scale_conductance
        elif plot_what == 'ca_nmda':
            yval = self.cai_nmda_in_syns
            ylabel = 'ca_nmda (mM)'
        elif plot_what == 'ca_nmda + cali':
            self.cai_nmda_cali_in_syns = [np.add(np.add(cn, cl), c) for cn,cl,c in
                                          zip(self.cai_nmda_in_syns, self.cali_in_syns, self.cai_in_syns)]
            yval = self.cai_nmda_cali_in_syns
            ylabel = 'ca_nmda x cali(mM)'
        elif plot_what == 'ca_nmda dend':
            yval = self.cai_nmda_dend
            ylabel = 'ca_nmda dend (mM)'
        elif plot_what == 'cali':
            yval = self.cali_in_syns
            ylabel = 'cali (mM)'
        elif plot_what == 'cati':
            yval = self.cati_in_syns
            ylabel = 'cati (mM)'
        elif plot_what == 'cai':
            yval = self.cai_in_syns
            ylabel = 'cai (mM)'
        elif plot_what == 'cao':
            yval = self.cao
            ylabel = 'cao (mM)'
        elif plot_what == 'ica':
            yval = self.ica_in_syns
            ylabel = 'ica'
        elif plot_what == 'thresh_LTP':
            yval = [s.ref_var_lthresh_LTP for s in l]
            ylabel = 'thresh_LTP'
            # ylimu = p.thresh_LTP_max
            # yliml = p.thresh_LTP_min
        elif plot_what == 'thresh_LTD':
            yval = [s.ref_var_lthresh_LTD for s in l]
            ylabel = 'thresh_LTD'
#            ylimu = p.thresh_LTD_max
#            yliml = p.thresh_LTD_min
        elif plot_what == 'lthresh_LTP':
            yval = [s.ref_var_lthresh_LTP for s in l]
            ylabel = 'lthresh_LTP'
#            ylimu = p.thresh_LTP_max
#            yliml = p.thresh_LTP_min
        elif plot_what == 'hthresh_LTP':
            yval = [s.ref_var_hthresh_LTP for s in l]
            ylabel = 'hthresh_LTP'
#            ylimu = p.thresh_LTP_max
#            yliml = p.thresh_LTP_min
        elif plot_what == 'kernel':
            yval = self.kernel
            ylabel = 'kernel'
        elif plot_what == 'kernel_theta_min_inh':
            yval = [s.ref_var_kernel_theta_min_inh for s in l]
            ylabel = 'kernel_theta_min_inh'
        elif plot_what == 'kernel_LTD':
            yval = self.kernel_LTD
            ylabel = 'kernel_LTD'
        figs = []; axess = []; legend = []

        if plot_what in ['winh', 'theta_inh', 'kernel_theta_min_inh', 'caint_max']:
            helper_list = l_inh
        else:
            helper_list = l

        r = [si for s in helper_list for si in re.findall("\[\d+\]", s.sec.name()) ]
        r = [ int(num) for elem in r for num in re.findall("\d+", elem)]
#        if len(p.input_dends) == 4:
#            new_plot = -1
#            fig, axes = plt.subplots(1,4, sharey = True)
#            for i in range(0, len(self.synlist)):
#                if i==0 or (r[i] != r[i-1]):
#                    new_plot +=1
#                if plot_what == 'wampa' or plot_what == 'wnmda' or plot_what == 'thresh_LTP':
#                    color, linestyle, marker = self.set_color(self.synlist[i].source)
#                    axes[new_plot].plot(self.tout, yval[i], color = color, linestyle = linestyle)
#                else:
#                    axes[new_plot].plot(self.tout, yval[i])
#                    if plot_what == 'ca_nmda' and self.exptype == 'xor':
#                        axes[new_plot].plot(self.tout, [p.thresh_LTP]*len(self.tout),
#                            linestyle = '--', color = 'grey')
#                    elif plot_what == 'cali' and self.exptype == 'xor':
#                        axes[new_plot].plot(self.tout, [p.thresh_LTD]*len(self.tout),
#                            linestyle = '--', color = 'grey')
#
#                if not (i == 0) and (r[i] != r[i-1]):
#                    axes[new_plot].set_title("Dendrite %d" % r[i])
#
#            for a in axes:
#                if plot_what == 'wampa' or plot_what == 'wnmda' or plot_what == 'thresh_LTP':
#                    if p.simtime > 1000:
#                        a.xaxis.set_ticks([0, 1, 2, 3])
#                        a.set_xlabel(xlabel)
#            if plot_what == 'wampa' or plot_what == 'wnmda' or plot_what == 'thresh_LTP':
#                axes[0].set_ylim(yliml, ylimu)
#                axes[0].yaxis.set_ticks([50, 100, 150])
#                axes[0].set_ylabel(ylabel)
#                axes[0].set_title("Dendrite %d" % r[0])
#            figs.append(fig); axess.append(axes)
        if plot_what in ['winh', 'theta_inh', 'kernel_theta_min_inh', 'caint_max']:
            end_range = len(l_inh)
        else:
            end_range = len(l)
        for i in range(0,end_range):
            if i==0 or (r[i] != r[i-1]):
                figs.append(plt.figure(figsize = (p.fig_width, p.fig_height)))
                axess.append(figs[-1].add_subplot(111))
#                    sns.despine()
                axess[-1].set_xlabel(xlabel)
                axess[-1].set_ylabel(ylabel)

                if plot_what in ['wampa', 'wnmda']:
                    axess[-1].set_ylim(yliml-5, ylimu+5)
                    axess[-1].yaxis.set_ticks(p.weight_ticks)
                # if plot_what in ['wampa', 'wnmda', 'thresh_LTP']:
                    # if p.simtime > 1000:
                    #     axess[-1].xaxis.set_ticks(np.arange(0,self.tout[-1], 10))
                # if plot_what in ['wampa', 'wnmda']:
                #     # axess[-1].set_ylim(yliml, ylimu)
                #     print("Entered here before setting exc ticks.")
                #     axess[-1].yaxis.set_ticks([50, 100, 150, 200])
                        # axess[-1].xaxis.set_ticks(np.arange(0,self.tthresh[-1], 100000))
            if plot_what in ['wampa','wnmda','thresh_LTP', 'lthresh_LTP', 'hthresh_LTP',
                             'thresh_LTD', 'winh', 'theta_inh', 'kernel_theta_min_inh', 'caint_max']:
                if plot_what in ['winh', 'theta_inh', 'kernel_theta_min_inh', 'caint_max']:
                    color, linestyle, marker = self.set_color(l_inh[i].source)
                    print(r[i], l_inh[i].source, color, l_inh[i].obj.weight)
                    if l_inh[i].source == 1:
                        linewidth = p.linewidth_A
                    elif l_inh[i].source == 2:
                        linewidth = p.linewidth_B
                    elif l_inh[i].source == 3:
                        linewidth = p.linewidth_C
                    elif l_inh[i].source == 4:
                        linewidth = p.linewidth_D
                else:
                    color, linestyle, marker = self.set_color(l[i].source)
                if p.simtime > 10000:
                    t = np.asarray(self.t)/1000
                    tthresh = np.asarray(self.tthresh)/1000
                else:
                    t = self.t
                    tthresh = self.tthresh
                if plot_what in ['theta_inh', 'caint_max']:
                    if i%p.inh_cluster_size == 0:
                        print("Printing only theta_inh for the first synapse in the inhibitory cluster %d" % i)
                        axess[-1].plot(tthresh, yval[i], color = color, linestyle = linestyle, linewidth = linewidth)
                        axess[-1].set_ylim([p.ymin_theta_inh, p.ymax_theta_inh])
                        axess[-1].set_yticks([0, int(p.tick_theta_inh/2), p.tick_theta_inh])
                else:
                    axess[-1].plot(tthresh, yval[i], color = color, linestyle = linestyle)

                if plot_what == 'winh':
                    axess[-1].set_ylim([p.ymin_winh, p.ymax_winh])
                    axess[-1].set_yticks([0, p.tick_winh])

                elif plot_what == 'thresh_LTP':
                    inds, peaks = ss.find_peaks(self.cai_nmda_max[i].to_python(), height = 0.001)
                    t = np.asarray(self.tout)[list(inds)]
                    axess[-1].plot(t, peaks['peak_heights'], color = color, linestyle = '', marker = 'o')

                elif plot_what == 'thresh_LTD':
                    inds, peaks = ss.find_peaks(self.cali_max[i].to_python(), height = 0.001)
                    t = np.asarray(self.tout)[list(inds)]
                    axess[-1].plot(t, peaks['peak_heights'], color = color, linestyle = '', marker = 'o')

                elif plot_what in ['lthresh_LTP', 'hthresh_LTP']:
                    inds, peaks = ss.find_peaks(l[i].ref_var_cai.to_python(), height = 0.0001)
                    t = np.asarray(self.tout)[list(inds)]
                    axess[-1].plot(t*1000, peaks['peak_heights'], color = color, linestyle = '', marker = 'o')

                elif plot_what == 'theta_inh':
                    if i%p.inh_cluster_size == 0:
                        print("Printing only caint for the first synapse in the inhibitory cluster %d" % i)
                        axess[-1].plot(t, np.multiply(p.theta_inh_sf, l_inh[i].ref_var_caint.to_python()), color = color, linewidth = 0.5)
                    if self.exptype in ['pattern']:
                        axess[-1].plot(tthresh, l_inh[i].ref_var_theta_min_inh.to_python(), color = color, linewidth = 0.5)

                elif plot_what == 'caint_max':
                    # axess[-1].plot(t, np.multiply(p.theta_inh_sf, l[i].ref_var_caint_max.to_python()), color = color, linewidth = 0.5)
                    # synapse_indices = self.get_first_syn_in_cluster('i')
                    # if i in synapse_indices:
                    if i%p.inh_cluster_size == 0:
                        print("Printing only theta_inh for the first synapse in the inhibitory cluster %d" % i)
                        inds, peaks = ss.find_peaks( np.multiply(p.theta_inh_sf, l_inh[i].ref_var_caint_max.to_python()), height = 0.01)
                        inds2 = inds[p.max_ca_first_elem::p.max_ca_step];
                        peaks2 = peaks['peak_heights'][p.max_ca_first_elem::p.max_ca_step]

                        t = np.asarray(self.tout)[list(inds2)]
                        axess[-1].plot(t, peaks2, color = color, linestyle = '', marker = 'o', markersize = 2.0)
                    if self.exptype in ['pattern']:
                        axess[-1].plot(tthresh, l_inh[i].ref_var_theta_min_inh.to_python(), color = color, linewidth = 0.5)

            elif plot_what in ['cai', 'ca_nmda', 'cali', 'ica']:
                color, linestyle, marker = self.set_color(l[i].source)
                axess[-1].plot(self.t, yval[i], color = color, linestyle = linestyle)
            else:
                axess[-1].plot(self.tout, yval[i])

            if not (i == 0) and (r[i] != r[i-1]):
#                axes[-2].legend(legend)
                legend = []
                axess[-2].set_title("Dendrite %d" % r[i-1])
            if plot_what in ['winh', 'theta_inh', 'kernel_theta_min_inh', 'caint_max']:
                element = l_inh[i]
            else:
                element = l[i]
            string = '%s(%.2f) = %.2f um' % (element.sec.name(), element.pos,
                    h.distance(element.pos, sec = element.sec) )
            legend.append(string)

#        axes[-1].legend(legend)
        print(axess[-1], i, len(r))
        axess[-1].set_title("Dendrite %d" % r[i-1])

        return figs, axess

    def get_first_syn_in_cluster(self, drive_type):
        synlist = [0]
        if drive_type == 'e':
            syns = self.dendstatobj.dend_syns
        elif drive_type == 'i':
            syns = self.dendstatobj.dend_inh_syns
        else:
            print("From method  get_first_syn_in_cluster:")
            print("drive_type %s not supported" % drive_type)
            sys.exit(-1)

        for sl in syns:
            synlist.extend(sl)

        for i,el in enumerate(synlist[1:]):
            synlist[i+1] = synlist[i] + synlist[i+1]

        return synlist

    def write_results(self):

        if type(self.dendstatobj) == type(dsd.DendStat()):
            if self.exptype in ['xor', 'xor_hom', 'xor_spillover', 'xor_hom_spillover',
                                'xor_shom_spillover', 'xor_cshom_spillover', 'xor_sspillover']:
                if self.exptype == 'xor':
                    self.synlist = self.get_synapse_list('adaptive_glutamate2', clustered_flag = True)

                elif self.exptype == 'xor_hom':
                    self.synlist = self.get_synapse_list('adaptive_glutamate_hom', clustered_flag = True)

                elif self.exptype == 'xor_spillover':
                    self.synlist_a = self.get_synapse_list('adaptive_AMPA', clustered_flag = True)
                    synlist_nmda = self.get_synapse_list('adaptive_NMDA', clustered_flag = True)
                    self.synlist_n = []
                    self.synlist_ne = []
                    for syn in synlist_nmda:
                        if 'spine' in syn.sec.name():
                            self.synlist_n.append(syn)
                        elif not ('spine' in syn.sec.name()):
                            self.synlist_ne.append(syn)
                    self.synlist = self.synlist_a

                elif self.exptype == 'xor_sspillover':
                    self.synlist_a = self.get_synapse_list('adaptive_sAMPA', clustered_flag = True)
                    synlist_nmda = self.get_synapse_list('adaptive_sNMDA', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_sNMDA', clustered_flag = False)
                    self.synlist_n = []
                    self.synlist_ne = []
                    for syn in synlist_nmda:
                        if 'spine' in syn.sec.name():
                            self.synlist_n.append(syn)
                        elif not ('spine' in syn.sec.name()):
                            self.synlist_ne.append(syn)
                    self.synlist = self.synlist_a

                elif self.exptype == 'xor_hom_spillover':
                    self.synlist_a = self.get_synapse_list('adaptive_hom_AMPA', clustered_flag = True)
                    synlist_nmda = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
                    self.synlist_n = synlist_nmda
                    self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_glutamate_hom', clustered_flag = False)

#                    for syn in synlist_nmda:
#                        if 'spine' in syn.sec.name():
#                            self.synlist_n.append(syn)
#                        elif not ('spine' in syn.sec.name()):
#                            self.synlist_ne.append(syn)
                    self.synlist = self.synlist_a
#                self.synlist.sort(key = lambda f: (f.sec.name(), f.source))
                elif self.exptype == 'xor_shom_spillover':
#                    self.synlist_a = self.get_synapse_list('adaptive_shom_AMPA', clustered_flag = True)
                    synlist_nmda = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
                    self.synlist_n = synlist_nmda
#                    self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = False)
#                    self.synlist = self.synlist_a

                elif self.exptype == 'xor_cshom_spillover':
                    self.synlist_a = self.get_synapse_list('adaptive_cshom_AMPA', clustered_flag = True)
                    synlist_nmda = self.get_synapse_list('adaptive_cshom_NMDA', clustered_flag = True)
                    self.synlist_n = synlist_nmda
                    self.synlist_ne = self.get_synapse_list('adaptive_NMDAe', clustered_flag = True)
                    self.synlist_agh = self.get_synapse_list('adaptive_glutamate_cshom', clustered_flag = False)
                    self.synlist = self.synlist_a

                r = [si for s in self.synlist for si in re.findall("\[\d+\]", s.sec.name()) ]
                r = [ int(num) for elem in r for num in re.findall("\d+", elem)]
                print("Len of synlist is %d." % len(self.synlist))
#                syns_ampa = []
                syns_nmda = []
#                syns_nmda_e = []
                for i in range(0,len(self.synlist)):
                    if i!=0 and (self.synlist[i].source != self.synlist[i-1].source) and (r[i] == r[i-1]):
#                        self.dendstatobj.syns_ampa[p.input_dends.index(r[i])].append(syns_ampa)
                        self.dendstatobj.syns_nmda[self.dendstatobj.dends.index(r[i])].append(syns_nmda)
#                        if self.exptype in ['xor_spillover', 'xor_hom_spillover', 'xor_shom_spillover',
#                                            'xor_cshom_spillover', 'xor_sspillover']:
#                            self.dendstatobj.syns_nmda_e[p.input_dends.index(r[i])].append(syns_nmda_e)
                    elif i!=0 and (r[i] != r[i-1]):
#                        self.dendstatobj.syns_ampa[p.input_dends.index(r[i-1])].append(syns_ampa)
                        self.dendstatobj.syns_nmda[self.dendstatobj.dends.index(r[i-1])].append(syns_nmda)
#                        if self.exptype in ['xor_spillover', 'xor_hom_spillover', 'xor_shom_spillover',
#                                            'xor_cshom_spillover', 'xor_sspillover']:
#                            self.dendstatobj.syns_nmda_e[p.input_dends.index(r[i-1])].append(syns_nmda_e)

                    if (self.synlist[i].source != self.synlist[i-1].source) or (r[i] != r[i-1]):
#                        syns_ampa = []
                        syns_nmda = []
#                        syns_nmda_e = []
                    if self.exptype in ['xor', 'xor_hom']:
#                        syns_ampa.append(self.synlist[i].ref_var_ampa.to_python()[-1])
                        syns_nmda.append(self.synlist[i].ref_var_nmda.to_python()[-1])
                    elif self.exptype in ['xor_spillover', 'xor_hom_spillover','xor_shom_spillover',
                                          'xor_cshom_spillover', 'xor_sspillover']:
#                        syns_ampa.append(self.synlist_a[i].ref_var_ampa.to_python()[-1])
                        syns_nmda.append(self.synlist_n[i].ref_var_nmda.to_python()[-1])
#                        syns_nmda_e.append(self.synlist_ne[i].ref_var_nmda.to_python()[-1])

#                self.dendstatobj.syns_ampa[p.input_dends.index(r[i])].append(syns_ampa)
                self.dendstatobj.syns_nmda[self.dendstatobj.dends.index(r[i])].append(syns_nmda)
#                self.dendstatobj.syns_nmda_e[p.input_dends.index(r[i])].append(syns_nmda_e)

                for s in self.distributed_input_synlist:
                    dend = re.findall("\[\d+\]", s.sec.name())
                    dend = re.findall("\d+", dend[0])
                    dend = int(dend[0])
                    if p.adaptive_distributed_inputs:
#                        ampa = s.ref_var_ampa[-1]
                        if not s.type in ['adaptive_glutamate_shom', 'adaptive_glutamate_cshom']:
                            nmda = s.ref_var_nmda[-1]
                        else:
                            nmda = [] # NMDA_AMPA_ratio = 1 in this synapse
                    else:
                        ampa = p.gAMPAmax_plateau
                        nmda = p.gNMDAmax_plateau
                    self.dendstatobj.distributed_inputs.append([dend, s.pos, s.source, ampa, nmda])

                if self.exptype in ['xor_shom_spillover', 'xor_cshom_spillover']:
                    e, verror, window_error = self.error(p.window_error)
                    self.dendstatobj.verror = verror
                    self.dendstatobj.window_error = window_error
                    self.dendstatobj.rewards_delivered = self.rewards_delivered
                    self.dendstatobj.training_set = self.training_set
                    for group in range(0,4):
                        for i in range(0, p.distributed_input_size):
                            self.dendstatobj.syns_ampa_agh[group].append(self.synlist_agh[i+group*p.distributed_input_size].ref_var_ampa.to_python())
                    for v in self.vdlist:
                        self.dendstatobj.vdlist.append(v.to_python())
                    self.dendstatobj.vs = self.vs.to_python()

            elif self.exptype in ['xor_test_set', 'xor_spillover_test']:
                spike_flags = np.zeros(len(self.reward_times))
                spike_times = self.soma_recorder_tvec[0].to_python()
                # spike_times = self.soma_recorder_tvec[0].to_python()
                for st in spike_times:
                    first_update = 0
                    for i, rt in enumerate(self.reward_times):
                        if (st < rt) and (st > rt - p.time_to_reward):
                            if spike_flags[i] == 0:
                                first_update = 1
                            spike_flags[i] = 1
                            break
                    if first_update:
                        if (st < p.first_training_input_start + p.test_set_size_per_group*p.session_length):
                            self.dendstatobj.count_01 += 1
                        elif (st < p.first_training_input_start + 2*p.test_set_size_per_group*p.session_length):
                            self.dendstatobj.count_10 += 1
                        elif (st < p.first_training_input_start + 3*p.test_set_size_per_group*p.session_length):
                            self.dendstatobj.count_11 += 1
                        elif (st < p.first_training_input_start + 4*p.test_set_size_per_group*p.session_length):
                            self.dendstatobj.count_00 += 1

        else:
            print("There is no DendStat object to write the results into.")

    def set_up_experiment(self):
        if self.exptype in ['xor', 'xor_hom', 'xor_test_set', 'xor_gen', 'xor_spillover',
                            'xor_hom_spillover', 'xor_spillover_test','xor_shom_spillover',
                            'xor_cshom_spillover', 'xor_sspillover', 'xor_zahra_spillover',
                            'xor_shom_my_spillover', 'xor_shom_my_spillover_stp', 'nfbp_inh']:
            self.insert_synapses('noise_SPN')
            self.create_dopamine()
            ts = self.create_training_set(p.training_set_size)
            print("Training set"); print(ts)
            self.training_set = ts; self.training_set_copy = self.training_set
            self.rewards_delivered = []
            self.xor_input_times, self.reward_times = self.convert_ts_to_times(ts)
            self.create_xor_inputs(p.xor_input_size)
            self.connect_xor_inputs()
            self.spike_recorder(self.cell.somalist[0], 0.5, -30)
            if p.with_diffusion:
                self.cell.set_up_diffusion()
                self.set_up_diffusion()

        elif self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                              'pattern_homo_bcm', 'pattern_hetero_bcm', 'pattern_hetero',
                              'pattern_hetero_amp','pattern_cont_hetero_amp']:
            self.insert_synapses('noise_SPN')
            self.create_dopamine()
            ts = self.create_training_set(p.training_set_size)
            print("Training set"); print(ts)
            self.training_set = ts; self.training_set_copy = self.training_set
            self.rewards_delivered = []
            self.pattern_input_times, self.reward_times = self.convert_ts_to_times(ts)
            self.create_pattern_inputs(p.pattern_input_size)
            self.connect_pattern_inputs()
            self.spike_recorder(self.cell.somalist[0], 0.5, -10)
            for dend in self.dendstatobj.dends:
                start_pos = p.cluster_start_poss[p.independent_dends.index(dend)]
                self.spike_recorder(self.cell.dendlist[dend], start_pos, -50)

    def simulate(self, simtime = p.simtime, parallel = False):
        start = time.time()
        gmtime = time.gmtime(start)
        print("Starting simulation... %d:%d:%d" %(gmtime.tm_hour + 1, gmtime.tm_min, gmtime.tm_sec))

        if not parallel:
            h.load_file("stdrun.hoc")
            h.init()
            h.tstop = simtime
            fih3 = h.FInitializeHandler((self.seti_print_status))
            if self.exptype in ['xor','xor_hom', 'xor_test_set', 'xor_gen', 'xor_spillover',
                                'xor_hom_spillover', 'xor_spillover_test','xor_shom_spillover',
                                'xor_cshom_spillover', 'xor_sspillover', 'xor_zahra_spillover',
                                'xor_shom_my_spillover', 'xor_shom_my_spillover_stp', 'nfbp_inh',
                                'pattern', 'pattern_homo', 'pattern_homo_sz',
                                'pattern_homo_bcm', 'pattern_hetero_bcm',
                                'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp']:
                fih1 = h.FInitializeHandler((self.seti_dopamine_release, self.reward_times))

#                fih2 = h.FInitializeHandler((self.seti_xor_input_times, self.xor_input_times))
#                fih4 = h.FInitializeHandler((self.seti_reset_calcium, self.xor_input_times))
                if self.exptype in ['xor_gen', 'xor_spillover', 'xor_hom_spillover',
                                    'xor','xor_hom','xor_shom_spillover','xor_cshom_spillover',
                                    'xor_sspillover', 'xor_zahra_spillover', 'xor_shom_my_spillover',
                                    'xor_shom_my_spillover_stp', 'nfbp_inh']:
                    fih2 = h.FInitializeHandler((self.seti_stimulus_indicators, self.xor_input_times))
                    fih_wu = h.FInitializeHandler((self.seti_weight_update_indicators, self.xor_input_times))
                if self.exptype in ['xor_shom_my_spillover', 'xor_shom_my_spillover_stp', 'nfbp_inh']:
                    fih5 = h.FInitializeHandler((self.set_exglu_weights))
                    print("Reward times"); print(self.reward_times)
                    print("Xor input times"); print(self.xor_input_times)
                if self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                    'pattern_homo_bcm', 'pattern_hetero_bcm',
                                    'pattern_hetero', 'pattern_hetero_amp',
                                    'pattern_cont_hetero_amp']:
                    fih2 = h.FInitializeHandler((self.seti_stimulus_indicators, self.pattern_input_times))
                    fih_wu = h.FInitializeHandler((self.seti_weight_update_indicators, self.pattern_input_times))
                    print("Reward times"); print(self.reward_times)
                    print("Xor input times"); print(self.pattern_input_times)

            if self.exptype in ['inhibitory_plasticity']:
                fih6 = h.FInitializeHandler((self.set_inh_plasticity, (0, False)))
                fih7 = h.FInitializeHandler((self.set_inh_plasticity, (p.start_inh_plasticity, True)))
                if p.inh_exptype == 'rates':
                    fih8 = h.FInitializeHandler((self.seti_input_rates, p.start_inh_plasticity+p.start_inh_plasticity_offset))
                elif p.inh_exptype == 'weights':
                    fih9 = h.FInitializeHandler((self.seti_weights, (p.start_inh_plasticity+p.start_inh_plasticity_offset, p.new_weight)))
                else:
                    print("From method simulate %s" % type(self))
                    print("Inh_exptype '%s' not supported" % p.inh_exptype)
                    sys.exit(-1)
            h.run()

        end = time.time()
        print("It took %.2f hours or %.4f seconds to simulate." % ((end-start)/3600 , (end-start)))

        if p.simtime > 1000:
                self.tout = np.multiply(np.asarray(self.t.to_python()), 0.001)
        else:
            self.tout = self.t.to_python()

    def get_recording_points(self, dend, step):
        points = []
        if dend.L < step:
            points.append(1.0)
        else:
            q = int(dend.L/step)
            for q in range(1,q+1):
                points.append(q*step/dend.L)
            points.append(1.0)
        return points

    def create_dopamine(self):
        h('dopamine = 0')
        self.dopamine = h.dopamine
        h('stimulus_flag = 0')
        self.stimulus_flag = h.stimulus_flag
        h('weight_update_flag = 0')
        self.weight_update_flag = h.weight_update_flag

    def seti_stimulus_indicators(self, xor_input_times):
        times = []
        for x in xor_input_times:
            times.extend(x)
        times.sort()
        times = (np.unique(times)).tolist()
        for t in times:
            h.cvode.event(t, (self.seti_stimulus_flag, 1))

    def seti_weight_update_indicators(self, xor_input_times):
        times = []
        for x in xor_input_times:
            times.extend(x)
        times.sort()
        times = (np.unique(times)).tolist()
        for t in times:
            h.cvode.event(t + p.time_to_weight_update, (self.seti_weight_update_flag, 1))

    def seti_stimulus_flag(self, flag):
        h.stimulus_flag = flag
        if flag:
            h.cvode.event(h.t + p.training_input_length + p.time_to_reward + p.reward_length, (self.seti_stimulus_flag, 0))
        else:
            if self.exptype in ['xor_shom_my_spillover', 'xor_shom_my_spillover_stp', 'nfbp_inh']:
                self.set_exglu_weights()
                self.reset_exglu()
            if self.exptype in ['nfbp_inh']:
                self.reset_caint()

    def seti_weight_update_flag(self, flag):
        h.weight_update_flag = flag
        if flag:
            h.cvode.event(h.t + p.weight_update_length, (self.seti_weight_update_flag, 0))
        else:
            if self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                'pattern_homo_bcm', 'pattern_hetero_bcm',
                                'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp', 'nfbp_inh']:
                self.reset_caint()

    def seti_reset_calcium(self, xor_input_times):
        times = []
        for t in xor_input_times:
            times.extend(t)
        times.sort()
        times = list(set(times))[1:]
        for t in times:
            h.cvode.event(t-1, (self.reset_calcium))

    def reset_calcium(self):
        heads = [h.head for h in self.cell.spines]
        necks = [h.neck for h in self.cell.spines]
        allsecslist = []
        allsecslist.extend(self.cell.dendlist)
        allsecslist.extend(heads)
        allsecslist.extend(necks)
        allsecslist.extend(self.cell.somalist)
        for s in allsecslist:
            s.cai = 50e-6

    def reset_exglu(self):
        for s in self.exglu:
            s.m = 0

    def reset_caint(self):
        if self.exptype in ['pattern']:
            syntype = 'adaptive2_inhexp2syn'
        elif self.exptype in ['pattern_homo']:
            syntype = 'adaptive2_homo_inhexp2syn'
        elif self.exptype in ['pattern_homo_sz']:
            syntype = 'adaptive2_homo_sz_inhexp2syn'
        elif self.exptype in ['pattern_homo_bcm']:
            syntype = 'adaptive2_homo_bcm_inhexp2syn'
        elif self.exptype in ['pattern_hetero_bcm']:
            syntype = 'adaptive2_hetero_bcm_inhexp2syn'
        elif self.exptype in ['pattern_hetero']:
            syntype = 'adaptive2_hetero_inhexp2syn'
        elif self.exptype in ['pattern_hetero_amp', 'nfbp_inh']:
            syntype = 'adaptive2_hetero_amp_inhexp2syn'
        elif self.exptype in ['pattern_cont_hetero_amp']:
            syntype = 'adaptive2_cont_hetero_amp_inhexp2syn'
        else:
            print("From method: reset_caint")
            print("Experiment type '%s' does not use variable F (caint)" % self.exptype )
            sys.exit(-1)

        synlist = self.get_synapse_list(syntype, clustered_flag = True, drive_type = 'i')
        for s in synlist:
            s.obj.caint = 0
        h.fcurrent()

    def set_exglu_weights(self):
        if self.exptype == 'xor_shom_my_spillover':
            synlist = self.get_synapse_list('adaptive_my_shom_NMDA', clustered_flag = True)
            for s, exglu in zip(synlist, self.exnc):
                exglu.weight[0] = s.obj.weight*p.exglu_norm_factor

        elif self.exptype == 'xor_shom_my_spillover_stp':
            synlist = self.get_synapse_list('adaptive_shom_NMDA_stp', clustered_flag = True)
            for s, exglu in zip(synlist, self.exnc):
                exglu.weight[0] = s.obj.weight*p.exglu_norm_factor

        elif self.exptype == 'record_ca':
            for exglu in self.exnc:
                exglu.weight[0] = p.weight*p.exglu_norm_factor

        elif self.exptype == 'nfbp_inh':
            synlist = self.get_synapse_list('adaptive_hom_NMDA', clustered_flag = True)
            for s, exglu in zip(synlist, self.exnc):
                exglu.weight[0] = s.obj.weight*p.exglu_norm_factor
        else:
            print("From method set_exglu_weights in %s" % type(self))
            print("Exptype '%s' not supported" % self.exptype)
            sys.exit(-1)

    def set_inh_plasticity(self, t, flag):
            h.cvode.event(t, (self.set_inh_learning_rate, flag))

    def set_inh_learning_rate(self, flag):
        synlist = self.get_synapse_list('adaptive_inhexp2syn', clustered_flag = False, drive_type = 'i')
        if flag:
            for s in synlist:
                s.obj.learning_rate = p.learning_rate_inh
                #if s.sec(s.pos).cai > p.theta_min_inh:
                if s.obj.caint*p.theta_inh_sf > p.theta_min_inh:
                    s.obj.theta = s.obj.caint*p.theta_inh_sf#s.sec(s.pos).cai
                else:
                    s.obj.theta = p.theta_inh    # s.obj.theta_min = s.obj.theta - (p.theta_inh - p.theta_min_inh)
                    # s.obj.theta_max = s.obj.theta + (p.theta_max_inh - p.theta_inh)
        else:
            for s in synlist:
                s.obj.learning_rate = 0.0

    def seti_input_rates(self, t):
        h.cvode.event(t, self.set_input_rates)

    def set_input_rates(self):
        for s in self.cell.esyn:
            s.stim[0].number *= p.new_erate/p.erate
            s.stim[0].interval /= p.new_erate/p.erate

        for s in self.cell.isyn:
            s.stim[0].number *= p.new_irate/p.irate
            s.stim[0].interval /= p.new_irate/p.irate

    def seti_weights(self, t, w):
        h.cvode.event(t, (self.set_weights, w))

    def set_weights(self, w):
        for s in self.cell.esyn:
            s.nc[0].weight[0] = w

    def seti_print_status(self):
        update_points = np.arange(0, p.simtime, p.simtime/100. )
        for t in update_points:
            h.cvode.event(t, self.print_status)

    def print_status(self):
        print("At time t %f, total simtime %f." % (h.t, p.simtime))

    def seti_dopamine_release(self, reward_times):
        for t in reward_times:
            h.cvode.event(t, (self.seti_dopamine_var, 0))

    def seti_dopamine_var(self, reset_flag):
        if reset_flag:
            h.dopamine = 0
        else:
            te = self.training_set_copy[0]
            spike_times = self.soma_recorder_tvec[-1].to_python()
            if len(spike_times) == 0:
                spike_flag = 0
            else:
                last_spike_time = spike_times[-1]
                if (last_spike_time >= (h.t - p.training_input_length - p.time_to_reward)) and (last_spike_time <= h.t):
                    spike_flag = 1
                else:
                    spike_flag = 0

            if self.training_mode == 'supra':
                if te in ['rb', 'ys']:
                    if spike_flag:
                        value = -1
                    else:
                        value = 0
                elif te in ['rs', 'yb']:
                    if spike_flag:
                        value = 1
                    else:
                        value = 0
            elif self.training_mode == 'sub':
                if te in ['rb', 'ys']:
                    value = -1
                elif te in ['rs', 'yb']:
                    value = 1

            if self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                'pattern_homo_bcm', 'pattern_hetero_bcm',
                                'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp']:
                value = 0
            self.training_set_copy = self.training_set_copy[1:len(self.training_set_copy)]
            h.dopamine = value
            self.rewards_delivered.append(value)
            self.spike_flags.append(spike_flag)
            h.cvode.event(h.t + p.reward_length, (self.seti_dopamine_var, 1))

    def create_xor_inputs(self, input_size):

        self.xor_input_vectors = [[], [], [], []]
        self.xor_input = [[], [], [], []]
        num_groups = 4

        if self.exptype == 'nfbp_inh':
            self.xor_inh_input_vectors = [[], [], [], []]
            self.xor_inh_input = [[], [], [], []]
            num_groups = 4

        self.pf_input_vectors = []
        self.pf_input = []

        if p.correlated_distributed_inputs:
            self.distributed_input_vectors = [ [], [], [], [] ]
            self.distributed_input = [ [], [], [], [] ]
        else:
            self.distributed_input_vectors = []
            self.distributed_input = []

        # Create vectors with spike times for each Xor clustered and distributed input
        group = 0
        self.merged_times = []
        for times_list in self.xor_input_times:
            for x in range(0, p.xor_input_size):
                spike_times = []
                for t in times_list:
                    for n in range(0, p.num_spikes):
                        spike_times.append(rnd.uniform(t, t + p.xor_input_window))
                spike_times.sort()
                self.xor_input_vectors[group].append(h.Vector(spike_times))

                if self.exptype == 'nfbp_inh':
                    spike_times = []
                    for t in times_list:
                        for n in range(0, p.num_inh_spikes):
                            spike_times.append(rnd.uniform(t, t + p.inh_input_window) + p.inh_delay)
                    spike_times.sort()
                    self.xor_inh_input_vectors[group].append(h.Vector(spike_times))

            if p.correlated_distributed_inputs:
                for x in range(0, p.distributed_input_size):
                    spike_times = []
                    for t in times_list:
                        for n in range(0,p.num_spikes):
                            spike_times.append(rnd.uniform(t + p.xor_input_window, t + p.xor_input_window + p.distributed_input_window))
#                        spike_times.append(rnd.uniform(t , t + p.distributed_input_window))
                    spike_times.sort()
                    self.distributed_input_vectors[group].append(h.Vector(spike_times))
            else:
                self.merged_times.extend(times_list)
            group = group + 1

        if p.correlated_distributed_inputs:
            for i in range(0,num_groups):
                for x in range(0, p.distributed_input_size):
                    self.distributed_input[i].append(h.VecStim())
                    self.distributed_input[i][x].play(self.distributed_input_vectors[i][x])

#        for i in range(0,num_groups):
#            for x in range(0, p.xor_input_size):
#                self.xor_input[i].append(h.VecStim())
#                self.xor_input[i][x].play(self.xor_input_vectors[i][x])
#                print(self.xor_input_vectors[i][x].to_python())
        else:
            self.merged_times = list(set(self.merged_times))
            self.merged_times.sort()
            print(self.merged_times)
            for x in range(0, p.distributed_input_size):
                spike_times = []
                for t in self.merged_times:
                    for n in range(0,p.num_spikes):
                        spike_times.append(rnd.uniform(t + p.xor_input_window, t + p.xor_input_window + p.distributed_input_window))
                spike_times.sort()
                self.distributed_input_vectors.append(h.Vector(spike_times))

            for i in range(0, p.distributed_input_size):
                self.distributed_input.append(h.VecStim())
                self.distributed_input[i].play(self.distributed_input_vectors[i])

#        print("Created %d 1L NetStims and %d NetCons to supply them with inputs." % (len(self.burst_input), len(self.burst_input_nc))
#        print("And this many NetCon groups"); print([len(x) for x in self.xor_nc])

    def create_pattern_inputs(self, input_size):

        self.pattern_input_vectors = [[], [], []]
        self.pattern_input = [[], [], []]
        num_groups = 3

        self.pattern_inh_input_vectors = [[], [], []]
        self.pattern_inh_input = [[], [], []]

        if p.correlated_distributed_inputs:
            self.distributed_input_vectors = [ [], [], [] ]
            self.distributed_input = [ [], [], [] ]
        else:
            self.distributed_input_vectors = []
            self.distributed_input = []

        # Create vectors with spike times for each Xor clustered and distributed input
        group = 0
        self.merged_times = []
        for times_list in self.pattern_input_times:
            for x in range(0, p.pattern_input_size):
                spike_times = []
                for t in times_list:
                    for n in range(0, p.num_spikes):
                        spike_times.append(rnd.uniform(t, t + p.pattern_input_window))
                spike_times.sort()
                self.pattern_input_vectors[group].append(h.Vector(spike_times))

                spike_times = []
                for t in times_list:
                    for n in range(0, p.num_inh_spikes):
                        spike_times.append(rnd.uniform(t, t + p.pattern_input_window) + p.inh_delay)
                spike_times.sort()
                self.pattern_inh_input_vectors[group].append(h.Vector(spike_times))

            if p.correlated_distributed_inputs:
                for x in range(0, p.distributed_input_size):
                    spike_times = []
                    for t in times_list:
                        for n in range(0,p.num_spikes):
                            spike_times.append(rnd.uniform(t + p.xor_input_window, t + p.xor_input_window + p.distributed_input_window))
#                        spike_times.append(rnd.uniform(t , t + p.distributed_input_window))
                    spike_times.sort()
                    self.distributed_input_vectors[group].append(h.Vector(spike_times))
            else:
                self.merged_times.extend(times_list)
            group = group + 1

        if p.correlated_distributed_inputs:
            for i in range(0,num_groups):
                for x in range(0, p.distributed_input_size):
                    self.distributed_input[i].append(h.VecStim())
                    self.distributed_input[i][x].play(self.distributed_input_vectors[i][x])

#        for i in range(0,num_groups):
#            for x in range(0, p.xor_input_size):
#                self.xor_input[i].append(h.VecStim())
#                self.xor_input[i][x].play(self.xor_input_vectors[i][x])
#                print(self.xor_input_vectors[i][x].to_python())
        else:
            self.merged_times = list(set(self.merged_times))
            self.merged_times.sort()
            print(self.merged_times)
            for x in range(0, p.distributed_input_size):
                spike_times = []
                for t in self.merged_times:
                    for n in range(0,p.num_spikes):
                        spike_times.append(rnd.uniform(t + p.xor_input_window, t + p.xor_input_window + p.distributed_input_window))
                spike_times.sort()
                self.distributed_input_vectors.append(h.Vector(spike_times))

            for i in range(0, p.distributed_input_size):
                self.distributed_input.append(h.VecStim())
                self.distributed_input[i].play(self.distributed_input_vectors[i])

#        print("Created %d 1L NetStims and %d NetCons to supply them with inputs." % (len(self.burst_input), len(self.burst_input_nc))
#        print("And this many NetCon groups"); print([len(x) for x in self.xor_nc])


    def create_training_set(self, size):
        training_set = []

        if self.exptype in ['xor', 'xor_hom', 'xor_gen', 'xor_sspillover',
                            'xor_spillover', 'xor_hom_spillover','xor_shom_spillover',
                            'xor_cshom_spillover', 'xor_zahra_spillover', 'xor_shom_my_spillover',
                            'xor_shom_my_spillover_stp', 'nfbp_inh']:
            if not p.random_training_sequence:
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('rs')
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('yb')
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('ys')
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('rb')
            else:
                training_set.extend(['rs', 'yb', 'ys', 'rb'])
                # for i in range(0,3):
                #     training_set.extend(rnd.sample(['rs', 'yb', 'yb', 'ys'], 4))
                for i in range(0, p.training_set_size):
                    training_set.append(rnd.choice(['rb', 'rs', 'yb', 'ys']))
                training_set.extend(['rs', 'yb', 'ys', 'rb'])

        elif self.exptype in ['xor_test_set', 'xor_spillover_test']:
#            for i in range(0,p.training_set_size):
            for j in range (0,p.test_set_size_per_group):
                training_set.append('rs')
            for j in range (0,p.test_set_size_per_group):
                training_set.append('yb')
            for i in range (0,p.test_set_size_per_group):
                training_set.append('ys')
            for i in range (0,p.test_set_size_per_group):
                training_set.append('rb')

        elif self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                              'pattern_homo_bcm', 'pattern_hetero_bcm',
                              'pattern_hetero', 'pattern_hetero_amp',
                              'pattern_cont_hetero_amp']:
            if not p.random_training_sequence:
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('a')
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('b')
                for i in range (0,p.training_set_size_per_group):
                    training_set.append('c')
            else:
                training_set.extend(['a', 'b', 'c'])
                for i in range(0, p.training_set_size):
                    training_set.append(rnd.choice(['a', 'b', 'c']))
                training_set.extend(['a', 'b', 'c'])
        return training_set

    def convert_ts_to_times(self, training_set):
        time = p.first_training_input_start
        reward_times = []

        if self.exptype not in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                'pattern_homo_bcm', 'pattern_hetero_bcm',
                                'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp']:
            exp_input = [[], [], [], []]
            for t in training_set:
                if t == 'rb':
                    exp_input[0].append(time)
                    exp_input[3].append(time)
                elif t == 'rs':
                    exp_input[0].append(time)
                    exp_input[2].append(time)
                elif t == 'yb':
                    exp_input[1].append(time)
                    exp_input[3].append(time)
                elif t == 'ys':
                    exp_input[1].append(time)
                    exp_input[2].append(time)

                reward_times.append(time + p.training_input_length + p.time_to_reward)
    #            time = time + p.training_input_length + p.time_to_reward + p.reward_length
                time = time + p.session_length

    #        time = p.first_training_input_start + p.training_input_length + p.time_to_reward
            time = time + p.session_length
        else:
            exp_input = [[], [], []]
            for t in training_set:
                if t == 'a':
                    exp_input[0].append(time)
                elif t == 'b':
                    exp_input[1].append(time)
                elif t == 'c':
                    exp_input[2].append(time)

                reward_times.append(time + p.training_input_length + p.time_to_reward)
                time = time + p.session_length

            time = time + p.session_length

        return exp_input, reward_times

    def connect_xor_inputs(self):
        if (type(self.dendstatobj) == type(dsd.DendStat())):
            for dend, di, ds in zip(self.dendstatobj.dends, self.dendstatobj.dend_inputs, self.dendstatobj.dend_syns):
                start_pos = p.cluster_start_poss[p.independent_dends.index(dend)]
                end_pos = p.cluster_end_poss[p.independent_dends.index(dend)]

                if self.exptype in ['xor_shom_my_spillover', 'nfbp_inh']:
                    self.exglusec.append(h.Section(name = 'exglusec%d' % len(self.exglusec)))
                    self.exglu.append(h.IntFire1(self.exglusec[-1](0.5)))
                    self.exglu[-1].tau = p.exglu_tau
                    self.exglu[-1].refrac = p.session_length - p.training_input_length
                for group, numsyns in zip(di, ds):
                    if group == 'r':
                        group = 1
                    elif group == 'y':
                        group = 2
                    elif group == 's':
                        group = 3
                    elif group == 'b':
                        group = 4

                    sources = rnd.sample([i for i in range(0,p.xor_input_size)], numsyns)
                    for i,s in enumerate(sources):
                        gen = h.VecStim()
                        self.xor_input[group-1].append(gen)
                        gen.play(self.xor_input_vectors[group-1][s])
                        syn_step = 1.0/numsyns
                        dist = h.distance(end_pos, sec = self.cell.dendlist[dend]) - h.distance(start_pos, sec = self.cell.dendlist[dend])
                        offset =  ((end_pos -start_pos)*dist/numsyns)/dist
                        pos = start_pos + offset*(group-1)/3 + \
                              (end_pos - start_pos -offset + offset*(group-1)/3)*i*syn_step

                        # syn_step = 1.0/sum(ds)
                        # pos = start_pos + (end_pos-start_pos)*self.cell.num_spines_on_dends[dend]*syn_step
#                        print(self.cell.num_spines_on_dends[dend])
#                        print(start_pos, end_pos, pos)

                        if self.exptype in ['xor_shom_my_spillover']:
                            syn1 = self.cell.insert_synapse('adaptive_shom_AMPA', self.cell.dendlist[dend], pos, add_spine = 1)
                            self.connect_input_generator(syn1, 'adaptive_shom_AMPA', gen)
                            self.cell.spines[-1].syn_on = 0
                            syn2 = self.cell.insert_synapse('adaptive_my_shom_NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                            self.connect_input_generator(syn2, 'adaptive_my_shom_NMDA', gen)
                            syn3 = self.cell.insert_synapse('adaptive_NMDAe', self.cell.dendlist[dend], pos,
                                                                       add_spine = 0, on_spine = 0)
                            self.connect_input_generator(self.exglu[-1], 't_exglu', gen)
                            self.connect_input_generator(syn3, 's_exglu', self.exglu[-1], delay = p.delay_exnmda)

                            h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                            h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                            h.setpointer(syn2.obj._ref_weight, 'weight', syn3.obj)
                            h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)
                            if p.random_initial_weights:
                                if p.distribution == 'gaussian':
                                    syn2.obj.w0 = rnd.gauss(0.5*(p.start_weight+ p.end_weight),0.5*(p.end_weight- p.start_weight))
                                elif p.distribution == 'uniform':
                                    syn2.obj.w0 = rnd.uniform(p.start_weight, p.end_weight)

                        elif self.exptype in ['nfbp_inh']:
                            syn1 = self.cell.insert_synapse('adaptive_hom_AMPA', self.cell.dendlist[dend], pos, add_spine = 1)
                            self.connect_input_generator(syn1, 'adaptive_hom_AMPA', gen)
                            self.cell.spines[-1].syn_on = 0
                            syn2 = self.cell.insert_synapse('adaptive_hom_NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                            self.connect_input_generator(syn2, 'adaptive_hom_NMDA', gen)
                            syn3 = self.cell.insert_synapse('adaptive_NMDAe', self.cell.dendlist[dend], pos,
                                                                       add_spine = 0, on_spine = 0)
                            self.connect_input_generator(self.exglu[-1], 't_exglu', gen)
                            self.connect_input_generator(syn3, 's_exglu', self.exglu[-1], delay = p.delay_exnmda)

                            h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                            h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                            h.setpointer(syn2.obj._ref_weight, 'weight', syn3.obj)
                            h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)

                            # syn1 = self.cell.insert_synapse('adaptive_hom_AMPA', self.cell.dendlist[dend], pos, add_spine = 1)
                            # self.connect_input_generator(syn1, 'adaptive_hom_AMPA', gen)
                            # self.cell.spines[-1].syn_on = 0
                            # syn2 = self.cell.insert_synapse('adaptive_hom_NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                            # self.connect_input_generator(syn2, 'adaptive_hom_NMDA', gen)
                            # syn3 = self.cell.insert_synapse('adaptive_NMDAe', self.cell.dendlist[dend], pos,
                            #                                            add_spine = 0, on_spine = 0)
                            # self.connect_input_generator(syn3, 'adaptive_NMDAe', gen, delay = p.delay_exnmda)
                            # h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                            # h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                            # h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)
                            # h.setpointer(syn2.obj._ref_weight, 'weight', syn3.obj)
                            if p.random_initial_weights:
                                if p.distribution == 'gaussian':
                                    syn2.obj.w0 = rnd.gauss(0.5*(p.start_weight+ p.end_weight),0.5*(p.end_weight- p.start_weight))
                                elif p.distribution == 'uniform':
                                    syn2.obj.w0 = rnd.uniform(p.start_weight, p.end_weight)

                            # syn1 = self.cell.insert_synapse('AMPA', self.cell.dendlist[dend], pos, add_spine = 1)
                            # self.connect_input_generator(syn1, 'AMPA', gen)
                            # self.cell.spines[-1].syn_on = 0
                            # syn2 = self.cell.insert_synapse('NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                            # self.connect_input_generator(syn2, 'NMDA', gen)
                            # # syn3 = self.cell.insert_synapse('NMDAe', self.cell.dendlist[dend], pos,
                            # #                                            add_spine = 0, on_spine = 0)
                            # # self.connect_input_generator(self.exglu[-1], 't_exglu', gen)
                            # # self.connect_input_generator(syn3, 's_exglu', self.exglu[-1], delay = p.delay_exnmda)
                            #
                            # if p.random_initial_weights:
                            #     if p.distribution == 'gaussian':
                            #         w = rnd.gauss(0.5*(p.start_weight+ p.end_weight),0.5*(p.end_weight- p.start_weight))
                            #     elif p.distribution == 'uniform':
                            #         w = rnd.uniform(p.start_weight, p.end_weight)
                            #
                            #     # syn3.obj.weight = w
                            #     syn2.nc[-1].weight[0] = w
                            #     syn1.nc[-1].weight[0] = w

                        if not (self.exptype in ['xor_spillover', 'xor_hom_spillover',
                                                 'xor_spillover_test','xor_shom_spillover',
                                                 'xor_cshom_spillover', 'xor_sspillover','xor_zahra_spillover',
                                                 'xor_shom_my_spillover', 'xor_shom_my_spillover_stp', 'nfbp_inh']):

                            syn.source = group
                            syn.clustered_flag = True
                        else:
                            syn1.source = group
                            syn1.clustered_flag = True

                            syn2.source = group
                            syn2.clustered_flag = True

                            # syn3.source = group
                            # syn3.clustered_flag = True

            for dend, di, ds in zip(self.dendstatobj.dends,
                                    self.dendstatobj.dend_inh_inputs,
                                    self.dendstatobj.dend_inh_syns):
                start_pos = p.cluster_start_poss[p.independent_dends.index(dend)] #+ 0.1
                end_pos = p.cluster_end_poss[p.independent_dends.index(dend)] #+ 0.1
                print(dend, di, ds)
                for group, numsyns in zip(di, ds):
                    if group == 'r':
                        group = 1
                    elif group == 'y':
                        group = 2
                    elif group == 's':
                        group = 3
                    elif group == 'b':
                        group = 4

                    if self.exptype in ['nfbp_inh']:
                        inh_sources = rnd.sample([i for i in range(0,p.xor_inh_size)], numsyns)

                        for i,s in enumerate(inh_sources):
                            gen = h.VecStim()
                            self.xor_inh_input[group-1].append(gen)
                            gen.play(self.xor_inh_input_vectors[group-1][s])
                            syn_step = 1.0/numsyns
                            dist = h.distance(end_pos, sec = self.cell.dendlist[dend]) - h.distance(start_pos, sec = self.cell.dendlist[dend])
                            offset =  ((end_pos -start_pos)*dist/numsyns)/dist
                            pos = start_pos + offset*(group-1)/3 + \
                                  (end_pos - start_pos -offset + offset*(group-1)/3)*i*syn_step

                            syn = self.cell.insert_synapse('adaptive2_hetero_amp_inhexp2syn', self.cell.dendlist[dend], pos, add_spine = 0)
                            self.connect_input_generator(syn, 'adaptive2_hetero_amp_inhexp2syn', gen)

                            h.setpointer(h._ref_weight_update_flag, 'weight_update_flag', syn.obj)
                            if p.random_inh_initial_weights:
                                if p.distribution == 'gaussian':
                                    syn.obj.w0 = rnd.gauss(0.5*(p.start_weight+ p.end_weight),0.5*(p.end_weight- p.start_weight))
                                elif p.distribution == 'uniform':
                                    syn.obj.w0 = rnd.uniform(p.start_weight, p.end_weight)

                            syn.source = group
                            syn.clustered_flag = True

                self.distributed_input_list = []
                self.distributed_input_synlist = []
                if self.exptype in ['xor', 'xor_hom', 'xor_gen',
                                    'xor_spillover','xor_hom_spillover','xor_shom_spillover',
                                    'xor_cshom_spillover', 'xor_sspillover', 'xor_spillover_test',
                                    'xor_zahra_spillover','xor_shom_my_spillover', 'xor_shom_my_spillover_stp']:

                    if p.adaptive_distributed_inputs:
                        if self.exptype in  ['xor', 'xor_spillover']:
                            syntype = 'adaptive_glutamate2'
                        elif self.exptype in ['xor_hom', 'xor_hom_spillover']:
                            syntype = 'adaptive_glutamate_hom'
                        elif self.exptype in ['xor_spillover_test']:
                            syntype = 'glutamate_xor_test'
                    elif self.exptype == 'xor_gen':
                        syntype = 'generalized_rule_dist'
                    else:
                        syntype = 'glutamate_ica_nmda'
                    if p.correlated_distributed_inputs:
                        num_groups = 4
                    else:
                        num_groups = 1
                    for group_counter in range(0, num_groups):
                        distributed_input_dends = []
#                    distributed_input_dends = []
#                        dendlist = [4,5,8,12,15,21,22,24,26,28,35,36,37,41,46,47,51,52,53,57,3,14,17,18,29,38,40,45,48,56,2,13,20,27,43,44]
#                        positions = [0.1, 0.3, 0.5, 0.7, 0.95]
                        for i in range(0, p.distributed_input_size):
#                            idx_dend = int(i/5)
#                            idx_pos = i%5
#                            distributed_input_dends.append([dendlist[idx_dend], positions[idx_pos]])
                            distributed_input_dends.append([rnd.choice(range(1, len(self.cell.dendlist)-1)), rnd.uniform(0,1)])
                        self.distributed_input_list.append(distributed_input_dends)

                        for i,d in enumerate(distributed_input_dends):
                            dend = d[0]; pos = d[1]
                            if self.exptype in  ['xor_sspillover']:
                                syn1 = self.cell.insert_synapse('adaptive_sAMPA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 0)
                                if p.correlated_distributed_inputs:
                                    gen = self.distributed_input[group_counter][i]
                                else:
                                    gen = self.distributed_input[i]
                                self.connect_input_generator(syn1, 'adaptive_sAMPA', gen)
                                syn2 = self.cell.insert_synapse('adaptive_sNMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 0)
                                self.connect_input_generator(syn2, 'adaptive_sNMDA', gen)
                                h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                                h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)
                                syn2.obj.learning_rate_w_LTP = p.learning_rate_w_LTP
                                syn2.obj.learning_rate_w_LTD = p.learning_rate_w_LTD
                                syn2.obj.KD_LTD = p.KD_LTD
                                syn1.source = 'distributed'
                                syn2.source = 'distributed'
                            elif self.exptype == 'xor_shom_spillover' or self.exptype == 'xor_shom_my_spillover':
                                if self.exptype == 'xor_shom_spillover':
                                    syntype = 'adaptive_shom_NMDA'
                                elif self.exptype == 'xor_shom_my_spillover':
                                    syntype = 'adaptive_my_shom_NMDA'
                                syn1 = self.cell.insert_synapse('adaptive_shom_AMPA', self.cell.dendlist[dend], pos, add_spine = 1, on_spine = 0)
                                if p.correlated_distributed_inputs:
                                    gen = self.distributed_input[group_counter][i]
                                else:
                                    gen = self.distributed_input[i]
                                self.connect_input_generator(syn1, 'adaptive_shom_AMPA', gen)
                                self.cell.spines[-1].syn_on = 0
                                syn2 = self.cell.insert_synapse(syntype, self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                                self.connect_input_generator(syn2, syntype, gen)
                                h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                                h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)
                                if p.random_initial_weights:
                                    syn2.obj.w0 = rnd.uniform(p.start_weight, p.end_weight)
                                syn2.obj.learning_rate_w_LTP = p.learning_rate_w_LTP
                                syn2.obj.learning_rate_w_LTD = p.learning_rate_w_LTD
                                syn2.obj.KD_LTD = p.KD_LTD
                                syn1.source = 'distributed'
                                syn2.source = 'distributed'
                            elif self.exptype == 'xor_shom_my_spillover_stp':
                                syn1 = self.cell.insert_synapse('adaptive_shom_AMPA_stp', self.cell.dendlist[dend], pos, add_spine = 1, on_spine = 0)
                                if p.correlated_distributed_inputs:
                                    gen = self.distributed_input[group_counter][i]
                                else:
                                    gen = self.distributed_input[i]
                                self.connect_input_generator(syn1, 'adaptive_shom_AMPA_stp', gen)
                                self.cell.spines[-1].syn_on = 0
                                syn2 = self.cell.insert_synapse('adaptive_shom_NMDA_stp', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                                self.connect_input_generator(syn2, 'adaptive_shom_NMDA_stp', gen)
                                h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                                h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)
                                if p.random_initial_weights:
                                    syn2.obj.w0 = rnd.uniform(p.start_weight, p.end_weight)
                                syn2.obj.learning_rate_w_LTP = p.learning_rate_w_LTP
                                syn2.obj.learning_rate_w_LTD = p.learning_rate_w_LTD
                                syn2.obj.KD_LTD = p.KD_LTD
                                syn1.source = 'distributed'
                                syn2.source = 'distributed'
                            elif self.exptype == 'xor_cshom_spillover':
                                syn1 = self.cell.insert_synapse('adaptive_cshom_AMPA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 0)
                                if p.correlated_distributed_inputs:
                                    gen = self.distributed_input[group_counter][i]
                                else:
                                    gen = self.distributed_input[i]
                                self.connect_input_generator(syn1, 'adaptive_cshom_AMPA', gen)
                                syn2 = self.cell.insert_synapse('adaptive_cshom_NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 0)
                                self.connect_input_generator(syn2, 'adaptive_cshom_NMDA', gen)
                                h.setpointer(h._ref_dopamine, 'dopamine', syn1.obj)
                                h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
                                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn1.obj)
                                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                                syn1.obj.learning_rate_w_LTP = p.learning_rate_w_LTP*0.01
                                syn1.obj.learning_rate_w_LTD = p.learning_rate_w_LTD
                                syn2.obj.learning_rate_w_LTP = p.learning_rate_w_LTP*0.01
                                syn2.obj.learning_rate_w_LTD = p.learning_rate_w_LTD
                                syn1.source = 'distributed'
                                syn2.source = 'distributed'
                            elif self.exptype == 'xor_zahra_spillover':
                                syn1 = self.cell.insert_synapse('adaptive_zahra_AMPA', self.cell.dendlist[dend], pos, add_spine = 1, on_spine = 0)
                                if p.correlated_distributed_inputs:
                                    gen = self.distributed_input[group_counter][i]
                                else:
                                    gen = self.distributed_input[i]
                                self.connect_input_generator(syn1, 'adaptive_zahra_AMPA', gen)
                                self.cell.spines[-1].syn_on = 0
                                syn2 = self.cell.insert_synapse('adaptive_zahra_NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                                self.connect_input_generator(syn2, 'adaptive_zahra_NMDA', gen)
                                h.setpointer(h._ref_dopamine, 'dopamine', syn2.obj)
    #                            h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn2.obj)
                                h.setpointer(syn2.obj._ref_weight, 'weight', syn1.obj)

                                syn1.source = 'distributed'
                                syn2.source = 'distributed'
                            else:
                                syn = self.cell.insert_synapse(syntype, self.cell.dendlist[d[0]], d[1], add_spine = 0)
                                if p.correlated_distributed_inputs:
                                    gen = self.distributed_input[group_counter][i]
                                else:
                                    gen = self.distributed_input[i]
                                self.connect_input_generator(syn, syntype, gen)
                                if p.adaptive_distributed_inputs:
                                    h.setpointer(h._ref_dopamine, 'dopamine', syn.obj)
                                    h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn.obj)
                                    if self.exptype in ['xor_hom_spillover']:
                                        syn.obj.thresh_LTP_0 = p.thresh_LTP_di
                                        syn.obj.thresh_LTD_0 = p.thresh_LTD_di
                                        if syn.type in ['adaptive_glutamate_hom', 'adaptive_glutamate_hom2']:
                                            syn.obj.thresh_LTD_min = p.thresh_LTP_min_di
                                            syn.obj.thresh_LTD_min = p.thresh_LTD_min_di
                                            if syntype == 'adaptive_glutamate_hom':
                                                syn.obj.thresh_LTP_max = p.thresh_LTP_max_di
                                                syn.obj.thresh_LTD_max = p.thresh_LTD_max_di
                                        if syntype in ['adaptive_glutamate_shom', 'adaptive_glutamate_cshom']:
                                            syn.obj.hthresh_LTP_0 = p.thresh_LTP_max_di
                                syn.source = 'distributed'
                                self.distributed_input_synlist.append(syn)
                            if self.exptype == 'xor_gen':
                                h.setpointer(h._ref_dopamine, 'dopamine', syn.obj)
                                h.setpointer(h._ref_stimulus_flag, 'stimulus_flag', syn.obj)

        if self.exptype in ['xor_test_set', 'xor_spillover_test']:
            self.set_syn_weights()


    def connect_pattern_inputs(self):
        if (type(self.dendstatobj) == type(dsd.DendStat())):
            for dend, di, ds in zip(self.dendstatobj.dends, self.dendstatobj.dend_inputs, self.dendstatobj.dend_syns):
                start_pos = p.cluster_start_poss[p.independent_dends.index(dend)]
                end_pos = p.cluster_end_poss[p.independent_dends.index(dend)]

                # if self.exptype in ['pattern']:
                #     self.exglusec.append(h.Section(name = 'exglusec%d' % len(self.exglusec)))
                #     self.exglu.append(h.IntFire1(self.exglusec[-1](0.5)))
                #     self.exglu[-1].tau = p.exglu_tau
                #     self.exglu[-1].refrac = p.session_length - p.training_input_length

                for group, numsyns in zip(di, ds):
                    if group == 'a':
                        group = 1
                    elif group == 'b':
                        group = 2
                    elif group == 'c':
                        group = 3

                    sources = rnd.sample([i for i in range(0,p.pattern_input_size)], numsyns)
                    for i,s in enumerate(sources):
                        gen = h.VecStim()
                        self.pattern_input[group-1].append(gen)
                        gen.play(self.pattern_input_vectors[group-1][s])
                        # syn_step = 1.0/numsyns
                        # dist = h.distance(end_pos, sec = self.cell.dendlist[dend]) - h.distance(start_pos, sec = self.cell.dendlist[dend])
                        # offset =  ((end_pos -start_pos)*dist/numsyns)/dist
                        # pos = start_pos + offset*(group-1)/3 + \
                        #       (end_pos - start_pos -offset + offset*(group-1)/3)*i*syn_step

                        syn_step = 1.0/sum(ds)
                        pos = start_pos + (end_pos-start_pos)*self.cell.num_spines_on_dends[dend]*syn_step
#                        print(self.cell.num_spines_on_dends[dend])
#                        print(start_pos, end_pos, pos)

                        if self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                            'pattern_homo_bcm', 'pattern_hetero_bcm',
                                            'pattern_hetero', 'pattern_hetero_amp',
                                            'pattern_cont_hetero_amp']:
                            syn1 = self.cell.insert_synapse('AMPA', self.cell.dendlist[dend], pos, add_spine = 1)
                            self.connect_input_generator(syn1, 'AMPA', gen)
                            self.cell.spines[-1].syn_on = 0
                            syn2 = self.cell.insert_synapse('NMDA', self.cell.dendlist[dend], pos, add_spine = 0, on_spine = 1)
                            self.connect_input_generator(syn2, 'NMDA', gen)
                            # syn3 = self.cell.insert_synapse('NMDAe', self.cell.dendlist[dend], pos,
                            #                                            add_spine = 0, on_spine = 0)
                            # self.connect_input_generator(self.exglu[-1], 't_exglu', gen)
                            # self.connect_input_generator(syn3, 's_exglu', self.exglu[-1], delay = p.delay_exnmda)

                            if p.random_initial_weights:
                                if p.distribution == 'gaussian':
                                    w = rnd.gauss(0.5*(p.start_weight+ p.end_weight),0.5*(p.end_weight- p.start_weight))
                                elif p.distribution == 'uniform':
                                    w = rnd.uniform(p.start_weight, p.end_weight)

                                # syn3.obj.weight = w
                                syn2.nc[-1].weight[0] = w
                                syn1.nc[-1].weight[0] = w

                            syn1.source = group
                            syn1.clustered_flag = True

                            syn2.source = group
                            syn2.clustered_flag = True

                            # syn3.source = group
                            # syn3.clustered_flag = True

            for dend, di, ds in zip(self.dendstatobj.dends,
                                    self.dendstatobj.dend_inh_inputs,
                                    self.dendstatobj.dend_inh_syns):
                start_pos = p.cluster_start_poss[p.independent_dends.index(dend)]
                end_pos = p.cluster_end_poss[p.independent_dends.index(dend)]
                print(dend, di, ds)
                for group, numsyns in zip(di, ds):
                    if group == 'a':
                        group = 1
                    elif group == 'b':
                        group = 2
                    elif group == 'c':
                        group = 3

                    if self.exptype in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                        'pattern_homo_bcm', 'pattern_hetero_bcm',
                                        'pattern_hetero', 'pattern_hetero_amp',
                                        'pattern_cont_hetero_amp']:
                        inh_sources = rnd.sample([i for i in range(0,p.pattern_inh_size)], numsyns)

                        for i,s in enumerate(inh_sources):
                            gen = h.VecStim()
                            self.pattern_inh_input[group-1].append(gen)
                            gen.play(self.pattern_inh_input_vectors[group-1][s])
                            syn_step = 1.0/numsyns
                            dist = h.distance(end_pos, sec = self.cell.dendlist[dend]) - h.distance(start_pos, sec = self.cell.dendlist[dend])
                            offset =  ((end_pos -start_pos)*dist/numsyns)/dist
                            pos = start_pos + offset*(group-1)/3 + \
                                  (end_pos - start_pos -offset + offset*(group-1)/3)*i*syn_step

                            if self.exptype in ['pattern']:
                                syntype = 'adaptive2_inhexp2syn'
                            elif self.exptype in ['pattern_homo']:
                                syntype = 'adaptive2_homo_inhexp2syn'
                            elif self.exptype in ['pattern_homo_sz']:
                                syntype = 'adaptive2_homo_sz_inhexp2syn'
                            elif self.exptype in ['pattern_homo_bcm']:
                                syntype = 'adaptive2_homo_bcm_inhexp2syn'
                            elif self.exptype in ['pattern_hetero_bcm']:
                                syntype = 'adaptive2_hetero_bcm_inhexp2syn'
                            elif self.exptype in ['pattern_hetero']:
                                syntype = 'adaptive2_hetero_inhexp2syn'
                            elif self.exptype in ['pattern_hetero_amp']:
                                syntype = 'adaptive2_hetero_amp_inhexp2syn'
                            elif self.exptype in ['pattern_cont_hetero_amp']:
                                syntype = 'adaptive2_cont_hetero_amp_inhexp2syn'
                            else:
                                print("From method: connect_pattern_inputs")
                                print("Unknown experiment type '%s'" % self.exptype)
                                sys.exit(-1)

                            syn = self.cell.insert_synapse(syntype, self.cell.dendlist[dend], pos, add_spine = 0)
                            self.connect_input_generator(syn, syntype, gen)

                            h.setpointer(h._ref_weight_update_flag, 'weight_update_flag', syn.obj)
                            if p.random_inh_initial_weights:
                                if p.distribution == 'gaussian':
                                    syn.obj.w0 = rnd.gauss(0.5*(p.start_weight+ p.end_weight),0.5*(p.end_weight- p.start_weight))
                                elif p.distribution == 'uniform':
                                    syn.obj.w0 = rnd.uniform(p.start_weight, p.end_weight)

                            syn.source = group
                            syn.clustered_flag = True

    def random_dend_list(self, input_dends, p):
        dend_list = []
        for d in input_dends:
            if rnd.random() < p:
                dend_list.append(d)
        return dend_list

    def error(self, window, mode = 'same'):
        if self.exptype not in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                'pattern_homo_bcm', 'pattern_hetero_bcm',
                                'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp']:
            verror = []
            vfull_error = []
            if len(self.training_set) > len(self.rewards_delivered):
                self.training_set = self.training_set[0:len(self.rewards_delivered)-12]
            for i in range(0,len(self.training_set)):
                error = 0
                full_error = 0
                t = self.training_set[i]
                r = self.rewards_delivered[i]
                if self.training_mode == 'supra':
                    if t in ['rb', 'ys']:
                        if r == -1 or r == 1:
                            error = 1
                            full_error = 1
                    elif t in ['rs', 'yb']:
                        if r == -1:
                            error = 1
                        if r == -1 or r == 0 or r == 1e-06:
                            full_error = 1
                elif self.training_mode == 'sub':
                    if t in ['rb', 'ys']:
                        if self.spike_flags[i] == 1:
                            error = 1
                            full_error = 1
                    elif t in ['rs', 'yb']:
                        if self.spike_flags[i] == 0:
                            error = 0
                            full_error = 1
                else:
                    print("From method error in %s" % type(self))
                    print("Unrecognized training mode '%s' not supported" % self.training_mode)
                    sys.exit(-1)

                verror.append(error)
                vfull_error.append(full_error)
            self.dendstatobj.verror = verror
            self.dendstatobj.vfull_error = vfull_error
            wferr = []
            for i in range(0,int(p.training_set_size / window)):
                wferr.append(sum(self.dendstatobj.vfull_error[i*window: (i+1)*window])/window)
            werr = []
            for i in range(0,int(p.training_set_size / window)):
                werr.append(sum(self.dendstatobj.verror[i*window: (i+1)*window])/window)

            window_error = werr
            window_full_error = wferr
            self.dendstatobj.window_error = window_error
            self.dendstatobj.window_full_error = window_full_error

            return error, verror, window_error, vfull_error, window_full_error

    def save_syn_weights(self, filename):
        synlist = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
        synlist_agh = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = False)
        synlist.sort(key = lambda f: f.sec.name())
        weights = [s.ref_var_nmda.to_python() for s in synlist]
        weights_agh = [s.ref_var_nmda.to_python() for s in synlist_agh]
        res_dict = {'weights':weights,
                    'weights_agh':weights_agh}
        to_save = json.dumps(res_dict)
        with open(filename,'w', encoding = 'utf-8') as f:
            json.dump(to_save, f)

    def save_syn_thresholds(self, filename):
        synlist = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
        synlist_agh = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = False)
        synlist.sort(key = lambda f: f.sec.name())
        lthresh_LTP = [s.ref_var_lthresh_LTP.to_python() for s in synlist]
        lthresh_LTP_agh = [s.ref_var_lthresh_LTP.to_python() for s in synlist_agh]
        res_dict = {'lthresh_LTP':lthresh_LTP,
                    'lthresh_LTP_agh':lthresh_LTP_agh}
        to_save = json.dumps(res_dict)
        with open(filename,'w', encoding = 'utf-8') as f:
            json.dump(to_save, f)

    def save_performance(self, filename):
        res_dict = {'error':self.dendstatobj.window_full_error,
                    }
        to_save = json.dumps(res_dict)
        with open(filename,'w', encoding = 'utf-8') as f:
            json.dump(to_save, f)

    def peak_dend_voltage(self):
        if p.filter_vd:
            from scipy.signal import medfilt
            kernel_size = p.filter_kernel_size
            self.vdlist[0] = medfilt(self.vdlist[0], kernel_size)
            self.vdlist[1] = medfilt(self.vdlist[1], kernel_size)
            self.vdlist[2] = medfilt(self.vdlist[2], kernel_size)
        else:
            self.vdlist[0] = self.vdlist[0].to_python()
            self.vdlist[1] = self.vdlist[1].to_python()
            self.vdlist[2] = self.vdlist[2].to_python()
        rd1 = {'a':[], 'b':[], 'c':[]}
        rd2 = {'a':[], 'b':[], 'c':[]}
        rd3 = {'a':[], 'b':[], 'c':[]}
        amps = [[] for i in range(0, len(self.vdlist))]
        for t in self.merged_times:
            for i, vd in enumerate(self.vdlist):
                maxv = -80
                #vd = vd.to_python()
                for tt in range(t, t+p.session_length):
                    if (vd[tt]> maxv):
                        maxv = vd[tt]
                amps[i].append(maxv)

        for i, ts in enumerate(self.training_set):
                rd1[ts].append(amps[0][i])
                rd2[ts].append(amps[1][i])
                rd3[ts].append(amps[2][i])

        return rd1, rd2, rd3

    def save_syn_calcium(self, filename):
        synlist = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
        synlist_agh = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = False)
        synlist.sort(key = lambda f: f.sec.name())

        cai_nmda = []
        cai_nmda_agh = []
        t_cai_nmda = []
        t_cai_nmda_agh = []
        cali = []
        cali_agh = []
        t_cali = []
        t_cali_agh = []

        for s in synlist:
            inds, peaks = ss.find_peaks(s.ref_var_cai_nmda.to_python(), height = 0.0001)
            t_cai_nmda.append(list(np.asarray(self.tout)[list(inds)]))
            cai_nmda.append(list(peaks['peak_heights']))

            inds, peaks = ss.find_peaks(s.ref_var_cali.to_python(), height = 0.0001)
            t_cali.append(list(np.asarray(self.tout)[list(inds)]))
            cali.append(list(peaks['peak_heights']))

        for s in synlist_agh:
            inds, peaks = ss.find_peaks(s.ref_var_cai_nmda.to_python(), height = 0.0001)
            t_cai_nmda_agh.append(list(np.asarray(self.tout)[list(inds)]))
            cai_nmda_agh.append(list(peaks['peak_heights']))

            inds, peaks = ss.find_peaks(s.ref_var_cali.to_python(), height = 0.0001)
            t_cali_agh.append(list(np.asarray(self.tout)[list(inds)]))
            cali_agh.append(list(peaks['peak_heights']))

        res_dict = {'t_cai_nmda': t_cai_nmda,
                    't_cali': t_cali,
                    't_cai_nmda_agh': t_cai_nmda_agh,
                    't_cali_agh': t_cali_agh,
                    'cai_nmda': cai_nmda,
                    'cai_nmda_agh': cai_nmda_agh,
                    'cali': cali,
                    'cali_agh': cali_agh}

        to_save = json.dumps(res_dict)
        with open(filename,'w', encoding = 'utf-8') as f:
            json.dump(to_save, f)

    def load_syn_weights(self, filename):
        with open(filename,'r', encoding = 'utf-8') as f:
            to_read = json.load(f)
            res_dict = json.loads(to_read)
        weights = res_dict['weights']
        weights_nmda = res_dict['weights_nmda']

        return weights, weights_nmda

    def set_syn_weights(self):
        if self.dendstatobj == []:
            w_ampa , w_nmda = self.load_syn_weights(p.load_weights_file)
            print(w_ampa, w_nmda)
            synlist = []
            for syn in self.cell.esyn:
                if syn.type in ['glutamate_test',
                                'adaptive_glutamate',
                                'glutamate_xor_test',
                                'adaptive_glutamate2',
                                'adaptive_glutamate_hom',
                                'adaptive_glutamate_hom2']:
                    synlist.append(syn)
            if synlist != []:
                synlist.sort(key = lambda f: f.sec.name())
                for i in range(0, len(w_ampa)):
                    synlist[i].obj.w_ampa = w_ampa[i]
                    print("AMPA = %f" % synlist[i].obj.w_ampa)
                    if synlist[i].type in ['adaptive_glutamate2', 'adaptive_glutamate_hom',
                    'adaptive_glutamate_hom2', 'glutamate_xor_test']:
                        synlist[i].obj.w_nmda = w_nmda[i]
                        print("NMDA = %f" % synlist[i].obj.w_nmda)
            else:
                print("No adaptive synapses to set weights.")

        elif type(self.dendstatobj) == type(dsd.DendStat()):
            if self.exptype in ['xor_test_set']:
                synlist = []
                for syn in self.cell.esyn:
                    if syn.type == 'glutamate_xor_test' and syn.clustered_flag == True:
                        synlist.append(syn)
    #            synlist.sort(key = lambda f: (f.sec.name(), f.source))
                for dend, num_syns, syns_on_dend_a, syns_on_dend_n in zip(self.dendstatobj.dends,
                                                                          self.dendstatobj.dend_syns,
                                                                          self.dendstatobj.syns_ampa,
                                                                          self.dendstatobj.syns_nmda):
                    temp_str = '[%d]' % (dend)
                    temp_synlist = [s for s in synlist if temp_str in s.sec.name()]
                    counter = 0
                    for syn_group_a, syn_group_n, n in zip(syns_on_dend_a, syns_on_dend_n, num_syns):
                        print(len(syn_group_a), len(syn_group_n))
                        for i in range(0, n):
                            print("*", i, counter)
                            temp_synlist[counter].obj.w_ampa = syn_group_a[i]
                            temp_synlist[counter].obj.w_nmda = syn_group_n[i]
                            counter += 1

            elif self.exptype in ['xor_spillover_test']:
                synlist_a = []
                synlist_n = []
                synlist_ne = []
                for syn in self.cell.esyn:
                    if syn.type == 'AMPA_test' and syn.clustered_flag == True:
                        synlist_a.append(syn)
                    if syn.type == 'NMDA_test' and syn.clustered_flag == True and ('spine' in syn.sec.name()):
                        synlist_n.append(syn)
                    if syn.type == 'NMDA_test' and syn.clustered_flag == True and not ('spine' in syn.sec.name()):
                        synlist_ne.append(syn)
                print(synlist_a)
                print(synlist_n)
                print(synlist_ne)
                for dend, num_syns, syns_on_dend_a, syns_on_dend_n, syns_on_dend_ne in zip(self.dendstatobj.dends,
                                                                          self.dendstatobj.dend_syns,
                                                                          self.dendstatobj.syns_ampa,
                                                                          self.dendstatobj.syns_nmda,
                                                                          self.dendstatobj.syns_nmda_e):

                    temp_str = '[%d]' % (dend)
                    temp_synlist_a = [s for s in synlist_a if temp_str in s.sec.name()]
                    temp_synlist_n = [s for s in synlist_n if temp_str in s.sec.name()]
                    temp_synlist_ne = [s for s in synlist_ne if temp_str in s.sec.name()]
                    counter = 0
                    for syn_group_a, syn_group_n, syn_group_ne, n in zip(syns_on_dend_a, syns_on_dend_n, syns_on_dend_ne, num_syns):
                        print(len(syn_group_a), len(syn_group_n))
                        for i in range(0, n):
                            temp_synlist_a[counter].obj.weight = syn_group_a[i]
                            temp_synlist_n[counter].obj.weight = syn_group_n[i]
                            temp_synlist_n[counter].obj.Cdur = p.Cdur_init + int(p.Cdur_factor*syn_group_n[i])
                            temp_synlist_ne[counter].obj.weight = syn_group_ne[i]
                            temp_synlist_ne[counter].obj.Cdur = p.Cdur_init + int(p.Cdur_factor*syn_group_ne[i])
                            counter += 1

    def get_synapse_list(self, syntype, clustered_flag = False, drive_type = 'e'):
        synlist = []
        if drive_type == 'e':
            syns =  self.cell.esyn
        elif drive_type == 'i':
            syns =  self.cell.isyn
        for syn in syns:
            if syn.type == syntype and syn.clustered_flag == clustered_flag:
                synlist.append(syn)

        return synlist

    def set_color(self, group_id):
        if self.exptype not in ['pattern', 'pattern_homo', 'pattern_homo_sz',
                                'pattern_homo_bcm', 'pattern_hetero_bcm',
                                'pattern_hetero', 'pattern_hetero_amp',
                                'pattern_cont_hetero_amp']:
            if group_id == 1:
                color = '#be0119'#DARD RED
                linestyle = "-"
                marker = None
            elif group_id == 2:
                color = '#fec615' #YELLOW
                linestyle = "-"
                marker = None
            elif group_id == 3:
                color = '#fc5a50' #RED
                linestyle = "-"
                marker = '$Y$'
            elif group_id == 4:
                color = '#b79400' #DARK YELLOW
                linestyle = "-"
                marker = '$/$'
        else:
            if group_id == 1:
                color = '#9467bd' #Purple
                linestyle = "-"
                marker = None
            elif group_id == 2:
                color = '#ff7f0e' #TEAL
                linestyle = "-"
                marker = None
            elif group_id == 3:
                color = '#008080'  #ORANGE
                linestyle = "-"
                marker = None
        return color, linestyle, marker

    def gmax_derivs(self):
        self.derivs = []
        for g in self.gmax_adaptive:
            self.derivs.append(h.Vector())
            self.derivs[-1].deriv(g)
        return self.derivs

    def save_xor_input_list(self):
        res_dict = {'xor_input_list': self.input_list}
        to_save = json.dumps(res_dict)
        with open(p.input_config_file, 'w', encoding = 'utf-8') as f:
             json.dump(to_save, f)

    def activator(self, KD, Ca, n):
        return np.divide( np.power(Ca,n), np.power(Ca,n) + np.power(KD,n))

    def repressor(self, KD, Ca, n):
        return np.divide( np.power(KD,n), np.power(Ca ,n) + np.power(KD,n))

    def plot_thresholds(self):
        figs_LTP = []; figs_LTD = [];
        axes_LTP = []; axes_LTD = [];

#        synlist_nc = self.get_synapse_list('adaptive_shom_NMDA', clustered_flag = True)
        synlist_nc = self.get_synapse_list('adaptive_cshom_NMDA', clustered_flag = True)
        self.heatmaps_LTP = []; self.heatmaps_LTD = []
        syns = [synlist_nc[p.syn_for_threshold]]
        Ca_LTP = np.linspace(0, 1, p.simtime/p.record_step_thresh)
        Ca_LTD = np.linspace(0, 0.01, p.simtime/p.record_step_thresh)

        for i,s in enumerate(synlist_nc[1:]):
            if synlist_nc[i].sec != synlist_nc[i-1].sec:
                syns.append(s)

        for s in syns:
            lthresh_LTP = s.ref_var_lthresh_LTP.to_python()
            hthresh_LTP = s.ref_var_hthresh_LTP.to_python()
            lthresh_LTD = s.ref_var_lthresh_LTD.to_python()
            curves = []
            lthreshs = []
            yticklabels = (np.arange(0, (p.simtime/1000), 10)).tolist()
            yticks = np.linspace(0, len(lthresh_LTP)-1, 10)
            xticklabels_LTP = [0, 0.5, 1.0]
            xticklabels_LTD = [0, 0.005, 0.01]
            xticks = [0, len(Ca_LTP)/2, len(Ca_LTP)]

            for l, hh, ll in zip(lthresh_LTP, hthresh_LTP, lthresh_LTD):
                act = self.activator(l, Ca_LTP, p.Hill_coefficient)
                rep = self.repressor(hh, Ca_LTP, p.Hill_coefficient)
                curve = (np.multiply(act, rep)).tolist()
                act_l = self.activator(ll, Ca_LTD, p.Hill_coefficient)
                curves.append(curve)
                lthreshs.append(act_l)
            self.heatmaps_LTP.append(curves)
            self.heatmaps_LTD.append(lthreshs)

            hm_LTP = np.asarray(curves)
            hm_LTD = np.asarray(lthreshs)

            figs_LTP.append(plt.figure());
            axes_LTP.append(figs_LTP[-1].add_subplot(111));
            axes_LTP[-1].imshow(hm_LTP, cmap = "Purples",
                                origin = "lower")

            axes_LTP[-1].set_ylabel('t'); axes_LTP[-1].set_xlabel('LTP threshold');
            axes_LTP[-1].set_xticks(xticks); axes_LTP[-1].set_xticklabels(xticklabels_LTP);
            axes_LTP[-1].set_yticks(yticks); axes_LTP[-1].set_yticklabels(yticklabels);

            figs_LTD.append(plt.figure())
            axes_LTD.append(figs_LTD[-1].add_subplot(111));
            axes_LTD[-1].imshow(hm_LTD)
            axes_LTD[-1].set_ylabel('t'); axes_LTD[-1].set_xlabel('LTD threshold');
            axes_LTD[-1].set_xticks(xticks); axes_LTD[-1].set_xticklabels(xticklabels_LTD);
            axes_LTD[-1].set_yticks(yticks); axes_LTD[-1].set_yticklabels(yticklabels);
            break
        return axes_LTP, axes_LTD

    def plot_cell(self):
        shape = h.Shape()
        shape.size(-130, 100, -130, 250)
        shape.color_all(3)
        shape.plot(plt)
        synlist = self.get_synapse_list('adaptive_cshom_AMPA', clustered_flag = True)
        synlist_agh = self.get_synapse_list('adaptive_cshom_NMDA', clustered_flag = False)

        for s in synlist:  # pattern Y_1 synapses
            shape.point_mark(s.obj, 4, 'O', 1+s.obj.weight*1e3*20)
        for s in synlist_agh:  # pattern Y_2 synapses
            shape.point_mark(s.obj, 5, 'O', 1+s.obj.weight*1e3*20)

    def delete_everything(self):
        self.dend_record_list = None
        self.vdlist = None
        self.t = None
        self.tout = None
        self.short_tout = None
        self.glu =  None
        self.cali = None
        self.cali_dend = None
        self.cai_nmda = None
        self.cai = None
        self.vspine = None
        self.vs = None
        self.record_spinelist = None
        self.vspine = None
        self.ical = None
        self.gsyn = None
        self.inmda = None
        self.A = None
        self.B = None
        self.w_ampa = None
        self.w_nmda = None
        self.lthresh_LTP = None
        self.hthresh_LTP = None
        self.lthresh_LTD = None

        self.soma_recorder_nc = None
        self.soma_recorder_tvec = None
        # self.soma_recorder_nc = None
        # self.soma_recorder_tvec = None
        self.recorder_nc = None
        self.recorder_tvec = None
        self.ramp_enc = None
        self.ramp_estim = None

        self.distributed_input = None
        self.distributed_input_vectors = None
        self.distributed_input_nc = None
        self.distributed_input_list = None

        self.xor_input = None
        self.xor_input_times = None
        self.xor_input_nc = None
        self.xor_input_vectors = None

        self.pf_input = None
        self.pf_input_vectors = None

        self.reward_times = None
        self.rewards_delivered = None
        self.cai_nmda_in_syns = None
        self.dopamine_vec = None

        self.enc = []
        self.estim = []
        self.cell.esyn = []
        self.inc = []
        self.istim = []
        self.cell.isyn = []

        self.g_nmda_pf = None
        self.cai_nmda_in_syns = None
        self.cali_in_syns = None
        self.presyn = h.Section()
        self.synlist = None

        self.training_set = None
        self.training_set_copy = None
        self.training_mode = None
