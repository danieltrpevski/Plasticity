# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:50:36 2019

@author: daniel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:48:41 2016

@author: daniel
"""

from neuron import h
import d1msn as msn
#import iMSN
import plasticity_experiment as pe
import pickle
import parameters as p
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import scipy.signal as ss

# --- 1. Create a cell and other useful stuff
dMSN_library = 'D1_71bestFit_updRheob.pkl'
iMSN_library = 'D2_34bestFit_updRheob.pkl'
with open(dMSN_library, 'rb') as f:
    model_sets  = pickle.load(f, encoding="latin1")

cell_ID = 34
#cell_ID = 1
variables = model_sets[cell_ID]['variables']
#cell = iMSN.iMSN(variables = variables)
cell = msn.MSN(variables = variables)

for d in p.input_dends:
    cell.dendlist[d].nseg *=5

print(cell.max_dist())

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

#independent_dends = [3, 5, 8, 12, 15, 22, 26, 35, 41, 47, 53, 57]
dend_record_list = [3] #[3,4,9,10,21,22,24,26,35,36,51,52]
dend_stim_list = []#[3,4,9,10,35,36]
plateau_cluster_list = [3]

plateau_cluster_size = np.arange(1,21,1)

vs = []
vspine = []
vd = []
cai_nmda = []
cali = []
cati = []
ical =[]
cali_dend = []
cati_dend = []
legend = []
cai_nmda_spine_plot = []
cali_spine_plot = []
cai_spine_plot = []

max_vs = []
max_vspine = []
max_vd = []
max_cai_nmda = []
max_cali = []
max_cati = []
g_nmda = []
i_nmda = []

sns.set(font_scale = 2.0)
sns.set_style('ticks')
fig_vs = plt.figure();
fig_vspine = plt.figure();
fig_vd = plt.figure();
fig_cai_nmda = plt.figure();
fig_cali = plt.figure();
fig_cati = plt.figure();
#fig_ical = plt.figure();
fig_cali_dend = plt.figure();
#fig_cati_dend = plt.figure();
fig_cai_nmda_spine = plt.figure(figsize = (5, 3.5));
fig_cai_cali_spine = plt.figure();
fig_cai_cati_spine = plt.figure();
fig_ica_nmda = plt.figure();
fig_cai_total_spine = plt.figure(figsize = (5, 3.5));

ax_vs = fig_vs.add_subplot(111); ax_vs.set_ylabel('Vs (mV)'); ax_vs.set_xlabel('t (ms)')
ax_vspine = fig_vspine.add_subplot(111); ax_vspine.set_ylabel('Vspine (mV)'); ax_vspine.set_xlabel('t (ms)')
ax_vd = fig_vd.add_subplot(111); ax_vd.set_ylabel('Vd (mV)'); ax_vd.set_xlabel('t (ms)')
ax_cai_nmda = fig_cai_nmda.add_subplot(111); ax_cai_nmda.set_ylabel('Cai_nmda dend(mM)'); ax_cai_nmda.set_xlabel('t (ms)')
ax_cali = fig_cali.add_subplot(111); ax_cali.set_ylabel('Cali spine(mM)'); ax_cali.set_xlabel('t (ms)')
ax_cai = fig_cati.add_subplot(111); ax_cai.set_ylabel('Cai spine(mM)'); ax_cai.set_xlabel('t (ms)')
#ax_ical = fig_ical.add_subplot(111); ax_ical.set_ylabel('Ical'); ax_ical.set_xlabel('t (ms)')
# ax_ica_nmda = fig_ica_nmda.add_subplot(111); ax_ica_nmda.set_ylabel('Ica_nmda'); ax_ica_nmda.set_xlabel('t (ms)')
ax_cali_dend = fig_cali_dend.add_subplot(111); ax_cali_dend.set_ylabel('Cali_dend (mM)'); ax_cali_dend.set_xlabel('t (ms)')
#ax_cati_dend = fig_cati_dend.add_subplot(111); ax_cati_dend.set_ylabel('Cati_dend (mM)'); ax_cati_dend.set_xlabel('t (ms)')
ax_cai_nmda_spine = fig_cai_nmda_spine.add_subplot(111); ax_cai_nmda_spine.set_ylabel('spine [Ca]$_\mathrm{NMDA}$ ($\mu$M)'); ax_cai_nmda_spine.set_xlabel('t (ms)')
ax_cai_cali_spine = fig_cai_cali_spine.add_subplot(111); ax_cai_cali_spine.set_ylabel('Cai_nmda + cali spine(mM)'); ax_cai_cali_spine.set_xlabel('t (ms)')
ax_cai_cati_spine = fig_cai_cati_spine.add_subplot(111); ax_cai_cati_spine.set_ylabel('Cai_nmda + cati spine(mM)'); ax_cai_cati_spine.set_xlabel('t (ms)')
ax_cai_total_spine = fig_cai_total_spine.add_subplot(111); ax_cai_total_spine.set_ylabel('spine [Ca]$_\mathrm{NMDA}$ + [Ca]$_\mathrm{VGCC}$ ($\mu$M)'); ax_cai_total_spine.set_xlabel('t (ms)')
colors = sns.color_palette("icefire", 21)

add_spine = 0
on_spine = 1

start_pos = p.cluster_start_poss[p.independent_dends.index(plateau_cluster_list[0])]
end_pos = p.cluster_end_poss[p.independent_dends.index(plateau_cluster_list[0])]
cell.insert_spines(plateau_cluster_list, start_pos, end_pos, num_spines = p.plateau_cluster_size_max)

#cell.dendlist[57].diam = 0.3
sns.set_style("ticks")
ci = 0
num_syns = 10
weight = 5.0
for num_syns in plateau_cluster_size:
# for weight in [ 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
    p.gmaxNMDA_spillover = weight*1e-3
    p.gmaxNMDAe_spillover = weight*1e-3
#    cell = msn.MSN(variables = variables)
    ex = pe.Plasticity_Experiment('record_ca', cell)
    ex.insert_synapses('MSN')
    ex.insert_synapses('my_spillover', plateau_cluster_list, deterministic = 0,
                       num_syns = num_syns, add_spine = add_spine, on_spine = on_spine)
#    ex.insert_synapses('input_syn', deterministic = 1,
#                       num_syns = p.distributed_input_size*2, add_spine = 0, on_spine = 0)
#    ex.insert_synapses('inhexpsyn_plateau', inhibitory_cluster_dict, deterministic = 1, num_syns = p.inhibitory_cluster_size)

#    p.gAMPAmax_plateau = 0.30e-3
#    p.nmda_ampa_ratio = 10/3.0
#    ex.insert_synapses('pf', plateau_cluster_list, deterministic = 1,
#                   num_syns = 4, add_spine = 0)
    ex.set_up_recording(dend_record_list)
    ex.simulate()
    tv = ex.tv.to_python()
    t = ex.t.to_python()
    legend.append("%d syns" % num_syns)

    if add_spine == 1 or on_spine == 1:
        vspine.append(ex.vspine[0].to_python())
        max_vspine.append(max(ex.vspine[0]))
        ax_vspine.plot(tv, ex.vspine[0].to_python(), color = colors[ci])
        # ax_ica_nmda.plot(tv, ex.ica_nmda[0].to_python(), color = colors[ci])
    ci = ci +1
    vd.append(ex.vdlist[0].to_python())
    max_vd.append(max(ex.vdlist[0]))

    cai_nmda.append(ex.cai_nmda[0].to_python()) ;
    cali.append(ex.cali[0].to_python()) ; max_cali.append(ex.cali[0])
#    cati.append(ex.cati[0].to_python()) ; max_cati.append(ex.cati[0])
#    ical.append(ex.ical[0].to_python()) ;
    # cali_dend.append(ex.cali_dend[0].to_python())
#    cati_dend.append(ex.cati_dend[0].to_python())
    max_vs.append(max(ex.vs))
    vs.append(ex.vs.to_python())
    # Reinitialize synapses
    if not p.with_diffusion:
        cai_nmda_spine = 0
        for c in ex.cai_nmda_spine:
            cai_nmda_spine += np.array(c.to_python())
        cai_nmda_spine = np.divide(cai_nmda_spine,len(ex.cai_nmda_spine))
        cai_nmda_spine_plot.append(cai_nmda_spine.tolist())

    cali_spine = 0
    for c in ex.cali_spine:
        cali_spine += np.array(c.to_python())
    cali_spine = np.divide(cali_spine,len(ex.cali_spine))
    cali_spine_plot.append(cali_spine.tolist())

    cai_spine = 0
    for c in ex.cai_spine:
        cai_spine += np.array(c.to_python())
    cai_spine = np.divide(cai_spine,len(ex.cai_spine))
    cai_spine_plot.append(cai_spine.tolist())

    cell.esyn = []
    ex.estim = []
    ex.enc = []
    for s in cell.spines:
        s.syn_on = 0
    cell.isyn = []
    ex.istim = []
    ex.inc = []
#
vs_indices = []; vs_widths = []
vd_indices = []; vd_widths = []
vspine_indices = []; vspine_wid = []
for v in vs:
    vs_indices.append(ss.find_peaks(v))
    vs_widths.append(ss.peak_widths(v, (vs_indices[-1])[0], rel_height = 0.15))

for v in vd:
    vd_indices.append(ss.find_peaks(v))
    vd_widths.append(ss.peak_widths(v, (vd_indices[-1])[0] ,rel_height = 0.15))

for i in range(0, len(cai_nmda)):
    if add_spine ==0 and on_spine ==0:
        ax_vd.plot(tv, vd[i]);
    ax_cai_nmda.plot(t, cai_nmda[i], color = colors[i])
    ax_cali.plot(t, cali_spine_plot[i], color = colors[i])
    ax_cai.plot(t, cai_spine_plot[i], color = colors[i])
    ax_vs.plot(tv, vs[i], color = colors[i]); ax_vs.set_title("weight = %.2f, Cdur_factor = %d" % (p.weight, p.eCdur_factor))
#    ax_vs.hlines(*(vs_widths[i])[1:])
    ax_vd.plot(tv, vd[i], color = colors[i]); ax_vs.set_title("weight = %.2f, Cdur_factor = %d" % (p.weight, p.eCdur_factor))
#    ax_vd.hlines(*(vd_widths[i])[1:])
    # ax_cali_dend.plot(t, cali_dend[i], color = colors[i])
    if not p.with_diffusion:
        ax_cai_nmda_spine.plot(t, 1000*np.asarray(cai_nmda_spine_plot[i]), color = colors[i])
        ax_cai_total_spine.plot(t, 1000*(np.asarray(cai_nmda_spine_plot[i]) + np.asarray(cai_spine_plot[i])), color = colors[i])
    # ax_cai_cali_spine.plot(t, np.add(np.add(cai_nmda_spine_plot[i] , cali_spine_plot[i]), cai_spine_plot[i]), color = colors[i])
#    ax_cai_cati_spine.plot(t, np.add(cai_nmda_spine_plot[i] , cati_spine_plot[i]), color = colors[i])
ax_vd.set_title("weight = %.2f, Cdur_factor = %d" % (p.weight, p.eCdur_factor))
#    ax_ical.plot(t, ical[i])

sns.despine()
res_dict = {'t': t,
            'vs': vs,
            'vspine': vspine,
            'cai_nmda': cai_nmda,
            'cali_dend': cali_dend,
            'cai_nmda_spine': cai_nmda_spine_plot,
            'cali_spine': cali_spine_plot}
to_save = json.dumps(res_dict)
# filename = './results/data_spillover_steep.dat'
# with open(filename,'w', encoding = 'utf-8') as f:
#    json.dump(to_save, f)

#ax_vs.legend(legend)
#ax_vspine.legend(legend)
#ax_vd.legend(legend)
#ax_cai_nmda.legend(legend)
#ax_cali.legend(legend)
#ax_ical.legend(legend)
#ax_cali_dend.legend(legend)


#fig_all_spines = plt.figure(); plt.hold(True)
#ax_all_spines = fig_all_spines.add_subplot(111);
#ax_all_spines.set_ylabel('v_spines'); ax_all_spines.set_xlabel('t')
#counter = 0
#spines_legend = []
#for v_spine in ex.vspine:
#    ax_all_spines.plot(t, v_spine)
#    spines_legend.append(ex.cell.spines[counter].id)
#    counter = counter +1
#ax_all_spines.legend(spines_legend)

#fig4 = plt.figure(); plt.hold(True)
#ax4 = fig4.add_subplot(111); ax4.set_ylabel('max(Cai_nmda)'); ax4.set_xlabel('max(vd)')
#ax4.plot(max_vd, max_cai_nmda, marker = 'o')

#fig6 = plt.figure(); plt.hold(True)
#ax6 = fig6.add_subplot(111); ax6.set_ylabel('max(Vs)'); ax6.set_xlabel('max(vd)')
#ax6.plot(plateau_cluster_size, max_vs, marker = 'o')

#ex.plot_results()

plt.show()
