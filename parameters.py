# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:39:04 2016

@author: daniel
"""
#from math import sqrt
results_directory = './results'

#-----------------------------------------------------------#
#      1. General recording and simulation parameters       #
#-----------------------------------------------------------#
step = 20.0
record_step = 1
record_step_v = 1
record_step_PDC = 1000
skip_first_x_ms = 100

nrn_dots_per_1ms = 1.0/record_step
time_to_avg_over = 20 # in seconds

simtime = 700
training_mode = 'sub'

num_trials = 20
NUMBER_OF_PROCESSES = 7

#-----------------------------------#
#      2. Synaptic parameters       #
#-----------------------------------#
esyn_tau = 6
isyn_tau = 6
isyn_plateau_tau = 10
e_esyn = 0
e_gaba = -60
erev_NMDA = 0
erate = 1.2
irate = 1.2
pos = 0.05
# NMDA parameters
#Mg = 1.0
#alpha = 0.072
#eta = 0.28
#
Mg = 1.0
alpha = 0.062
eta = 0.381679389

# Mg = 1.0
# alpha = 0.086
# eta = 0.4

#Mg = 1.4
#alpha = 0.099
#eta = 1.0/12

g_ramp_max = 0.000255
nmda_ampa_ratio = 1
gAMPAmax = 1.0e-3
gNMDAmax = gAMPAmax*nmda_ampa_ratio
gGABAmax = 1.5e-3
g_expsyn_max =  0.1e-3
g_inhexpsyn_max = gGABAmax
U = 0.3
u0 = 0.0
tauF = 5.0

gAMPAmax_plateau = 1.5*1.0e-3#0.75*1.0e-3
gNMDAmax_plateau = 3.5*1.0e-3#4.5*1.0e-3
gGABAmax_plateau = 1.5e-3
nmda_ampa_ratio = gNMDAmax_plateau/gAMPAmax_plateau
ratio_glutamate_syn = 1.0
ratio_distributed_synapses = 0.2
nmda_ca_fraction = 0.175

gmaxAMPA_spillover = 1.5e-3 #0.015
gmaxNMDA_spillover = 3.5e-3
gmaxNMDAe_spillover = 3.5e-3 #0.0075
gmaxAMPA_pf = 2.5e-3
gmaxNMDA_pf = 4.5e-3
ampa_alpha = 12.5
ampa_beta = 0.25
nmda_alpha = 4
nmda_beta = 0.01
weight = 0.35
Cdur = 1.1
Cdur_pf = 50
eCdur_init = 50
eCdur_factor = 200
eCdur = eCdur_init + eCdur_factor*weight
width = 0.05
delay_exnmda = 5
random_initial_weights = True
start_weight = 0.4
end_weight = 0.45
distribution = 'uniform'

deterministic_interval = 1
net_con_interval = 0
num_spikes = 3

exglu_weight = 0.37
exglu_tau = simtime
thresh_weight = 0.5
thresh_syns = 20
exglu_norm_factor = 1/(thresh_syns*thresh_weight)*1/num_spikes

tau1_NMDA = 2.76
tau2_NMDA = 115.5

tau1_exp2syn = 1.9
tau2_exp2syn = 4.8
tau1_inhexp2syn = 1
tau2_inhexp2syn = 10
tau_cadyn_nmda = 150
tau_caldyn = 100
tau_catdyn = 100
#-----------------------------------------#
#      3. Synaptic input parameters       #
#-----------------------------------------#

plateau_syn_rate = 400
plateau_burst_start = 200
plateau_burst_end = 230
plateau_cluster_size = 20
plateau_cluster_size_max = 41
cluster_start_pos = 0.45
cluster_end_pos = 0.60
xor_input_window = 35
xor_input_size = 30
syns_per_feature = 5

pf_input_rate = 1
pf_input_start = plateau_burst_start
pf_input_window = 70
pf_input_end = plateau_burst_end+pf_input_window
pf_input_size = 10
pf_num_spikes = 3
pf_input_interval = 1

inhibitory_syn_rate = 105.0
inhibitory_burst_start = 180
inhibitory_burst_end = 230
inhibitory_cluster_size = 5
inh_cluster_start_pos = 0.45
inh_cluster_end_pos = 0.60

distributed_input_rate = 1000.0/40
distributed_input_start = 130
distributed_input_end = 200
distributed_input_size = 0
distributed_input_window = 75
correlated_distributed_inputs = False

ramp_syn_rate = 100.0
ramp_slope = 0.5  # Hz/ms
ramp_burst_start = 200
ramp_burst_end = 1000

e_interval = 1.0/erate*(10**3)
i_interval = 1.0/irate*(10**3)
plateau_syn_interval = 1.0/plateau_syn_rate*(10**3)
distributed_input_interval = 1.0/distributed_input_rate*(10**3)

ramp_syn_interval = 1.0/ramp_syn_rate*(10**3)
inhibitory_syn_interval = 1.0/inhibitory_syn_rate*(10**3)

#--------------------------------#
#      4. Spine parameters       #
#--------------------------------#
head_L = 0.5
head_diam = 0.5
neck_L = 0.5
neck_diam = 0.125
neck_Ra = 1130.0
head_Ra = 150

kb_cadyn_nmda = 96
kt_cadyn_nmda = 1e-4
kd_cadyn_nmda = 1e-4#0.3e-3
include_empty_spines = False
#-----------------------------------------------------------#
#      5. XOR problem and adaptive synapse parameters       #
#-----------------------------------------------------------#

event_times = [200, 500, 900]

x5_training_file = 'data_fig3_red_banana.dat'
save_weights_file = 'one_neuron.dat'
load_weights_file = 'one_neuron.dat'
random_training_sequence = True
adaptive_distributed_inputs = True
plot_distributed_inputs = True
# if distributed_input_size > 0:
#     plot_distributed_inputs = True
long_simulation = False
training_set_size_per_group = 90
training_set_size = training_set_size_per_group*4
training_input_length = 30
first_training_input_start = 200
time_to_reward = 400 - training_input_length
reward_length = 20
session_length = 600
test_set_size_per_group = 1
test_set_size = test_set_size_per_group*4

window_error = 2
record_step_thresh = session_length/2

LTP_factor = 2.0
LTD_factor = 0.1
thresh_LTP = 0.003
thresh_LTD = 0.0001
hthresh_LTP = 0.035
thresh_LTD_min = 0.0001
thresh_LTP_min = 0.001
LTD_thresh_factor = 1.0

learning_rate_w_LTP = 0.025#0.5#0.01
learning_rate_w_LTD = 0.025#3.5
learning_rate_w_LTD_pf = 0#0.05
learning_rate_thresh_LTP = 0.05#0.025
learning_rate_thresh_LTPl = 0.0025*0.5
learning_rate_thresh_LTD = 0.0#0.00035*4
learning_rate_thresh_KD_LTD = 0.05
lthresh_LTP_min = 0.01
threshold_scale_factor = 1.0

# learning_rate_w_LTP = 0#0.035
# learning_rate_w_LTD = 0#0.035
# learning_rate_w_LTD_pf = 0#0.05
# learning_rate_thresh_LTP = 0#0.00035
# learning_rate_thresh_LTD = 0#0.00035
# learning_rate_thresh_KD_LTD = 0# 0.05
# lthresh_LTP_min = 0.01

n1 = 2#200
n2 = 16#1000
KD1 = 0.02#0.025#0.004
KD2 = 0.6#0.0525
KD2_min = 0.02
KD1_min = 0.005
KD_LTD = 0.0002
n_LTD = 750
KD_LTD_pf = 0.0001
n_LTD_pf = 1000

random_weights = False
read_input_config_from_file = False
input_config_file = 'xor_inputs_to_dends.dat'#'xor_dense.dat'
input_dends = [8, 15] #[3, 5, 8, 12, 15, 22, 26, 35, 41, 47, 53, 57]#
id1 = [3, 5, 8, 12, 15, 22, 26, 47, 52]
id2 = [3, 5, 8, 12, 15, 22, 26, 47, 52]
id3 = [3, 5, 8, 12, 15, 22, 26, 4, 35]
sp1 = [0.35, 0.55, 0.4, 0.35, 0.3, 0.3, 0.4, 0.4, 0.25]
ep1 = [0.5, 0.7, 0.5, 0.45, 0.5, 0.45, 0.55, 0.6, 0.45]
sp2 = [0.45, 0.65, 0.55, 0.5, 0.62, 0.5, 0.5, 0.7, 0.55]
ep2 = [0.6, 0.8, 0.65, 0.6, 0.77, 0.65, 0.65, 0.9, 0.99]
sp3 = [0.6, 0.8, 0.65, 0.6, 0.77, 0.65, 0.65, 0.60, 0.05]
ep3 = [0.75, 0.95, 0.75, 0.7, 0.97, 0.85, 0.8, 0.75, 0.35]
independent_dends = id1
cluster_start_poss = sp1
cluster_end_poss = ep1

simtime = first_training_input_start + (training_set_size)*session_length
#simtime = first_training_input_start + test_set_size*session_length
#simtime = first_training_input_start + training_set_size*(training_input_length+
#            time_to_reward + reward_length)

#----------------------------------#
#      6. Diffusion parameters      #
#----------------------------------#
with_diffusion = True
plot_diffusion = False
Dca = 200
Dbuff = 66
kf_ca_nmda_calbindin = 28; kr_ca_nmda_calbindin = 0.7e-6*28
kf_ca_nmda_CaMN = 100; kr_ca_nmda_CaMN = 1e-5
kf_ca_nmda_CaMC = 6; kr_ca_nmda_CaMC = 1.5e-6*6
kf_ca_nmda_fixed = 400; kr_ca_nmda_fixed = 100*400e-3

kcat_pmca_soma = 85;
kcat_pmca_dend = 5.0e3;
kcat_pmca_spine = 0.6
Kd_pmca = 0.3

ca_in0 = 5e-5
ca_out0 = 1.0
calbindin0 = 0.08
camn0 = 0.015
camc0 = 0.015
fixed0 = 0.15
#-----------------------------------#
#      6.1. For XOR experiment      #
#-----------------------------------#

#-------------------------------------------------------#
#      7. Miscellaneous and parameter dictionaries      #
#-------------------------------------------------------#

dends_per_plot = 1
scale_conductance = 1000
