# A local inhibitory plasticity rule for control of neuronal firing rate and supralinear dendritic integration

This is the code for simulating the inhibitory plasticity rule presented in the article "A local inhibitory plasticity rule for control of neuronal firing rate and supralinear dendritic integration" (https://doi.org/10.64898/2026.01.20.700499).
There are two types of simulations in the article:
- Type 1 - control of neuronal firing rate: synaptic inputs are rate-coded and distributed across the dendrites
- Type 2 - control of supralinear dendritic integration: synaptic inputs are sparsely-coded and clustered on a dendrite

The inhibitory plasticity rule is used in a morphlogically realistic, biophysically detailed model of a striatal projection neuron, taken from the model library in  Lindroos R, Hellgren Kotaleski J. Predicting complex spikes in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. Eur J Neurosci. 2021; 53: 2117–2134. https://doi.org/10.1111/ejn.14891 

---

## How to run

To reproduce the figures in the article, run the Python files below. Very important - for each figure, parameters settings need to be set in the file `parameters_inh.py`.  

### Type 1 simulations - control of neuronal firing rate
- **Figure 4A**: run file `type1_single_run.py`.
  Parameter settings in file `parameters_inh.py`:
  ```
  record_step = 1
  record_step_v = 1
  
  simtime = 100000

  erate = 2.3
  irate = 2.3
  new_erate = (1 + 1/3.0)*erate
  new_irate = irate

  theta_min_inh = 0.1
  steepness_inh = 5
  steepness_inh_s1 = 5
  learning_rate_inh = 0.2e-3
  learning_rate_theta_inh = 0
  inh_exptype = 'rates'
  ```
  
- **Figure 4B**: run file `type1_single_run.py`.
  Parameter settings in file `parameters_inh.py`: same as above for Figure 4A, except for the following lines:
  ```
  new_erate = erate
  inh_exptype = 'weights'
  ```
- **Figure 5**: this figure is obtained by running the simulations for Figure 4 in parallel on a computing cluster, and averaging the results. The Python scripts are the following:
  ```
  plas_rates_mpi.py
  plas_weights_mpi.py
  ```
  After obtaining the results, the scripts for analyzing them and plotting Figure 5 are:
  ```
  analyze_plas_rates.py
  analyze_plas_weights.py
  ```
  The bash scripts for running on the computing cluster are:
  ```
  plas_rates.sh
  plas_weights.sh
  ```
### Type 2 simulations - control of supralinear dendritic integration
- **Figure 6**: run file `pattern_homo_bcm.py`.
  Parameter settings in file `parameters_inh.py`:
  ```
  record_step = 800
  record_step_v = 1

  erate = 0.5
  irate = 0.5

  training_set_size_per_group = 600
  training_set_size = training_set_size_per_group*3
  extra_training_set = 2*3

  calcium_amp = 10/theta_inh_sf
  theta_min_inh = 0.5
  theta_inh = 8
  steepness_inh = 10
  steepness_inh_s1 = 10
  learning_rate_inh = 0.05e-3
  learning_rate_theta_inh = 0.5e-3

  simtime = first_training_input_start + (training_set_size + extra_training_set) * session_length

  ymin_winh = -10
  ymax_winh = 300
  tick_winh = 300

  ymin_theta_inh = -0.2
  ymax_theta_inh = 30
  tick_theta_inh = 30

  with_diffusion = False
  ```
- **Figure 7**: run file `pattern_homo.py`.
  Parameter settings in file `parameters_inh.py`: same as above for Figure 6, except for the following lines:
  ```
  training_set_size_per_group = 300

  ymax_theta_inh = 80
  tick_theta_inh = 80
  ```
- **Figure 9**: run file `pattern_hetero.py`.
  Parameter settings in file `parameters_inh.py`: same as above for Figure 6, except for the following lines:
  ```
  ymax_winh = 600
  tick_winh = 600
  
  ymax_theta_inh = 60
  tick_theta_inh = 60
  ```
- **Figure 11**: run file `pattern_hetero_amp.py`.
  Parameter settings in file `parameters_inh.py`: same as above for Figure 6, except for the following lines:
  ```
  ymax_winh = 400
  tick_winh = 400
  
  ymax_theta_inh = 60
  tick_theta_inh = 60
  ```
### NFBP simulations
The nonlinear feature binding problem (NFBP) is used to show a case where inhibitory plasticity works in tandem with excitatoty plasticity. The results are shown in two figures:
- **Figure 12**: run file `nfbp_inh.py`.
Parameter settings in file `parameters_inh.py`:
```
  record_step = 800
  record_step_v = 1

  erate = 1.4
  irate = 1.4

  training_set_size_per_group = 300
  training_set_size = training_set_size_per_group*4
  extra_training_set = 2*4

  calcium_amp = 350/theta_inh_sf
  theta_min_inh = 0.5
  theta_inh = 8
  steepness_inh = 10
  steepness_inh_s1 = 10
  learning_rate_inh = 0.05e-3
  learning_rate_theta_inh = 0.5e-3

  simtime = first_training_input_start + (training_set_size + extra_training_set) * session_length

  ymin_winh = -10
  ymax_winh = 300
  tick_winh = 300

  ymin_theta_inh = -0.2
  ymax_theta_inh = 30
  tick_theta_inh = 30

  with_diffusion = True
  ```
- **Figure 13**: run file `nfbp_inh_sym.py`.
Parameter settings in file `parameters_inh.py`: same as above for Figure 12, except for the following lines:
```
ymax_winh = 500
tick_winh = 500
```
