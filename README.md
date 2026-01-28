# Opposite and complementary roles of the two calcium thresholds for inducing LTP and LTD in models of striatal projection neurons

This is the accompanying code for the article "Opposite and complementary roles of the two calcium thresholds for inducing LTP and LTD in models of striatal projection neurons" [https://doi.org/10.64898/2026.01.20.700499].
The article studies the role of metaplasticity in two separate calcium thresholds for eliciting LTP (long-term potentiation) and LTD (long-term depression), 
using a simple, local, biologically-based rule for synaptic plasticity in corticostriatal synapses onto direct-pathway striatal projection neurons (dSPNs). Two tasks are used for learning, the feature binding problem (FBP) and the nonlinear feature binding problem (NFBP).

A morphologically realistic, biophysically-detailed model of a dSPN is used, taken from the model library in Lindroos R, Hellgren Kotaleski J. Predicting complex spikes 
in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. Eur J Neurosci. 2021; 53: 2117–2134. https://doi.org/10.1111/ejn.14891

## How to run

Running the simulations is very simple. Almost all figures in the article deal either with the FBP or the NFBP. To simulate learning on one of these tasks, simply run one of the files:
```
fbp.py
nfbp.py
```
To get the results for a specific figure, you need to set some parameters in the file `parameters.py`. For each figure, a list of parameters and their values are given below. Before running the simulation for that figure, find the listed parameters in the "default" file `parameters.py` and set their values. 

Also, many figures show the average performance on the two tasks, the FBP and NFBP. These results were obtained by running many trials on a computing cluster. The code for this is given below.

## Simulations run locally on a desktop/laptop

### 1) Figure 3C<sub>1</sub> and Figure 4A<sub>1</sub> - D<sub>1</sub>
Run `fbp.py`. In the file `parameters.py`, set:
```
connectivity = 'random'
```
### 2) Figure 3C<sub>2</sub> and Figure 4A<sub>2</sub> - D<sub>2</sub>
Run `fbp.py`. Use the provided `parameters.py` file (no parameters need to be set).

### 3) Figure 3C<sub>3</sub> and Figure 4A<sub>3</sub> - D<sub>3</sub>
Run `nfbp.py`. Use the provided `parameters.py` file (no parameters need to be set).

### 4) Figure 4 - figure supplement 1.
Repeat the same runs under 1)-3), but in addition set the following parameters in the file `parameters.py`:
```
max_ca_step = 5
max_ca_step_LTD = 1
max_ca_step_LTP = 5
```
### 5) Figure 4 - figure supplement 2.
Repeat the same runs under 2)-3), but in addition set the following parameter in the file `parameters.py`:
```
hthresh_LTP = 0.2
```
(This value should be set high, to a value that is not obtainable by the [Ca]<sub>NMDA</sub> in the simulations.)

### 6) Figure 5
Run `nfbp.py`. In the file `parameters.py`, set:
```
learning_rate_thresh_LTD = 0.0
```
### 7) Figure 5 - figure supplement 1
Run `fbp.py`. In the file `parameters.py`, set:
```
learning_rate_thresh_LTD = 0.0
```
For panels A<sub>1</sub> - D<sub>1</sub>, in addition set:
```
connectivity = 'random'
```
(No additional changes are needed for panels A<sub>2</sub> - D<sub>2</sub>).

### 8) Figure 6
Run `nfbp.py`. In the file `parameters.py`, set:
```
learning_rate_thresh_LTP = 0.0
```

### 9) Figure 6 - figure supplement 1
Run `fbp.py`. In the file `parameters.py`, set:
```
learning_rate_thresh_LTP = 0.0
```
For panels A<sub>1</sub> - D<sub>1</sub>, in addition set:
```
connectivity = 'random'
```
(No additional changes are needed for panels A<sub>2</sub> - D<sub>2</sub>).

### 10) Figure 8
Run `nfbp.py`. In the file `parameters.py`, set:
```
learning_rate_thresh_LTP = 0.0
learning_rate_thresh_LTD = 0.0
```

### 10) Figure 8 - figure supplement 1
Run `fbp.py`. In the file `parameters.py`, set:
```
learning_rate_thresh_LTP = 0.0
learning_rate_thresh_LTD = 0.0
```
For panels A<sub>1</sub> - D<sub>1</sub>, in addition set:
```
connectivity = 'random'
```
(No additional changes are needed for panels A_2 - D_2).
