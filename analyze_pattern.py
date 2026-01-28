import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


filename = './results/pattern/theta_init_8/pattern_homo.dat'
with open(filename, 'r', encoding = 'utf-8') as f:
  fload  = json.load(f)
  res_dict = json.loads(fload)

rd1 = res_dict['rd1']
rd2 = res_dict['rd2']
rd3 = res_dict['rd3']
# weights = res_dict['weights']
# theta_inh = res_dict['theta_inh']
# theta_min_inh = res_dict['theta_min_inh']
num_inh_syns = res_dict['num_inh_syns']
trials = res_dict['trials']
# tw = res_dict['tw']
input_dends_list = res_dict['input_dends_list']
weights = res_dict['weights']
theta_inh = res_dict['theta_inh']


# d1a1 = 0; d1a2 = 0; d2a1 = 0; d2a2 = 0; d3a1 = 0; d3a2 = 0;
# d1b1 = 0; d1b2 = 0; d2b1 = 0; d2b2 = 0; d3b1 = 0; d3b2 = 0;
# d1c1 = 0; d1c2 = 0; d2c1 = 0; d2c2 = 0; d3c1 = 0; d3c2 = 0;
D1a = []; D1b = []; D1c = []
D2a = []; D2b = []; D2c = []
D3a = []; D3b = []; D3c = []
nap = 1 # number of analyzed patterns
napb = 1

for i in range(0,trials):

    end1 = int(len(rd1[i][0])/1); end2 = int(len(rd1[i][1])/1); end3 = int(len(rd1[i][2])/1)
    d1a1 = np.mean(rd1[i][0][0:napb]); d1a2 = np.mean(rd1[i][0][end1-nap:end1])
    d2a1 = np.mean(rd2[i][0][0:napb]); d2a2 = np.mean(rd2[i][0][end1-nap:end1])
    d3a1 = np.mean(rd3[i][0][0:napb]); d3a2 = np.mean(rd3[i][0][end1-nap:end1])
    D1a.append(d1a2-d1a1); D2a.append(d2a2-d2a1); D3a.append(d3a2-d3a1);

    d1b1 = np.mean(rd1[i][1][0:napb]); d1b2 = np.mean(rd1[i][1][end2-nap:end2])
    d2b1 = np.mean(rd2[i][1][0:napb]); d2b2 = np.mean(rd2[i][1][end2-nap:end2])
    d3b1 = np.mean(rd3[i][1][0:napb]); d3b2 = np.mean(rd3[i][1][end2-nap:end2])
    D1b.append(d1b2-d1b1); D2b.append(d2b2-d2b1); D3b.append(d3b2-d3b1);

    d1c1 = np.mean(rd1[i][2][0:napb]); d1c2 = np.mean(rd1[i][2][end3-nap:end3])
    d2c1 = np.mean(rd2[i][2][0:napb]); d2c2 = np.mean(rd2[i][2][end3-nap:end3])
    d3c1 = np.mean(rd3[i][2][0:napb]); d3c2 = np.mean(rd3[i][2][end3-nap:end3])
    D1c.append(d1c2-d1c1); D2c.append(d2c2-d2c1); D3c.append(d3c2-d3c1)

# d1a1 /= trials; d1a2 /= trials; d2a1 /= trials; d2a2 /= trials; d3a1 /= trials; d3a2 /= trials;
# d1b1 /= trials; d1b2 /= trials; d2b1 /= trials; d2b2 /= trials; d3b1 /= trials; d3b2 /= trials;
# d1c1 /= trials; d1c2 /= trials; d2c1 /= trials; d2c2 /= trials; d3c1 /= trials; d3c2 /= trials;
mD1a = np.mean(D1a); mD1b = np.mean(D1b); mD1c = np.mean(D1c);
mD2a = np.mean(D2a); mD2b = np.mean(D2b); mD2c = np.mean(D2c);
mD3a = np.mean(D3a); mD3b = np.mean(D3b); mD3c = np.mean(D3c);

sD1a = np.std(D1a); sD1b = np.std(D1b); sD1c = np.std(D1c);
sD2a = np.std(D2a); sD2b = np.std(D2b); sD2c = np.std(D2c);
sD3a = np.std(D3a); sD3b = np.std(D3b); sD3c = np.std(D3c);

before_values = [d1a1, d1b1, d1c1, d2a1, d2b1, d2c1, d3a1, d3b1, d3c1]
after_values = [d1a2, d1b2, d1c2, d2a2, d2b2, d2c2, d3a2, d3b2, d3c2]
values = [mD1a, mD1b, mD1c, mD2a, mD2b, mD2c, mD3a, mD3b, mD3c]
values_std = [sD1a, sD1b, sD1c, sD2a, sD2b, sD2c, sD3a, sD3b, sD3c]
colors = ['#9467bd', '#ff7f0e', '#008080']
bar_width = 0.15*0.5
spacing = 0.2
extra_spacing = 0.0
sns.set(font_scale = 1.5)
sns.set_style("ticks")

# Create subplots
fig = plt.figure()
ax = fig.add_subplot(111)

# Loop through each x-axis variable
for i in range(0,3):
    # Plot before and after bar charts for each subvariable
    for j in range(0,3):
        ax.bar(i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing), values[i*3+j], bar_width, color=colors[j])
        # ax.bar(i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing), before_values[i*3+j], bar_width*0.5, color=colors[j], alpha=0.5)
        # ax.bar(i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing) +bar_width*0.5, after_values[i*3+j], bar_width*0.5, color=colors[j])

# Add labels and title
# ax.set_xlabel()
ax.set_ylabel('$\mathrm{v_{d}}$ (mV)')
ax.set_xticks([bar_width * (3 - 1) / 2, 0.43+ bar_width * (3 - 1) / 2, 0.855+ bar_width * (3 - 1) / 2])
ax.set_xticklabels(['Dendrite 1', 'Dendrite 2', 'Dendrite 3'])
ax.legend()
#
for i in range(0, 10):
#     # if i!=1 and i!=3 and i!= 4:
    fig_plas, ax_plas = plt.subplots(3,1)
    plas_legend = []
    for d in range(0, 3): # (three dendrites)
        for f in range(0,3):
            for j in range(0, num_inh_syns):
                ax_plas[d].plot(weights[i][d*18 + num_inh_syns*f+j][:], color = colors[f])
        ax_plas[d].set_xlabel('t (ms)')
        ax_plas[d].set_ylabel('$\mathrm{w_{inh}}$, dend %d' % f)
        # plas_legend.append(['A', 'B', 'C'])
# ax_plas[-1].legend(plas_legend)

plt.show()
