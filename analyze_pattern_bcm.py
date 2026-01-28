import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


filename = './results/pattern/long/pattern_homo_bcm_new.dat'
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


# d1a1 = 0; d1a2 = 0; d2a1 = 0; d2a2 = 0; d3a1 = 0; d3a2 = 0;
# d1b1 = 0; d1b2 = 0; d2b1 = 0; d2b2 = 0; d3b1 = 0; d3b2 = 0;
# d1c1 = 0; d1c2 = 0; d2c1 = 0; d2c2 = 0; d3c1 = 0; d3c2 = 0;
D1a = []; D1b = []; D1c = []
D2a = []; D2b = []; D2c = []
D3a = []; D3b = []; D3c = []
nap = 10 # number of analyzed patterns
napb = 10

for i in range(0,trials):

    end1 = int(len(rd1[i][0])/1); end2 = int(len(rd1[i][1])/1); end3 = int(len(rd1[i][2])/1)
    d1a1 = np.mean(rd1[i][0][0:napb]); d1a2 = np.mean(rd1[i][0][end1-nap:end1])
    d2a1 = np.mean(rd2[i][0][0:napb]); d2a2 = np.mean(rd2[i][0][end1-nap:end1])
    d3a1 = np.mean(rd3[i][0][0:napb]); d3a2 = np.mean(rd3[i][0][end1-nap:end1])

    d1b1 = np.mean(rd1[i][1][0:napb]); d1b2 = np.mean(rd1[i][1][end2-nap:end2])
    d2b1 = np.mean(rd2[i][1][0:napb]); d2b2 = np.mean(rd2[i][1][end2-nap:end2])
    d3b1 = np.mean(rd3[i][1][0:napb]); d3b2 = np.mean(rd3[i][1][end2-nap:end2])

    d1c1 = np.mean(rd1[i][2][0:napb]); d1c2 = np.mean(rd1[i][2][end3-nap:end3])
    d2c1 = np.mean(rd2[i][2][0:napb]); d2c2 = np.mean(rd2[i][2][end3-nap:end3])
    d3c1 = np.mean(rd3[i][2][0:napb]); d3c2 = np.mean(rd3[i][2][end3-nap:end3])

# d1a1 /= trials; d1a2 /= trials; d2a1 /= trials; d2a2 /= trials; d3a1 /= trials; d3a2 /= trials;
# d1b1 /= trials; d1b2 /= trials; d2b1 /= trials; d2b2 /= trials; d3b1 /= trials; d3b2 /= trials;
# d1c1 /= trials; d1c2 /= trials; d2c1 /= trials; d2c2 /= trials; d3c1 /= trials; d3c2 /= trials;

values = [d1a1, d1b1, d1c1, d1a2, d1b2, d1c2, d2a1, d2b1, d2c1,
          d2a2, d2b2, d2c2, d3a1, d3b1, d3c1, d3a2, d3b2, d3c2]

colors = ['#9467bd', '#ff7f0e', '#008080']
bar_width = 0.2*0.5
spacing = 0.15
extra_spacing = 0.0
sns.set(font_scale = 1.5)
sns.set_style("ticks")

# Create subplots
fig = plt.figure()
ax = fig.add_subplot(111)

# Loop through each x-axis variable
for i in range(0,6):
    # Plot before and after bar charts for each subvariable
    for j in range(0,3):
        if not i%2:
            ax.bar((i%2 - 1)*(-1)*spacing/3 + i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing), values[i*3+j], bar_width, color=colors[j], alpha = 0.5)
        else:
            ax.bar((i%2 - 1)*(-1)*spacing/3 + i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing), values[i*3+j], bar_width, color=colors[j])
        # ax.bar(i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing), before_values[i*3+j], bar_width*0.5, color=colors[j], alpha=0.5)
        # ax.bar(i*spacing + i*bar_width*3 + j*(bar_width+extra_spacing) +bar_width*0.5, after_values[i*3+j], bar_width*0.5, color=colors[j])

# Add labels and title
ax.set_ylabel('$\mathrm{v_{d}}$ (mV)')
ax.set_xticks([0.35, 0.55 + 0.7, 2.15])
ax.set_xticklabels(['Dendrite 1', 'Dendrite 2', 'Dendrite 3'])
ax.legend()

# for i in range(0, 10):
#     # if i!=1 and i!=3 and i!= 4:
#     fig_plas, ax_plas = plt.subplots(3,1)
#     plas_legend = []
#     for d in range(0, 3): # (three dendrites)
#         for f in range(0,3):
#             for j in range(0, num_inh_syns):
#                 ax_plas[d].plot(tw, weights[i][0][d*18 + num_inh_syns*f+j][:], color = colors[f])
#         ax_plas[d].set_xlabel('t (ms)')
#         ax_plas[d].set_ylabel('$\mathrm{w_{inh}}$, dend %d' % f)
        # plas_legend.append(['A', 'B', 'C'])
# ax_plas.legend(plas_legend)

plt.show()
