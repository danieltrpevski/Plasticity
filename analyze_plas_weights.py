import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


filename = './results/new_plas_weights.dat'
with open(filename, 'r', encoding = 'utf-8') as f:
  fload  = json.load(f)
  res_dict = json.loads(fload)
e_weights = res_dict['e_weights']
trials = res_dict['trials']
output_freqs = res_dict['output_freqs']
caint = res_dict['caint']

t = np.arange(2, 101, 1)
output_freqs = np.reshape(output_freqs, (len(e_weights), trials, len(t)))
caint = np.reshape(caint, (len(e_weights), trials, 50000))
for i in range(0, len(e_weights)):
    for j in range(0,trials):
        for k in range(0, len(t)):
            # if i < 8:
                if output_freqs[i,j,k] > 100 and k!= 0 and k!= len(t)-1:
                    print("Here %f" % output_freqs[i,j,k])
                    output_freqs[i,j,k] = 0.5*(output_freqs[i,j,k-1] +output_freqs[i,j,k+1])
                elif k == 0:
                    output_freqs[i,j,k] = output_freqs[i,j,k+1]
                elif k == len(t)-1:
                    output_freqs[i,j,k] = output_freqs[i,j,k-1]

mean_output_freqs = np.mean(output_freqs, axis = (1))

sns.set(font_scale = 1.5)
sns.set_style("ticks")
colors = sns.color_palette("Blues", len(e_weights))

fig_plas = plt.figure()
ax_plas = fig_plas.add_subplot(111)
plas_legend = []
for i in range(0, len(e_weights)):
    # if i!=1 and i!=3 and i!= 4:
        ax_plas.plot(t[18:], mean_output_freqs[i, 18:], color = colors[i])
        w = e_weights[i]*1e6
        plas_legend.append("%f (pS)" % w)
# ax_plas.legend(plas_legend)
ax_plas.set_xlabel('t (s)')
ax_plas.set_ylabel('$\mathrm{f_{out}}$ (Hz)')

t = np.arange(0, 100000, 2)
t = t*1e-3
fig_caint = plt.figure()
ax_caint = fig_caint.add_subplot(111)
for i in range(0, len(e_weights)):
    # if  i!=3 and i!=7:
        ax_caint.plot(t,np.asarray(caint[i,0, :])*1000, color = colors[i])
# ax_caint.legend(plas_legend)
ax_caint.set_ylabel('$\mathrm{F([Ca^{2+}]_i) (\mu}$M ms) ')
ax_caint.set_xlabel('t(s)')

plt.show()
