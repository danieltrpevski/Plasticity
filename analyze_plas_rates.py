import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import efel

filename = './results/new_plas_rates.dat'
with open(filename, 'r', encoding = 'utf-8') as f:
  fload  = json.load(f)
  res_dict = json.loads(fload)
e_rates = res_dict['e_rates']
trials = res_dict['trials']
output_freqs = res_dict['output_freqs']
caint = res_dict['caint']
# w_inh = res_dict['w_inh']
# vs = res_dict['vs']
t = np.arange(2, 101, 1)
output_freqs = np.reshape(output_freqs, (len(e_rates), trials, len(t)))

for i in range(0, len(e_rates)):
    for j in range(0,trials):
        for k in range(0, len(t)):
            # if i <= 6:
                if output_freqs[i,j,k] > 35 and k!= 0 and k!= len(t)-1:
                    print("Here %f" % output_freqs[i,j,k])
                    output_freqs[i,j,k] = 0.5*(output_freqs[i,j,k-1] +output_freqs[i,j,k+1])
                elif k == 0:
                    output_freqs[i,j,k] = output_freqs[i,j,k+1]
                elif k == len(t)-1:
                    output_freqs[i,j,k] = output_freqs[i,j,k-1]


mean_output_freqs = np.mean(output_freqs, axis = (1))
std_output_freqs = np.std(output_freqs, axis = (1))
# vs = np.reshape(vs, (len(e_rates), trials, 20001))
caint = np.reshape(caint, (len(e_rates), trials, 50000))

sns.set(font_scale = 1.5)
sns.set_style("ticks")
colors = sns.color_palette("Blues", len(e_rates))
#
t = np.arange(2, 101, 1)
fig_plas = plt.figure()
ax_plas = fig_plas.add_subplot(111)
plas_legend = []
for i in range(0, len(e_rates)):
    ax_plas.plot(t[18:], mean_output_freqs[i, 18:], color = colors[i])
    # ax_plas.errorbar(t, mean_output_freqs[i, :], yerr=std_output_freqs[i,:],
    #                  capsize=5, color = colors[i], label='Error Bars')

plas_legend.append("%.1f" % e_rates[i])
#ax_plas.legend(plas_legend)
ax_plas.set_xlabel('t (s)')
ax_plas.set_ylabel('$\mathrm{f_{out}}$ (Hz)')
# for v in vs:
#     for vv in v:
#
#         trace = {}
#         trace['T'] = np.arange(0, 20001, 1)
#         trace['V'] = vv
#         traces = [trace]
#         freqs = []
#         for t in np.arange(2000, 20000, 1000):
#             trace['stim_start'] = [t-1000]
#             trace['stim_end'] = [t]
#             res = efel.getFeatureValues(traces, ['mean_frequency'])
#             try:
#                 f = res[0]['mean_frequency'][0]
#             except TypeError as e:
#                 f = 0
#             freqs.append(f)
#         print(freqs)
#         plt.plot(vv)
#         plt.show()

t = np.arange(0, 100000, 2)
t = t*1e-3
fig_caint = plt.figure()
ax_caint = fig_caint.add_subplot(111)
for i in range(0, len(e_rates)):
    # if  i!=3 and i!=7:
        ax_caint.plot(t,np.asarray(caint[i,0, :])*1000, color = colors[i])
# ax_caint.legend(plas_legend)
ax_caint.set_ylabel('$\mathrm{F([Ca^{2+}]_i) (\mu}$M ms)')
ax_caint.set_xlabel('t(s)')
# fig_vs = plt.figure()
# ax_vs = fig_vs.add_subplot(111)
# for i in range(0, len(e_rates)):
#     ax_vs.plot(vs[i][0])
#     ax_vs.legend([e_rates[i]])
#     plt.show()
#
# ax_vs.set_xlabel('t')
# ax_vs.set_ylabel('Vs')
#
# fig_winh = plt.figure()
# ax_winh = fig_winh.add_subplot(111)
# for w in w_inh[0][0]:
#     ax_winh.plot(w)

plt.show()
