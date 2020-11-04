import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
use_torque = False
use_noise = False
import seaborn

nb_try = 100
matcontent = sio.loadmat("solutions/stats_ac_activation_drivenTrue.mat")
err_ac = matcontent['err_tries']
err_ac_mhe = err_ac[:-1, :].reshape(-1, nb_try, 7)
err_ac_full = err_ac[-1, :].reshape(1, 7)
err_mean_ac = np.mean(err_ac_mhe, axis=1)
err_std_ac = np.std(err_ac_mhe, axis=1)
err_mean_ac_full = np.concatenate((err_mean_ac, err_ac_full))
Nmhe_ac = (err_mean_ac_full[:, 0])
time_ac = (err_mean_ac_full[:, 1])
time_std_ac = err_std_ac[:, 1]
time_mean_ac = (err_mean_ac[:, 1])
err_q_ac = np.log10(err_mean_ac_full[:, 2])
err_std_q_ac = (err_std_ac[:, 2])
err_dq_ac = np.log10(err_mean_ac_full[:, 3])
err_std_dq_ac = np.log10(err_std_ac[:, 3])
# err_tau_ac = np.log10(err_mean_ac_full[:, 4])
err_muscles_ac = np.log10(err_mean_ac_full[:, 5])
err_std_muscles_ac = np.log10(err_std_ac[:, 5])
err_markers_ac = np.log10(err_mean_ac_full[:, 6])
err_std_markers_ac = np.log10(err_std_ac[:, 6])

matcontent = sio.loadmat("solutions/stats_ac_activation_drivenFalse.mat")
err_ex = matcontent['err_tries']
err_ex_mhe = err_ex[:-1, :].reshape(-1, nb_try, 7)
err_ex_full = err_ex[-1, :].reshape(1, 7)
err_mean_ex = np.mean(err_ex_mhe, axis=1)
err_std_ex = np.std(err_ac_mhe, axis=1)
err_mean_ex_full = np.concatenate((err_mean_ex, err_ex_full))
Nmhe_ex = err_mean_ex_full[:, 0]
time_ex = (err_mean_ex_full[:, 1])
time_mean_ex = (err_mean_ex[:, 1])
time_std_ex = (err_std_ex[:, 1])
err_q_ex = np.log10(err_mean_ex_full[:, 2])
err_std_q_ex = np.log10(err_std_ex[:, 2])
err_dq_ex = np.log10(err_mean_ex_full[:, 3])
err_std_dq_ex = np.log10(err_std_ex[:, 3])
# err_tau_ex = np.log10(err_mean_ex_full[:, 4])
err_muscles_ex = np.log10(err_mean_ex_full[:, 5])
err_std_muscles_ex = np.log10(err_std_ex[:, 5])
err_markers_ex = np.log10(err_mean_ex_full[:, 6])
err_std_markers_ex = np.log10(err_std_ex[:, 6])

# seaborn.set_style("whitegrid")
seaborn.color_palette()
# Configure plot
err_ac = '-x'
err_ex = '-^'
err_full_ac = '--'
err_full_ex = '--'
lw = 1.5
lw_err = 1.8
ms_ac = 6
ms_ex = 4
mew = 0.08
err_lim = ':'
x_y_label_size = 11
x_y_ticks = 12
legend_size = 11

#
grid_line_style = '--'
fig = plt.figure()
plt.gcf().subplots_adjust(left=0.05, bottom=0,
                       right=1, top=0.95, wspace=0, hspace=0.3)
fig = plt.subplot(511)
plt.grid(axis='x', linestyle=grid_line_style)
plt.plot(range(24), err_q_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='mhe act. driven')
plt.plot(err_q_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='mhe excit. driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw_err, label = 'limit_err_1e-3')
plt.gca().set_prop_cycle(None)
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_q_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac, label='full window act. driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_q_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex, label='full window excit. driven')

fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels([])
plt.legend(ncol=5, bbox_to_anchor=(0.5, 1.3), loc='upper center', borderaxespad=0., fontsize=legend_size)
plt.ylabel('q log err.', fontsize=x_y_label_size)
plt.yticks(fontsize=x_y_ticks)


fig = plt.subplot(512)
plt.grid(axis='x', linestyle=grid_line_style)
plt.plot(err_dq_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
plt.plot(err_dq_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw_err, label = 'limit_err_1e-3')
plt.gca().set_prop_cycle(None)
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_dq_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac,  label='err. full window activation driven')

plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_dq_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex,  label='err. full window excitation driven')

fig.set_xticks(range(len(Nmhe_ac)-1))
# fig.set_xticklabels()
fig.set_xticklabels([])
plt.yticks(fontsize=x_y_ticks)
plt.ylabel('dq log err.', fontsize=x_y_label_size)
# plt.xlabel('Size of MHE window')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(513)
plt.grid(axis='x', linestyle=grid_line_style)
plt.plot(err_muscles_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
plt.plot(err_muscles_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw_err, label = 'limit_err_1e-3')
plt.gca().set_prop_cycle(None)
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_muscles_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac,  label='err. full window activation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_muscles_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex,  label='err. full window excitation driven')


fig.set_xticks(range(len(Nmhe_ac)-1))
# fig.set_xticklabels(Nmhe_ac[:-1])
fig.set_xticklabels([])
plt.yticks(fontsize=x_y_ticks)
plt.ylabel('Activation log err.', fontsize=x_y_label_size)


fig = plt.subplot(514)
plt.grid(axis='x', linestyle=grid_line_style)
plt.plot(err_markers_ac[:-1], err_ac, lw=lw, ms=ms_ac, label='err. mhe activation driven')
plt.plot(err_markers_ex[:-1], err_ex, lw=lw, ms=ms_ac, mew=mew, label='err. mhe excitation driven')
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.log10(np.array(1e-3)), (len(Nmhe_ac)-1, 1)), err_lim, lw=lw_err, label = 'limit_err_1e-3')
plt.gca().set_prop_cycle(None)
plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(err_markers_ac[-1], (len(Nmhe_ac)-1, 1)), err_full_ac,  label='err. full window activation driven')
plt.plot(np.arange(len(Nmhe_ex)-1), np.tile(err_markers_ex[-1], (len(Nmhe_ac)-1, 1)), err_full_ex,  label='err. full window excitatiojn driven')

# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-4), (len(Nmhe_ac)-1, 1)), c='purple', label = 'limit_err_1e-4')
# plt.plot(np.arange(len(Nmhe_ac)-1), np.tile(np.array(1e-5), (len(Nmhe_ac)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe_ac)-1))
fig.set_xticklabels(range(int(Nmhe_ac[0]), int(Nmhe_ac[-2]+1)))
# fig.set_xticklabels([])
plt.ylabel('Mark. log err.', fontsize=x_y_label_size)
plt.xlabel('Size of MHE window', fontsize=x_y_label_size)
plt.xticks(fontsize=x_y_ticks)
plt.yticks(fontsize=x_y_ticks)
plt.show()