import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
use_torque = False
use_noise = False

nb_try = 100
matcontent = sio.loadmat("solutions/stats_ac_noiseFalse_tries.mat")
err_mat = matcontent['err_tries']
err_mat_mhe = err_mat[:-1, :].reshape(-1, nb_try, 7)
err_mat_full = err_mat[-1, :].reshape(1, 7)
err_mean_mhe = np.mean(err_mat_mhe, axis=1)
err_max_mhe = np.max(err_mat_mhe, axis=1)
err_min_mhe = np.min(err_mat_mhe, axis=1)
err_mean = np.concatenate((err_mean_mhe, err_mat_full))
err_test = sio.loadmat("solutions/stats_ac_noiseFalse.mat")['err_tries']
err_test_mhe = err_test[:-1, :].reshape(-1, nb_try, 7)
err_mean_test = np.mean(err_test_mhe[-1, :, :], axis=1)
time_test = err_mean_test[:, 1]
Nmhe = err_mean[:, 0]
time = err_mean[:, 1]
time_mean = err_mean_mhe[:, 1]
time_max = err_max_mhe[:, 1]
time_min = err_min_mhe[:, 1]  
err_q = (err_mean[:, 2])
err_dq = (err_mean[:, 3])
err_tau = err_mean[:, 4]
err_muscles = (err_mean[:, 5])
err_markers = (err_mean[:, 6])

fig = plt.subplot()
plt.plot(1/time_mean, label='mhe multi thread', c='blue')
plt.plot(1/time_test, 'rx', label='mhe one thread')
plt.plot(np.arange(len(Nmhe)-1), np.tile(1/0.075, (len(Nmhe)-1, 1)), '--', label='biofeedback standard', c='orange')
plt.fill_between(range(len(1/time_min)), 1/time_min, 1/time_max, facecolor='blue', alpha=0.2)
fig.set_xticks(range(len(Nmhe)-1))
fig.set_xticklabels(Nmhe[:-1])
plt.legend()
plt.ylabel('Freq. (Hz)')
plt.xlabel('Size of MHE window')

fig=plt.figure()
fig = plt.subplot(511)
plt.plot(err_q[:-1], 'x', label='err. mhe')
plt.plot(np.arange(len(Nmhe)-1), np.tile(err_q[-1], (len(Nmhe)-1, 1)), '--',  label='err. full window')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-3), (len(Nmhe)-1, 1)), c='red', label = 'limit_err_1e-3')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-4), (len(Nmhe)-1, 1)), c='purple', label = 'limit_err_1e-4')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-5), (len(Nmhe)-1, 1)), 'k', label = 'limit_err_1e-5')

fig.set_xticks(range(len(Nmhe)-1))
fig.set_xticklabels(Nmhe[:-1])
plt.ylabel('q err. (rad)')
plt.xlabel('Size of MHE window')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(512)
plt.plot(err_dq[:-1], 'x', label='err. mhe')
plt.plot(np.arange(len(Nmhe)-1), np.tile(err_dq[-1], (len(Nmhe)-1, 1)), '--',  label='err. full window')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-3), (len(Nmhe)-1, 1)), c='red', label = 'limit_err_1e-3')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-4), (len(Nmhe)-1, 1)), c='purple', label = 'limit_err_1e-4')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-5), (len(Nmhe)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe)-1))
fig.set_xticklabels(Nmhe[:-1])
plt.ylabel('dq err. (rad/s)')
plt.xlabel('Size of MHE window')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

# fig = plt.subplot(513)
# plt.plot(err_tau[:-1], 'x', label='err. mhe')
# plt.plot(np.arange(len(Nmhe)-1), np.tile(err_tau[-1], (len(Nmhe)-1, 1)), '--',  label='err. full window')
# fig.set_xticks(range(len(Nmhe)-1))
# fig.set_xticklabels(Nmhe[:-1])
# plt.ylabel('Tau err. (Nm)')
# plt.xlabel('Size of MHE window')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(513)
plt.plot(err_muscles[:-1], 'x', label='err. mhe')
plt.plot(np.arange(len(Nmhe)-1), np.tile(err_muscles[-1], (len(Nmhe)-1, 1)), '--',  label='err. full window')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-3), (len(Nmhe)-1, 1)), c='red', label = 'limit_err_1e-3')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-4), (len(Nmhe)-1, 1)), c='purple', label = 'limit_err_1e-4')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-5), (len(Nmhe)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe)-1))
fig.set_xticklabels(Nmhe[:-1])
plt.ylabel('Muscle act. err.')
plt.xlabel('Size of MHE window')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

fig = plt.subplot(514)
plt.plot(err_markers[:-1], 'x', label='err. mhe')
plt.plot(np.arange(len(Nmhe)-1), np.tile(err_markers[-1], (len(Nmhe)-1, 1)), '--',  label='err. full window')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-3), (len(Nmhe)-1, 1)), c='red', label = 'limit_err_1e-3')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-4), (len(Nmhe)-1, 1)), c='purple', label = 'limit_err_1e-4')
plt.plot(np.arange(len(Nmhe)-1), np.tile(np.array(1e-5), (len(Nmhe)-1, 1)), 'k', label = 'limit_err_1e-5')
fig.set_xticks(range(len(Nmhe)-1))
fig.set_xticklabels(Nmhe[:-1])
plt.ylabel('Marker err. (m)')
plt.xlabel('Size of MHE window')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.show()