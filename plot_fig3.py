import pickle
import bioviz
import matplotlib.pyplot as plt
import biorbd
import scipy.io as sio
import numpy as np
import csv
import pandas as pd
import seaborn
from matplotlib.colors import LogNorm

seaborn.set_style("whitegrid")
seaborn.color_palette()

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 0.8
Ns = 100
motion = "REACH2"
nb_try = 20
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 0.05, 0.1, 0.2, 0]
EMG_lvl_label = ['track, n_lvl=0', 'track, n_lvl=0.05', 'track, n_lvl=0.1', 'track, n_lvl=0.2', 'minimize']
states_controls = ['q', 'dq', 'act', 'exc']
co_lvl = 4
co_lvl_label = ['None', 'low', 'mid', 'high']
RMSEmin = np.ndarray((co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 4 * nb_try))
RMSEtrack = np.ndarray((co_lvl * len(marker_noise_lvl) * len(EMG_noise_lvl) * 4 * nb_try))
W_LOW_WEIGHTS = False

co_lvl_df = [co_lvl_label[0]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try \
            + [co_lvl_label[1]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try \
            + [co_lvl_label[2]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try \
            + [co_lvl_label[3]]*len(marker_noise_lvl)*len(EMG_noise_lvl)*4*nb_try

marker_n_lvl_df = ([marker_noise_lvl[0]]*len(EMG_noise_lvl)*4*nb_try
                   + [marker_noise_lvl[1]]*len(EMG_noise_lvl)*4*nb_try
                   + [marker_noise_lvl[2]]*len(EMG_noise_lvl)*4*nb_try
                   + [marker_noise_lvl[3]]*len(EMG_noise_lvl)*4*nb_try)*co_lvl

EMG_n_lvl_df = ([EMG_lvl_label[0]]*4*nb_try + [EMG_lvl_label[1]]*4*nb_try
                + [EMG_lvl_label[2]]*4*nb_try + [EMG_lvl_label[3]]*4*nb_try
                + [EMG_lvl_label[4]]*4*nb_try)*co_lvl*len(marker_noise_lvl)

states_controls_df = ([states_controls[0]]*nb_try + [states_controls[1]]*nb_try + [states_controls[2]]*nb_try
                      + [states_controls[3]]*nb_try)*co_lvl*len(marker_noise_lvl)*len(EMG_noise_lvl)
count = 0
count_nc_min = 0
count_nc_track = 0
for co in range(co_lvl):
    for marker_lvl in range(len(marker_noise_lvl)):
        for EMG_lvl in range(len(EMG_noise_lvl)):
            if W_LOW_WEIGHTS:
                mat_content_min = sio.loadmat(
                    f"solutions/wt_track_low_weight/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )
                mat_content_track = sio.loadmat(
                    f"solutions/w_track_low_weight/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )
            else:
                mat_content_min = sio.loadmat(
                    f"solutions/wt_track_emg/track_mhe_wt_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )
                mat_content_track = sio.loadmat(
                    f"solutions/with_track_emg/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )

            Nmhe = mat_content_track['N_mhe']
            N = mat_content_track['N_tot']
            NS = int(N - Nmhe)

            Xt_est = mat_content_track['X_est']
            Ut_est = mat_content_track['U_est']
            qt_ref = mat_content_track['x_sol'][:biorbd_model.nbQ(), :NS + 1]
            dqt_ref = mat_content_track['x_sol'][biorbd_model.nbQ():biorbd_model.nbQ() * 2, :NS + 1]
            at_ref = mat_content_track['x_sol'][-biorbd_model.nbMuscles():, :NS + 1]
            ut_ref = mat_content_track['u_sol'][:, :NS]

            Xm_est = mat_content_min['X_est']
            Um_est = mat_content_min['U_est']
            qm_ref = mat_content_min['x_sol'][:biorbd_model.nbQ(), :NS + 1]
            dqm_ref = mat_content_min['x_sol'][biorbd_model.nbQ():biorbd_model.nbQ() * 2, :NS + 1]
            am_ref = mat_content_min['x_sol'][-biorbd_model.nbMuscles():, :NS + 1]
            um_ref = mat_content_min['u_sol'][:, :NS]

            qt_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
            dqt_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
            at_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS + 1))
            ut_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS))

            qm_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
            dqm_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
            am_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS + 1))
            um_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS))

            for i in range(nb_try):
                # already_cnt_min = False
                # already_cnt_track = False
                with open('solutions/status_track_EMGTrue.txt') as f:
                    if f"9; {co}; {marker_lvl}; {EMG_lvl}; {i}" in f.read():
                        qt_ref_try[i, :, :] = np.nan
                        dqt_ref_try[i, :, :] = np.nan
                        at_ref_try[i, :, :] = np.nan
                        ut_ref_try[i, :, :] = np.nan
                        # if not already_cnt_track:
                        count_nc_track += 1
                            # already_cnt_track = True
                    else:
                        qt_ref_try[i, :, :] = qt_ref
                        dqt_ref_try[i, :, :] = dqt_ref
                        at_ref_try[i, :, :] = at_ref
                        ut_ref_try[i, :, :] = ut_ref

                with open('solutions/status_track_EMGFalse.txt') as f:
                    if f"9; {co}; {marker_lvl}; {EMG_lvl}; {i}" in f.read():
                        qm_ref_try[i, :, :] = np.nan
                        dqm_ref_try[i, :, :] = np.nan
                        am_ref_try[i, :, :] = np.nan
                        um_ref_try[i, :, :] = np.nan
                        # if not already_cnt_min:
                        count_nc_min += 1
                            # already_cnt_min = True

                    else:
                        qm_ref_try[i, :, :] = qm_ref
                        dqm_ref_try[i, :, :] = dqm_ref
                        am_ref_try[i, :, :] = am_ref
                        um_ref_try[i, :, :] = um_ref



            Qmin_err = np.linalg.norm(Xm_est[:, :biorbd_model.nbQ(), :] - qm_ref_try, axis=2) / np.sqrt(NS + 1)
            Qmin_err = np.nanmean(Qmin_err, axis=1)
            DQmin_err = np.linalg.norm(
                Xm_est[:, biorbd_model.nbQ():biorbd_model.nbQ() * 2, :] - dqm_ref_try, axis=2) / np.sqrt(NS + 1)
            DQmin_err = np.nanmean(DQmin_err, axis=1)
            Amin_err = np.linalg.norm(
                Xm_est[:, -biorbd_model.nbMuscles():, :] - am_ref_try, axis=2) / np.sqrt(NS + 1)
            Amin_err = np.nanmean(Amin_err, axis=1)
            Umin_err = np.linalg.norm(
                Um_est[:, -biorbd_model.nbMuscles():, :] - um_ref_try, axis=2) / np.sqrt(NS)
            Umin_err = np.nanmean(Umin_err, axis=1)

            Qtrack_err = np.linalg.norm(Xt_est[:, :biorbd_model.nbQ(), :] - qt_ref_try, axis=2) / np.sqrt(NS + 1)
            Qtrack_err = np.nanmean(Qtrack_err, axis=1)
            DQtrack_err = np.linalg.norm(
                Xt_est[:, biorbd_model.nbQ():biorbd_model.nbQ() * 2, :] - dqt_ref_try, axis=2) / np.sqrt(NS + 1)
            DQtrack_err = np.nanmean(DQtrack_err, axis=1)
            Atrack_err = np.linalg.norm(
                Xt_est[:, -biorbd_model.nbMuscles():, :] - at_ref_try, axis=2) / np.sqrt(NS + 1)
            Atrack_err = np.nanmean(Atrack_err, axis=1)
            Utrack_err = np.linalg.norm(
                Ut_est[:, -biorbd_model.nbMuscles():, :] - ut_ref_try, axis=2) / np.sqrt(NS)
            Utrack_err = np.nanmean(Utrack_err, axis=1)

            markers_target_mean = np.mean(mat_content_min['markers_target'], axis=0)
            u_target_mean = np.mean(mat_content_min['u_target'], axis=0)

            if EMG_lvl == 4:
                RMSEtrack[count:count+nb_try] = Qmin_err
                RMSEtrack[count+nb_try:count+2*nb_try] = DQmin_err
                RMSEtrack[count+2*nb_try:count+3*nb_try] = Amin_err
                RMSEtrack[count+3*nb_try:count+4*nb_try] = Umin_err
            else:
                RMSEtrack[count:count+nb_try] = Qtrack_err
                RMSEtrack[count+nb_try:count+2*nb_try] = DQtrack_err
                RMSEtrack[count+2*nb_try:count+3*nb_try] = Atrack_err
                RMSEtrack[count+3*nb_try:count+4*nb_try] = Utrack_err
            count += 4*nb_try

print(f"Number of optim: {int(count/5)}")
print(f"Number of optim convergence with EMG tracking: {count_nc_track}")
print(f"Number of optim convergence without EMG tracking: {count_nc_min}")

print(f"Convergence rate with EMG tracking: {100-count_nc_track/(count/5)*100}%")
print(f"Convergence rate without EMG tracking: {100-count_nc_min/(count/5)*100}%")
RMSEtrack_pd = pd.DataFrame({"RMSE": RMSEtrack, "co-contraction level": co_lvl_df, "EMG_objective": EMG_n_lvl_df,
                             "Marker noise level (m)": marker_n_lvl_df, "component": states_controls_df})


ax = seaborn.boxplot(y = RMSEtrack_pd['RMSE'][RMSEtrack_pd['component'] == 'exc'], x = RMSEtrack_pd['co-contraction level'],
                hue=RMSEtrack_pd['EMG_objective'])

if W_LOW_WEIGHTS:
    title_str = "with lower weights on markers"
else:
    title_str = "with higher weights on markers"
ax.set(ylabel='RMSE on muscle excitations')
ax.xaxis.get_label().set_fontsize(20)
ax.yaxis.get_label().set_fontsize(20)
ax.legend(title='EMG objective')
ax.tick_params(labelsize=15)
plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
plt.title(f'Error on excitations {title_str}', fontsize=20)
plt.figure()

#& RMSEtrack_pd['co-contraction level'] == 0
ax2 = seaborn.boxplot(y = RMSEtrack_pd['RMSE'][(RMSEtrack_pd['component'] == 'q')],
                      x = RMSEtrack_pd['Marker noise level'],
                      hue=RMSEtrack_pd['EMG_objective'],)

ax2.set(ylabel='RMSE on joint positions (rad)')
ax2.xaxis.get_label().set_fontsize(20)
ax2.yaxis.get_label().set_fontsize(20)
ax2.tick_params(labelsize=15)
ax2.legend(title='EMG objective')
plt.setp(ax2.get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax2.get_legend().get_title(), fontsize='20') # for legend title
plt.title(f'Error on joint positions {title_str}', fontsize=20)
plt.show()