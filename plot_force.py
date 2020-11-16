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
from casadi import MX, Function, vertcat


def muscles_forces(q, qdot, act, controls, model):
    muscles_states = biorbd.VecBiorbdMuscleState(model.nbMuscles())
    for k in range(model.nbMuscles()):
        muscles_states[k].setExcitation(controls[k])
        muscles_states[k].setActivation(act[k])
    # muscles_tau = model.muscularJointTorque(muscles_states, True,  q, qdot).to_mx()
    muscles_force = model.muscleForces(muscles_states, q, qdot).to_mx()
    return muscles_force

seaborn.set_style("whitegrid")
seaborn.color_palette()

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
qMX = MX.sym("qMX", biorbd_model.nbQ(), 1)
dqMX = MX.sym("dqMX", biorbd_model.nbQ(), 1)
aMX = MX.sym("aMX", biorbd_model.nbMuscles(), 1)
uMX = MX.sym("uMX", biorbd_model.nbMuscles(), 1)
force_func = Function(
    "MuscleForce",
    [qMX, dqMX, aMX, uMX],
    [muscles_forces(qMX, dqMX, aMX, uMX, biorbd_model)],
    ["qMX", "dqMX", "aMX", "uMX"],
    ["Force"],
).expand()

T = 0.8
Ns = 100
motion = "REACH2"
nb_try = 30
marker_noise_lvl = [0, 0.002, 0.005, 0.01]
EMG_noise_lvl = [0, 0.05, 0.1, 0.2, 0]
co_lvl = 4
folder_w_track = ["solutions/w_track_low_weight_3", "solutions/w_track_low_weight_2", "solutions/with_track_emg"]
folder_wt_track = ["solutions/wt_track_low_weight_3", "solutions/wt_track_low_weight_2", "solutions/wt_track_emg"]
count = 0
count_nc_track = 0
for fold in folder_w_track:
    for co in range(co_lvl):
    # for co in range(2, 3):
        for marker_lvl in range(len(marker_noise_lvl)):
            for EMG_lvl in range(len(EMG_noise_lvl)):
                mat_content = sio.loadmat(
                    f"{fold}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat"
                )
                Nmhe = mat_content['N_mhe']
                N = mat_content['N_tot']
                NS = int(N - Nmhe)
                X_est = mat_content['X_est']
                q_est = X_est[:, :biorbd_model.nbQ(), :]
                dq_est = X_est[:, biorbd_model.nbQ():biorbd_model.nbQ()*2, :]
                a_est = X_est[:, -biorbd_model.nbMuscles():, :]
                U_est = mat_content['U_est']
                q_ref = mat_content['x_sol'][:biorbd_model.nbQ(), :NS + 1]
                dq_ref = mat_content['x_sol'][biorbd_model.nbQ():biorbd_model.nbQ() * 2, :NS + 1]
                a_ref = mat_content['x_sol'][-biorbd_model.nbMuscles():, :NS + 1]
                u_ref = mat_content['u_sol'][:, :NS]

                q_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
                dq_ref_try = np.ndarray((nb_try, biorbd_model.nbQ(), NS + 1))
                a_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS + 1))
                u_ref_try = np.ndarray((nb_try, biorbd_model.nbMuscles(), NS))
                force_ref = np.ndarray((biorbd_model.nbMuscles(), int(Ns - Nmhe)))
                force_sol = np.ndarray((nb_try, biorbd_model.nbMuscles(), int(Ns - Nmhe)))
                for tries in range(nb_try):
                    q_ref_try[tries, :, :] = q_ref
                    dq_ref_try[tries, :, :] = dq_ref
                    a_ref_try[tries, :, :] = a_ref
                    u_ref_try[tries, :, :] = u_ref
                    for i in range(biorbd_model.nbMuscles()):
                        for k in range(int(Ns - Nmhe)):
                            force_ref[i, k] = force_func(q_ref[:, k], dq_ref[:, k], a_ref[:, k], u_ref[:, k])[i, :]
                            force_sol[tries, i, k] = force_func(
                                q_est[tries, :, k], dq_est[tries, :, k], a_est[tries, :, k], U_est[tries, :, k]
                            )[i, :]

                if 'f_est' in mat_content:
                    mat_content.pop('f_est')
                if 'f_sol' in mat_content:
                    mat_content.pop('f_sol')
                mat_content['f_est'] = force_sol
                mat_content['f_sol'] = force_ref
                sio.savemat(
                    f"{fold}/track_mhe_w_EMG_excitation_driven_co_lvl{co}_noise_lvl_{marker_noise_lvl[marker_lvl]}_{EMG_noise_lvl[EMG_lvl]}.mat",
                mat_content)

                # force_sol_mean = np.mean(force_sol, axis=0)
                # force_sol_STD = np.std(force_sol, axis=0)

                # for i in range(biorbd_model.nbMuscles()):
                #     fig = plt.subplot(4, 5, i + 1)
                #     plt.plot(force_ref[i, :], 'r', label='force ref')
                #     for tries in range(nb_try):
                #         plt.plot(force_sol[tries, i, :], 'b--', label='force est')
                #     # plt.fill_between(force_sol_mean[i, :] - force_sol_STD[i, :],
                #     #                  force_sol_mean[i, :] + force_sol_STD[i, :],
                #     #                  alpha=0.2, color='blue')
                #     plt.title(biorbd_model.muscleNames()[i].to_string())
                # plt.legend(bbox_to_anchor=(1.05, 0.80), loc='upper left', frameon=False)
                # plt.show()

