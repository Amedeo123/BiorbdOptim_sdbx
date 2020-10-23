import numpy as np
import pickle
import biorbd
import scipy.io as sio
import matplotlib.pyplot as plt
from casadi import MX, SX, Function


def generate_noise(model, markers, excitations):
    T = 0.5
    u_co = excitations
    t = np.linspace(0, T, u_co.shape[1])
    biorbd_model = model
    # EMG : standard deviation of emg are 15%
    EMG_w_noise = np.ndarray((biorbd_model.nbMuscles(), u_co.shape[1]))
    for i in range(biorbd_model.nbMuscles()):
        EMG_noise = np.random.normal(0, u_co[i, :] * 0.15, u_co[i, :].shape)
        EMG_w_noise[i, :] = u_co[i, :] + EMG_noise

    # plt.figure("Muscles controls")
    # for i in range(biorbd_model.nbMuscles()):
    #     plt.subplot(4, 5, i + 1)
    #     plt.step(t, u_co[i, :])
    #     plt.step(t, EMG_w_noise[i, :], c='purple')
    #     plt.title(biorbd_model.muscleNames()[i].to_string())
    # plt.legend(labels=['without_noise', 'with_noise'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.show()

    # Marker : deviation is 0.3cm thorax, 0.4cm clav, 0.5 scap, 0.8cm rad/ulna, 1cm humerus
    n_mark = biorbd_model.nbMarkers()
    markers_w_noise = np.ndarray((3, n_mark, u_co.shape[1]))

    # humerus
    for i in [0, 1, 2, 3]:
        marker_noise = np.random.normal(0, 1e-2, markers_w_noise.shape[2])
        markers_w_noise[:, i, :] = markers[:, i, :] + np.tile(marker_noise, 3).reshape(3, len(marker_noise))

    for i in [4, 5, 6, 7]:
        marker_noise = np.random.normal(0, 8e-3, markers_w_noise.shape[2])
        markers_w_noise[:, i, :] = markers[:, i, :] + np.tile(marker_noise, 3).reshape(3, len(marker_noise))

    # plt.figure("Markers")
    # for i in range(markers.shape[1]):
    #     plt.plot(np.linspace(0, 1, markers_w_noise.shape[2]), markers[:, i, :].T, "r--")
    #     plt.plot(np.linspace(0, 1, markers_w_noise.shape[2]), markers_w_noise[:, i, :].T, "k")
    # plt.xlabel("Time")
    # plt.ylabel("Markers Position")
    # plt.show()

    return markers_w_noise, EMG_w_noise




