import numpy as np
import pickle
import biorbd
import matplotlib.pyplot as plt
from casadi import MX, SX, Function
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 0.5
Ns = 150
motion = "REACH2"
# co_weight = [0, 1.5, 2, 2.5, 3]
co_weight = [0]
q_co = np.ndarray((len(co_weight), biorbd_model.nbQ(), Ns+1))
qdot_co = np.ndarray((len(co_weight), biorbd_model.nbQ(), Ns+1))
a_co = np.ndarray((len(co_weight), biorbd_model.nbMuscles(), Ns+1))
u_co = np.ndarray((len(co_weight), biorbd_model.nbMuscles(), Ns+1))
for i in range(len(co_weight)):
    with open(
            f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_weight_{co_weight[i]}.bob", 'rb'
    ) as file:
        data = pickle.load(file)
    states = data['data'][0]
    controls = data['data'][1]
    q_co[i, :, :] = states['q']
    qdot_co[i, :, :] = states['q_dot']
    a_co[i, :, :] = states['muscles']
    u_co[i, :, :] = controls['muscles']
    # tau = controls['tau']
t = np.linspace(0, T, Ns + 1)
# EMG : standard deviation of emg are 15%
EMG_w_noise = np.ndarray((biorbd_model.nbMuscles(), u_co.shape[2]))
for j in co_weight:
    for i in range(biorbd_model.nbMuscles()):
        EMG_noise = np.random.normal(0, u_co[j, i, :]*0.15, u_co[j, i, :].shape)
        EMG_w_noise[i, :] = u_co[j, i, :] + EMG_noise

plt.figure("Muscles controls")
for i in range(biorbd_model.nbMuscles()):
    plt.subplot(4, 5, i + 1)
    for j in range(len(co_weight)):
        plt.step(t, u_co[j, i, :])
        plt.step(t, EMG_w_noise[i, :], c='purple')
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(co_weight, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Marker : deviation is 0.3cm thorax, 0.4cm clav, 0.5 scap, 0.8cm rad/ulna, 1cm humerus

n_mark = biorbd_model.nbMarkers()
markers = np.ndarray((3, n_mark, q_co.shape[2]))
symbolic_states = MX.sym("x", q_co.shape[1], 1)
markers_func = Function(
    "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
).expand()
for i in range(q_co.shape[2]):
    markers[:, :, i] = markers_func(q_co[0, :, i])

markers_w_noise = np.ndarray((3, n_mark, q_co.shape[2]))
#humerus
for i in [0,1,2,3]:
    marker_noise = np.random.normal(0, 1e-2, q_co.shape[2])
    markers_w_noise[:, i, :] = markers[:, i, :] + marker_noise

for i in [4,5,6,7]:
    marker_noise = np.random.normal(0, 8e-3, q_co.shape[2])
    markers_w_noise[:, i, :] = markers[:, i, :] + marker_noise

plt.figure("Markers")
for i in range(markers.shape[1]):
    plt.plot(np.linspace(0, 1, q_co.shape[2]), markers[:, i, :].T, "r--")
    plt.plot(np.linspace(0, 1, q_co.shape[2]), markers_w_noise[:, i, :].T, "k")
plt.xlabel("Time")
plt.ylabel("Markers Position")
plt.show()




