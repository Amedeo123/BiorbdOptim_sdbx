import pickle
import bioviz
import matplotlib.pyplot as plt
import biorbd
import numpy as np

biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
T = 0.5
Ns = 150
motion = "REACH2"
n_level = 4
q_co = np.ndarray((n_level, biorbd_model.nbQ(), Ns+1))
qdot_co = np.ndarray((n_level, biorbd_model.nbQ(), Ns+1))
a_co = np.ndarray((n_level, biorbd_model.nbMuscles(), Ns+1))
u_co = np.ndarray((n_level, biorbd_model.nbMuscles(), Ns+1))
for i in range(n_level):
    with open(
            f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}.bob", 'rb'
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
q_name = [biorbd_model.nameDof()[i].to_string() for i in range(biorbd_model.nbQ())]
plt.figure("Q")
for i in range(biorbd_model.nbQ()):
    plt.subplot(2, 3, i + 1)
    for j in range(n_level):
        plt.plot(t, q_co[j, i, :])
        plt.title(q_name[i])
plt.legend(range(n_level), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.figure("Q_dot")
for i in range(biorbd_model.nbQ()):
    plt.subplot(2, 3, i + 1)
    for j in range(n_level):
        plt.plot(t, qdot_co[j, i, :])
        plt.title(q_name[i])
plt.legend(range(n_level), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# plt.figure("Tau")
# for i in range(q.shape[0]):
#     plt.subplot(2, 3, i + 1)
#     plt.plot(t, tau[i, :], c='orange')
#     plt.title(biorbd_model.muscleNames()[i].to_string())

plt.figure("Muscles controls")
for i in range(biorbd_model.nbMuscles()):
    plt.subplot(4, 5, i + 1)
    for j in range(n_level):
        plt.step(t, u_co[j, i, :])
        # plt.plot(t, a[i, :], c='purple')
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(range(n_level), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

b = bioviz.Viz(model_path="arm_wt_rot_scap.bioMod")
b.load_movement(q_co[3, :, :])
b.exec()