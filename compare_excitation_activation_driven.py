import numpy as np
import biorbd
import pickle
import scipy.io as sio
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt


use_torque = False
T = 0.5
Ns = 150
motion = 'REACH2'
i = '0'
biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
with open(
        f"solutions/tracking_excitations_activations_driven_torque{use_torque}.bob", "rb"
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_act = states['q']
dq_act = states['q_dot']
# a_act = states['muscles']
a_act = controls['muscles']

with open(
        f"solutions/tracking_excitations_excitations_driven_torque{use_torque}.bob", "rb"
) as file:
    data = pickle.load(file)
states = data['data'][0]
controls = data['data'][1]
q_exc = states['q']
dq_exc = states['q_dot']
a_exc = states['muscles']
u_exc = controls['muscles']

plt.figure('excitation driven vs activation driven')
for i in range(biorbd_model.nbMuscles()):
    plt.subplot(4, 5, i + 1)
    plt.plot(a_act[i, :])
    plt.plot(a_exc[i, :])
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(
    labels=['activation_driven', 'excitation_driven'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.
)
plt.figure()
RMSE = np.sqrt(np.sum((a_exc - a_act)**2, axis=1)/Ns)
x1 = range(biorbd_model.nbMuscles())
width = 0.2
plt.bar(x1, RMSE, width, color='red')
plt.xlabel('muscles')
plt.ylabel('RMSE')
plt.show()