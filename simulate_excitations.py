import biorbd
from time import time
import numpy as np
import bioviz
import pickle
import os.path
import matplotlib.pyplot as plt

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuessOption,
    ShowResult,
    Solver,
    InterpolationType,
    Bounds,
    Instant
)


def prepare_ocp(biorbd_model, final_time, number_shooting_points, x0, xT, co_value, use_SX=False, nb_threads=1):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbQ = biorbd_model.nbQ()

    # tau_min, tau_max, tau_init = -10, 10, 0
    activation_min, activation_max, activation_init = 0, 1, 0.2
    excitation_min, excitation_max, excitation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=0.1)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1, states_idx=np.array(range(nbQ)))
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(nbQ, nbQ*2)))
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_STATE,
        weight=10,
        states_idx=np.array(range(nbQ * 2, nbQ * 2 + biorbd_model.nbMuscles()))
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL,
        weight=10,
        # muscles_idx=np.array([0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16])
    )
    objective_functions.add(
        Objective.Lagrange.TRACK_MUSCLES_CONTROL,
        weight=100,
        muscles_idx=np.array([6, 7, 8, 9, 10, 17, 18]),
        target=co_value
    )

    objective_functions.add(
        Objective.Mayer.TRACK_STATE,
        weight=1000,
        target=np.tile(xT[:nbQ], (number_shooting_points+1, 1)).T,
        states_idx=np.array(range(nbQ))
    )
    # Dynamics
    dynamics = DynamicsTypeList()
    # dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # x_bounds[0][:nbQ, 0] = [0., -0.5, 0, 0]
    # x_bounds[0][nbQ:, 0] = [0, 0, 0, 0]
    # x_bounds[0].min[:, -1] = [0, 0.8, -2.3, 0, 0.6, 0, 0, 0]
    # x_bounds[0].max[:, -1] = [0., 1.5, -1.5, 0, 1.5, 0, 0, 0]
    # x_bounds[0][:, -1] = xT
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )
    x_bounds[0].min[:nbQ, 0] = [-0.1, -0.3, 0.1, -0.3]
    x_bounds[0].max[:nbQ, 0] = [-0.1, 0, 0.3, 0]
    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            # [tau_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscleTotal(),
            # [tau_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscleTotal(),
            [excitation_min] * biorbd_model.nbMuscleTotal(),
            [excitation_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(np.tile(np.concatenate(
        (x0, [activation_init] * biorbd_model.nbMuscles()))
        , (number_shooting_points+1, 1)).T, interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([excitation_init] * biorbd_model.nbMuscleTotal())
    u_init = InitialGuessOption(np.tile(u0, (number_shooting_points, 1)).T, interpolation=InterpolationType.EACH_FRAME)
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_SX=use_SX,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    T = 0.5
    Ns = 150
    x0 = []
    xT = []
    co_value = []

    motion = 'REACH2'  # 'EXT', 'REACH'
    if motion == 'EXT':
        x0 = np.array([-1, 1, 1, 1, 0, 0, 0, 0])
        xT = np.array([-1, 1, 1, 0.1, 0, 0, 0, 0])
    if motion == 'EXT2':
        x0 = np.array([-2., 1.5, 2.5, 1.3, 0., 0., 0., 0.])
        xT = np.array([-2., 0.8, 2.2, -0.2, 0., 0., 0., 0.])
    if motion == 'REACH':
        x0 = np.array([0., -0.2, 0, 0, 0, 0, 0, 0])
        xT = np.array([0.6, -1, 0, 0.5, 0, 0, 0, 0])
    if motion == 'REACH2':
        x0 = np.array([0., -0.2, 0, 0, 0, 0, 0, 0])
        xT = np.array([-0.2, -1.3, -0.5, 0.5, 0, 0, 0, 0])
        # xT = np.array([1.2, -1.9, 0, 1.1, 0, 0, 0, 0])
    use_ACADOS = True
    use_IPOPT = False
    use_BO = False
    use_CO = True

    co_weight = [0, 1.5, 2, 2.5, 3] if use_CO is True else [0]
    for i in co_weight:
        co_value = None
        if i != 0:
            with open(
                    f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_weight_0.bob", 'rb'
            ) as file:
                data = pickle.load(file)
            controls = data['data'][1]
            u_ref = controls['muscles']
            co_value = u_ref[:, :-1] * i

        if use_ACADOS:
            ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, number_shooting_points=Ns,
                              x0=x0, xT=xT, co_value=co_value, use_SX=True)

            sol = ocp.solve(
                solver=Solver.ACADOS,
                show_online_optim=False,
                solver_options={
                    "nlp_solver_max_iter": 30,
                    "nlp_solver_tol_comp": 1e-4,
                    "nlp_solver_tol_eq": 1e-4,
                    "nlp_solver_tol_stat": 1e-4,
                    "integrator_type": "IRK",
                    "nlp_solver_type": "SQP",
                    "sim_method_num_steps": 1,
                })
            if os.path.isfile(f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}.bob"):
                ocp.save_get_data(
                    sol, f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}_1.bob"
                )
            else:
                ocp.save_get_data(
                    sol, f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}.bob"
                )

        if use_IPOPT:
            ocp = prepare_ocp(biorbd_model, final_time=T, number_shooting_points=Ns,
                              x0=x0, xT=xT, co_value=co_value, use_SX=False, nb_threads=8)

            sol = ocp.solve(
                solver=Solver.IPOPT,
                show_online_optim=False,
                solver_options={
                    "tol": 1e-4,
                    "dual_inf_tol": 1e-4,
                    "constr_viol_tol": 1e-4,
                    "compl_inf_tol": 1e-4,
                    "linear_solver": "ma57",
                    "max_iter": 500,
                    "hessian_approximation": "exact",
                },
            )
            if os.path.isfile(f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}.bob"):
                ocp.save_get_data(
                    sol, f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}_1.bob"
                )
            else:
                ocp.save_get_data(
                    sol, f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}.bob"
                )

        if use_BO:
            with open(
                    f"solutions/sim_ac_{int(T*1000)}ms_{Ns}sn_{motion}_co_weight_{i}.bob", 'rb'
            ) as file:
                data = pickle.load(file)
            states = data['data'][0]
            controls = data['data'][1]
            q = states['q']
            qdot = states['q_dot']
            a = states['muscles']
            u = controls['muscles']
            # tau = controls['tau']
            t = np.linspace(0, T, Ns + 1)
            q_name = [biorbd_model.nameDof()[i].to_string() for i in range(biorbd_model.nbQ())]
            plt.figure("Q")
            for i in range(q.shape[0]):
                plt.subplot(2, 3, i + 1)
                plt.plot(t, q[i, :], c='purple')
                plt.title(q_name[i])

            plt.figure("Q_dot")
            for i in range(q.shape[0]):
                plt.subplot(2, 3, i + 1)
                plt.plot(t, qdot[i, :], c='purple')
                plt.title(q_name[i])

            # plt.figure("Tau")
            # for i in range(q.shape[0]):
            #     plt.subplot(2, 3, i + 1)
            #     plt.plot(t, tau[i, :], c='orange')
            #     plt.title(biorbd_model.muscleNames()[i].to_string())

            plt.figure("Muscles controls")
            for i in range(u.shape[0]):
                plt.subplot(4, 5, i + 1)
                plt.step(t, u[i, :], c='orange')
                plt.plot(t, a[i, :], c='purple')
                plt.title(biorbd_model.muscleNames()[i].to_string())
            plt.legend(labels=['excitations', "activations"], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()

            b = bioviz.Viz(model_path="arm_wt_rot_scap.bioMod")
            b.load_movement(q)
            b.exec()

        # --- Show results --- #
        # result = ShowResult(ocp, sol)
        # result.graphs()
        # result.animate()

