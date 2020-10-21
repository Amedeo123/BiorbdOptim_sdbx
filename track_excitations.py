import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt
import pickle
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    Data,
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
)

def compute_err(Ns_mhe, X_est ,U_est, Ns, model, q, dq, a, tau, excitations, nbGT):
    model = model
    get_markers = markers_fun(model)
    err = dict()
    nbGT = nbGT
    Ns = Ns
    norm_err = np.sqrt(Ns-Ns_mhe)
    q_ref = q[:, :-Ns_mhe]
    dq_ref = dq[:, :-Ns_mhe]
    a_ref = a[:, :-Ns_mhe]
    tau_ref = tau[:, :-Ns_mhe-1]
    musces_ref = excitations[:, :-Ns_mhe-1]
    sol_mark = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    err['q'] = np.linalg.norm(X_est[:model.nbQ(), :]-q_ref)/norm_err
    err['q_dot'] = np.linalg.norm(X_est[model.nbQ():model.nbQ()*2, :]-dq_ref)/norm_err
    err['a'] = np.linalg.norm(X_est[model.nbQ()*2:, :] - a_ref) / norm_err
    err['tau'] = np.linalg.norm(U_est[:nbGT, :]-tau_ref)/norm_err
    err['muscles'] = np.linalg.norm(U_est[nbGT:, :]-musces_ref)/norm_err
    for i in range(Ns+1-Ns_mhe):
        sol_mark[:, :, i] = get_markers(X_est[:model.nbQ(), i])
        sol_mark_ref[:, :, i] = get_markers(q[:, i])
    err['markers'] = np.linalg.norm(sol_mark - sol_mark_ref)/norm_err
    return err


def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX], [biorbd_model.markers(qMX)])


def prepare_ocp(
        biorbd_model,
        final_time,
        x0,
        nbGT,
        number_shooting_points,
        use_SX=False,
        nb_threads=1,
        use_activation=True,
        use_torque=True
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd_model
    nbQ = biorbd_model.nbQ()
    nbGT = nbGT
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    activation_min, activation_max, activation_init = 0, 1, 0.2

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_activation and use_torque:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    elif use_activation is not True and use_torque:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    elif use_activation and use_torque is not True:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)
    elif use_activation is not True and use_torque is not True:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    if use_activation is not True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * nbGT + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * nbGT + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialGuessOption(np.tile(x0, (number_shooting_points+1, 1)).T,
                                     interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)+0.1
    u_init = InitialGuessOption(np.tile(u0, (number_shooting_points, 1)).T,
                                     interpolation=InterpolationType.EACH_FRAME)
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
    use_activation = False
    use_torque = False
    use_ACADOS = True
    T = 0.5
    Ns = 150
    motion = 'REACH2'
    i = '0'
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    with open(
            f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}_1.bob", 'rb'
    ) as file:
        data = pickle.load(file)
    states = data['data'][0]
    controls = data['data'][1]
    q_sol = states['q']
    dq_sol = states['q_dot']
    a_sol = states['muscles']
    u_sol = controls['muscles']
    if use_torque:
        nbGT = biorbd_model.nbGeneralizedTorque()
    else:
        nbGT = 0
    nbMT = biorbd_model.nbMuscleTotal()
    nbQ = biorbd_model.nbQ()
    w_tau = 'tau' in controls.keys()
    if w_tau:
        tau = controls['tau']
    else:
        tau = np.zeros((nbGT, Ns+1))
    if use_activation:
        x0 = np.hstack([q_sol[:, 0], dq_sol[:, 0]])
    else:
        x0 = np.hstack([q_sol[:, 0], dq_sol[:, 0], a_sol[:, 0]])
    tau_init = 0
    muscle_init = 0.5


    # get targets
    get_markers = markers_fun(biorbd_model)
    markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns+1))
    for i in range(Ns+1):
        markers_target[:, :, i] = get_markers(q_sol[:, i])
    muscles_target = u_sol

    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T, x0=x0, nbGT=nbGT,
                      number_shooting_points=Ns, use_torque=use_torque, use_activation=use_activation, use_SX=use_ACADOS)

    # set initial state
    ocp.nlp[0].x_bounds.min[:, 0] = x0
    ocp.nlp[0].x_bounds.max[:, 0] = x0

    # set initial guess on state
    x_init = InitialGuessOption(x0, interpolation=InterpolationType.CONSTANT)
    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuessOption(u0, interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    objectives = ObjectiveList()
    objectives.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=10000, target=muscles_target)
    objectives.add(Objective.Lagrange.TRACK_MARKERS, weight=100000, target=markers_target)
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(nbQ)))
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(nbQ, nbQ * 2)))
    if use_activation is not True:
        objectives.add(
            Objective.Lagrange.MINIMIZE_STATE, weight=10, states_idx=np.array(range(nbQ * 2, nbQ * 2 + nbMT))
        )
    if use_torque:
        objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)
    ocp.update_objectives(objectives)

    if use_ACADOS:
        tic = time()
        sol = ocp.solve(solver=Solver.ACADOS,
                        show_online_optim=False,
                        solver_options={
                            "nlp_solver_tol_comp": 1e-4,
                            "nlp_solver_tol_eq": 1e-4,
                            "nlp_solver_tol_stat": 1e-4,
                            "integrator_type": "IRK",
                            "nlp_solver_type": "SQP",
                            "sim_method_num_steps": 1,
                        })
        print(f"Time to solve with ACADOS : {time()-tic} s")
    else:
        tic = time()
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
            })
        print(f"Time to solve with IPOPT : {time() - tic} s")

    toc = time() - tic
    print(f"Total time to solve with ACADOS : {toc} s")

    data_sol = Data.get_data(ocp, sol)
    if use_activation:
        X_est = np.vstack([data_sol[0]['q'], data_sol[0]['q_dot']])
    else:
        X_est = np.vstack([data_sol[0]['q'], data_sol[0]['q_dot'], data_sol[0]['muscles']])
    if use_torque:
        U_est = np.vstack([data_sol[1]['tau'], data_sol[1]['muscles']])
    else:
        U_est = data_sol[1]['muscles']
    err_offset = 15
    err = compute_err(
        err_offset,
        X_est[:, :-err_offset], U_est[:, :-err_offset-1], Ns, biorbd_model, q_sol, dq_sol, a_sol, tau, u_sol, nbGT
    )
    if use_torque is not True:
        err['tau'] = None
    print(err)
    f = open(f"solutions/stats_flex_activ{use_activation}_torque{use_torque}_acados{use_ACADOS}.txt", "a")
    f.write(f"{Ns}; {toc}; {err['q']}; {err['q_dot']}; {err['tau']}; "
            f"{err['muscles']}; {err['markers']}\n")
    f.close()

    plt.subplot(211)
    plt.plot(X_est[:biorbd_model.nbQ(), :].T, 'x')
    plt.gca().set_prop_cycle(None)
    plt.plot(q_sol.T)
    plt.legend(labels=['Q estimate', 'Q truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(X_est[biorbd_model.nbQ():, :].T, 'x')
    plt.gca().set_prop_cycle(None)
    plt.plot(dq_sol.T)
    plt.legend(labels=['Qdot estimate', 'Qdot truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    # plt.tight_layout()

    plt.figure()
    if use_torque:
        plt.subplot(211)
        plt.plot(U_est[:nbGT, :].T, 'x', label='Tau estimate')
        plt.gca().set_prop_cycle(None)
        plt.plot(tau.T)
        plt.legend(labels=['Tau estimate', 'Tau truth'], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
        plt.subplot(212)
    plt.plot(U_est[nbGT:, :].T, 'x')
    plt.gca().set_prop_cycle(None)
    plt.plot(u_sol.T)
    plt.legend(
        labels=['Muscle excitation estimate', 'Muscle excitation truth'],
        bbox_to_anchor=(1, 1),
        loc='upper left', borderaxespad=0.
    )
    plt.tight_layout()
    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
