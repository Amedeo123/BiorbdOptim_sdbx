import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt
import pickle
from generate_data_noise_funct import generate_noise
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

def compute_err(Ns_mhe, X_est, U_est, Ns, model, q, dq, tau, excitations, nbGT):
    model = model
    get_markers = markers_fun(model)
    err = dict()
    nbGT = nbGT
    Ns = Ns
    norm_err = np.sqrt(Ns-Ns_mhe)
    q_ref = q[:, :-Ns_mhe]
    dq_ref = dq[:, :-Ns_mhe]
    tau_ref = tau[:, :-Ns_mhe-1]
    musces_ref = excitations[:, :-Ns_mhe-1]
    sol_mark = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    err['q'] = np.linalg.norm(X_est[:model.nbQ(), :]-q_ref)/norm_err
    err['q_dot'] = np.linalg.norm(X_est[model.nbQ():model.nbQ()*2, :]-dq_ref)/norm_err
    err['tau'] = np.linalg.norm(U_est[:nbGT, :]-tau_ref)/norm_err
    err['muscles'] = np.linalg.norm(U_est[nbGT:, :]-musces_ref)/norm_err
    for i in range(Ns+1-Ns_mhe):
        sol_mark[:, :, i] = get_markers(X_est[:model.nbQ(), i])
        sol_mark_ref[:, :, i] = get_markers(q[:, i])
    err['markers'] = np.linalg.norm(sol_mark - sol_mark_ref)/norm_err
    return err


def warm_start_mhe(ocp, sol):
    data = Data.get_data(ocp, sol)
    q = data[0]["q"]
    dq = data[0]["q_dot"]
    tau = []
    act = data[1]["muscles"]
    x = np.vstack([q, dq])
    w_tau ='tau' in data[1].keys()
    u = act
    if w_tau:
        tau = data[1]["tau"]
        u = np.vstack([tau, act])
    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, 1:]  # discard oldest estimate of the window
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out


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
        nb_threads=8,
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
    if use_torque:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

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
    use_torque = False
    use_ACADOS = True
    WRITE_STATS = False
    TRACK_EMG = True
    stats_file = 'stats_1_th'
    use_noise = True
    T = 0.5
    Ns = 150
    motion = 'REACH2'
    i = '0'
    biorbd_model = biorbd.Model("arm_wt_rot_scap.bioMod")
    with open(
            f"solutions/sim_ac_{int(T * 1000)}ms_{Ns}sn_{motion}_co_level_{i}_ref.bob", 'rb'
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
    x0 = np.hstack([q_sol[:, 0], dq_sol[:, 0]])

    tau_init = 0
    muscle_init = 0.5

    # get targets
    get_markers = markers_fun(biorbd_model)
    markers_target = np.zeros((3, biorbd_model.nbMarkers(), Ns + 1))
    for i in range(Ns + 1):
        markers_target[:, :, i] = get_markers(q_sol[:, i])
    muscles_target = u_sol

    if use_noise:
        markers_target, muscles_target = generate_noise(biorbd_model, markers_target, muscles_target)


    # setup MHE
    Ns_mhe = 25
    T_mhe = T / Ns * Ns_mhe
    X_est = np.zeros((biorbd_model.nbQ() * 2, Ns + 1 - Ns_mhe))
    U_est = np.zeros((nbGT + biorbd_model.nbMuscleTotal(), Ns - Ns_mhe))
    ocp = prepare_ocp(biorbd_model=biorbd_model, final_time=T_mhe, x0=x0, nbGT=nbGT,
                      number_shooting_points=Ns_mhe, use_torque=use_torque, use_SX=use_ACADOS)

    # set initial state
    ocp.nlp[0].x_bounds.min[:, 0] = x0
    ocp.nlp[0].x_bounds.max[:, 0] = x0

    # set initial guess on state
    x_init = InitialGuessOption(x0, interpolation=InterpolationType.CONSTANT)
    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)
    u_init = InitialGuessOption(u0, interpolation=InterpolationType.CONSTANT)
    ocp.update_initial_guess(x_init, u_init)

    objectives = ObjectiveList()
    if TRACK_EMG:
        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1000000,
                       target=muscles_target[:, :Ns_mhe],
                       )
        if use_torque:
            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)
    else:
        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1000000)
        if use_torque:
            objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=500)
    objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=100000,
                   target=markers_target[:, :, :Ns_mhe+1])
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10)
    ocp.update_objectives(objectives)

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

    x0, u0, x_out, u_out = warm_start_mhe(ocp, sol)
    X_est[:, 0] = x_out
    U_est[:, 0] = u_out
    tic = time()
    cnt = 0
    for i in range(1, Ns-Ns_mhe+1):
        cnt += 1
        # set initial state
        ocp.nlp[0].x_bounds.min[:, 0] = x0[:, 0]
        ocp.nlp[0].x_bounds.max[:, 0] = x0[:, 0]

        # set initial guess on state
        # ocp.nlp[0].X_init.init = x0
        # ocp.nlp[0].U_init.init = u0

        x_init = InitialGuessOption(x0, interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuessOption(u0, interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)

        objectives = ObjectiveList()
        if TRACK_EMG:
            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=100000,
                           target=muscles_target[:, i:Ns_mhe+i],
                           )
            if use_torque:
                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)
        else:
            objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000,
                           )
            if use_torque:
                objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=500)

        objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=10000000,
                       target=markers_target[:, :, i:Ns_mhe+i+1])
        objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10)
        ocp.update_objectives(objectives)

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
        x0, u0, x_out, u_out = warm_start_mhe(ocp, sol)
        X_est[:, i] = x_out
        if i < Ns-Ns_mhe:
            U_est[:, i] = u_out

    # X_est[:, i:] = x0
    # U_est[:, i:] = u0

    toc = time() - tic
    print(f"nb loops: {cnt}")
    print(f"Total time to solve with ACADOS : {toc} s")
    print(f"Time per MHE iter. : {toc/(Ns-Ns_mhe)} s")

    err_offset = Ns_mhe + 1
    err = compute_err(
        err_offset,
        X_est[:, :-err_offset+Ns_mhe], U_est[:, :-err_offset+Ns_mhe], Ns, biorbd_model, q_sol, dq_sol, tau, u_sol, nbGT
    )

    if use_torque is not True:
        err['tau'] = None
    if WRITE_STATS:
        f = open(f"solutions/stats_ACADOS{use_ACADOS}_torque{use_torque}.txt", "a")
        f.write(f"{Ns_mhe}; {toc/(Ns-Ns_mhe)}; {err['q']}; {err['q_dot']}; {err['tau']}; "
                f"{err['muscles']}; {err['markers']}\n")
        f.close()

    print(err)
    plt.subplot(211)
    for est, name in zip(X_est[:biorbd_model.nbQ(), :], biorbd_model.nameDof()):
        plt.plot(est, 'x', label=name.to_string()+'_q_est')
    plt.gca().set_prop_cycle(None)
    for tru, name in zip(q_sol, biorbd_model.nameDof()):
        plt.plot(tru, label=name.to_string()+'_q_tru')
    plt.legend()

    plt.subplot(212)
    for est, name in zip(X_est[biorbd_model.nbQ():, :], biorbd_model.nameDof()):
        plt.plot(est, 'x', label=name.to_string()+'_qdot_est')
    plt.gca().set_prop_cycle(None)
    for tru, name in zip(dq_sol, biorbd_model.nameDof()):
        plt.plot(tru, label=name.to_string()+'_qdot_tru')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    if use_torque:
        plt.subplot(211)
        for est, name in zip(U_est[:nbGT, :], biorbd_model.nameDof()):
            plt.plot(est, 'x', label=name.to_string()+'_tau_est')
        plt.gca().set_prop_cycle(None)
        for tru, name in zip(tau, biorbd_model.nameDof()):
            plt.plot(tru, label=name.to_string()+'_tau_tru')
        plt.legend()
        plt.subplot(212)

    for est, name in zip(U_est[nbGT:, :], biorbd_model.muscleNames()):
        plt.plot(est, 'x', label=name.to_string()+'_est')
    plt.gca().set_prop_cycle(None)
    for tru, name in zip(u_sol, biorbd_model.muscleNames()):
        plt.plot(tru, label=name.to_string()+'_tru')
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.show()
    print()
