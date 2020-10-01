import biorbd
from time import time
import numpy as np
from casadi import MX, Function
import matplotlib.pyplot as plt

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    Data,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsOption,
    ShowResult,
    Solver,
    InterpolationType,
)

def compute_err(Ns_mhe, X_est ,U_est, ocp_ref, sol_ref):
    data_ref = Data.get_data(ocp_ref, sol_ref)
    model = ocp_ref.nlp[0].model
    get_markers = markers_fun(model)
    err = dict()
    Ns = ocp_ref.nlp[0].ns
    norm_err = np.sqrt(Ns-Ns_mhe)
    q_ref = data_ref[0]['q'][:, :-Ns_mhe]
    dq_ref = data_ref[0]['q_dot'][:, :-Ns_mhe]
    tau_ref = data_ref[1]['tau'][:, :-Ns_mhe-1]
    musces_ref = data_ref[1]['muscles'][:, :-Ns_mhe-1]
    sol_mark = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), Ns+1-Ns_mhe))
    err['q'] = np.linalg.norm(X_est[:model.nbQ(), :]-q_ref)/norm_err
    err['q_dot'] = np.linalg.norm(X_est[model.nbQ():, :]-dq_ref)/norm_err
    err['tau'] = np.linalg.norm(U_est[:model.nbGeneralizedTorque(), :]-tau_ref)/norm_err
    err['muscles'] = np.linalg.norm(U_est[model.nbGeneralizedTorque():, :]-musces_ref)/norm_err
    for i in range(Ns+1-Ns_mhe):
        sol_mark[:, :, i] = get_markers(X_est[:model.nbQ(), i])
        sol_mark_ref[:, :, i] = get_markers(data_ref[0]['q'][:, i])
    err['markers'] = np.linalg.norm(sol_mark - sol_mark_ref)/norm_err
    return err


def warm_start_mhe(ocp, sol):
    data = Data.get_data(ocp, sol)
    q = data[0]["q"]
    dq = data[0]["q_dot"]
    tau = data[1]["tau"]
    act = data[1]["muscles"]
    x = np.vstack([q, dq])
    u = np.vstack([tau, act])
    x0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u0 = u[:, 1:]  # discard oldest estimate of the window
    x_out = x[:, 0]
    u_out = u[:, 0]
    return x0, u0, x_out, u_out

def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX], [biorbd_model.markers(qMX)])

def prepare_ocp(biorbd_model_path, final_time, x0, number_shooting_points, use_SX=False, nb_threads=1):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nbQ = biorbd_model.nbQ()
    nbGT = biorbd_model.nbGeneralizedTorque()
    nbMT = biorbd_model.nbMuscleTotal()
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

    # Control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    # Initial guesses
    x_init = InitialConditionsOption(np.tile(x0, (number_shooting_points+1, 1)).T,
                                     interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * nbGT + [muscle_init] * nbMT)+0.1
    u_init = InitialConditionsOption(np.tile(u0, (number_shooting_points, 1)).T,
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

    use_ACADOS = True
    WRITE_STATS = False
    stats_file = 'stats_flex'

    ocp_ref, sol_ref = OptimalControlProgram.load(f"solutions/sim_ip_1000ms_100sn_REACH.bo")
    T = ocp_ref.nlp[0].tf
    Ns = ocp_ref.nlp[0].ns
    model = ocp_ref.nlp[0].model
    data_sol = Data.get_data(ocp_ref, sol_ref)
    q_sol = data_sol[0]['q']
    dq_sol = data_sol[0]['q_dot']
    tau_sol = data_sol[1]['tau']
    muscle_sol = data_sol[1]['muscles']
    x_0 = np.hstack([q_sol[:, 0], dq_sol[:, 0]])

    # get targets
    get_markers = markers_fun(model)
    markers_target = np.zeros((3, model.nbMarkers(), Ns+1))
    for i in range(Ns+1):
        markers_target[:, :, i] = get_markers(q_sol[:, i])
    muscles_target = muscle_sol


    # setup MHE
    Ns_mhe = 6
    T_mhe = T/Ns*Ns_mhe
    X_est = np.zeros((model.nbQ()*2, Ns+1-Ns_mhe))
    U_est = np.zeros((model.nbGeneralizedTorque()+model.nbMuscleTotal(), Ns-Ns_mhe))


    ocp = prepare_ocp(biorbd_model_path="arm_wt_rot_scap.bioMod", final_time=T_mhe, x0=x_0,
                      number_shooting_points=Ns_mhe, use_SX=use_ACADOS)

    # set initial state
    ocp.nlp[0].X_bounds.min[:, 0] = x_0
    ocp.nlp[0].X_bounds.max[:, 0] = x_0

    # set initial guess on state
    ocp.nlp[0].X_init.init = np.tile(x_0, (Ns_mhe+1, 1))

    objectives = ObjectiveList()
    objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000,
                   target=muscles_target[:, :Ns_mhe], idx=0)
    objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000000,
                   target=markers_target[:, :, :Ns_mhe+1], idx=1)
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, idx=2)
    objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, idx=3)
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
        ocp.nlp[0].X_bounds.min[:, 0] = x0[:, 0]
        ocp.nlp[0].X_bounds.max[:, 0] = x0[:, 0]

        # set initial guess on state
        ocp.nlp[0].X_init.init = x0
        ocp.nlp[0].U_init.init = u0

        objectives = ObjectiveList()
        objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000,
                       target=muscles_target[:, i:Ns_mhe+i], idx=0)
        objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000000,
                       target=markers_target[:, :, i:Ns_mhe+i+1], idx=1)
        objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10, idx=2)
        objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, idx=3)
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

    err_offset = 15
    err = compute_err(err_offset, X_est[:, :-err_offset+Ns_mhe], U_est[:, :-err_offset+Ns_mhe], ocp_ref, sol_ref)

    if WRITE_STATS:
        f = open(f"solutions/{stats_file}.txt", "a")
        f.write(f"{Ns_mhe}; {toc/(Ns-Ns_mhe)}; {err['q']}; {err['q_dot']}; {err['tau']}; "
                f"{err['muscles']}; {err['markers']}\n")
        f.close()

    print(err)
    plt.subplot(211)
    plt.plot(X_est[:model.nbQ(), :].T, 'x', label='Q estimate')
    plt.gca().set_prop_cycle(None)
    plt.plot(q_sol.T, label='Q truth')
    plt.legend()
    plt.subplot(212)
    plt.plot(X_est[model.nbQ():, :].T, 'x', label='Qdot estimate')
    plt.gca().set_prop_cycle(None)
    plt.plot(dq_sol.T, label='Qdot truth')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.subplot(211)
    plt.plot(U_est[:model.nbGeneralizedTorque(), :].T, 'x', label='Tau estimate')
    plt.gca().set_prop_cycle(None)
    plt.plot(tau_sol.T, label='Tau truth')
    plt.legend()
    plt.subplot(212)
    plt.plot(U_est[model.nbGeneralizedTorque():, :].T, 'x', label='Muscle activation estimate')
    plt.gca().set_prop_cycle(None)
    plt.plot(muscle_sol.T, label='Muscle activation truth')
    plt.legend()
    plt.tight_layout()
    plt.show()
