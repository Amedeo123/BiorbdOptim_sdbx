import biorbd
from time import time
import numpy as np
from casadi import MX, Function

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    Data,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    InitialConditionsOption,
    ShowResult,
    Solver,
    InterpolationType,
)

def compute_err(ocp ,sol, ocp_ref, sol_ref):
    data = Data.get_data(ocp, sol)
    data_ref = Data.get_data(ocp_ref, sol_ref)
    model = ocp.nlp[0].model
    get_markers = markers_fun(model)
    err = dict()
    sol_mark = np.zeros((3, model.nbMarkers(), ocp.nlp[0].ns+1))
    sol_mark_ref = np.zeros((3, model.nbMarkers(), ocp.nlp[0].ns+1))
    err['q'] = np.linalg.norm(data[0]['q']-data_ref[0]['q'])
    err['q_dot'] = np.linalg.norm(data[0]['q_dot']-data_ref[0]['q_dot'])
    err['tau'] = np.linalg.norm(data[1]['tau']-data_ref[1]['tau'])
    err['muscles'] = np.linalg.norm(data[1]['muscles']-data_ref[1]['muscles'])
    for i in range(ocp.nlp[0].ns+1):
        sol_mark[:, :, i] = get_markers(data[0]['q'][:, i])
        sol_mark_ref[:, :, i] = get_markers(data_ref[0]['q'][:, i])
    err['markers'] = np.linalg.norm(sol_mark - sol_mark_ref)
    return err

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
    ocp_ref, sol_ref = OptimalControlProgram.load(f"solutions/sim_ip_1000ms_100sn_ext_wt_rot_scap.bo")
    T = ocp_ref.nlp[0].tf
    Ns = ocp_ref.nlp[0].ns
    model = ocp_ref.nlp[0].model
    data_sol = Data.get_data(ocp_ref, sol_ref)
    q_sol = data_sol[0]['q']
    dq_sol = data_sol[0]['q_dot']
    x_0 = np.hstack([q_sol[:, 0], dq_sol[:, 0]])

    # get targets
    get_markers = markers_fun(model)
    markers_target = np.zeros((3, model.nbMarkers(), Ns+1))
    for i in range(Ns+1):
        markers_target[:, :, i] = get_markers(q_sol[:, i])
    muscles_target = data_sol[1]['muscles']

    ocp = prepare_ocp(biorbd_model_path="arm_wt_rot_scap.bioMod", final_time=T, x0=x_0,
                      number_shooting_points=Ns, use_SX=use_ACADOS)

    # set initial state
    ocp.nlp[0].X_bounds.min[:, 0] = x_0
    ocp.nlp[0].X_bounds.max[:, 0] = x_0

    # set initial guess on state
    # ocp.nlp[0].X_init.init = np.tile(x_0, (Ns+1, 1))

    objectives = ObjectiveList()
    objectives.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10000, target=muscles_target)
    objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000000, target=markers_target)
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=10)
    objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)
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
                            "sim_method_num_steps": 2,
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

    err = compute_err(ocp, sol, ocp_ref, sol_ref)
    print(err)


    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
