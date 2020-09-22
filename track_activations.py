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

def warm_start_mhe(data_sol_prev):
    # TODO: This should be moved in a MHE module
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    X0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    U0 = u[:, 1:]  # discard oldest estimate of the window
    X_out = x[:, 0]
    return X0, U0, X_out

def markers_fun(biorbd_model):
    qMX = MX.sym('qMX', biorbd_model.nbQ())
    return Function('markers', [qMX], [biorbd_model.Markers(qMX)])

def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, use_SX=False, nb_threads=1):
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
    x_init = InitialConditionsOption(np.zeros(number_shooting_points+1, nbQ),
                                     interpolation=InterpolationType.EACH_FRAME)

    u0 = np.array([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
    u_init = InitialConditionsOption(np.zeros(number_shooting_points+1, nbGT + nbMT),
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

    ocp, sol = OptimalControlProgram.load(f"sim_ip_800ms_80sn_ext.bo")
    T = ocp.nlp[0].Tf
    Ns = ocp.nlp[0].Ns
    model = ocp.nlp[0].Model
    data_sol = Data.get_data(ocp, sol)
    q_sol = data_sol[0]['q']
    dq_sol = data_sol[0]['q_dot']
    x_0 = np.vstack([q_sol[:, 0], dq_sol[:, 0]])

    # get targets
    get_markers = markers_fun(model)
    markers_target = np.zeros((model.nbMarkers(), Ns+1))
    for i in range(Ns+1):
        markers_target[: , i] = get_markers(q_sol[:, i])
    muscles_target = data_sol[1]['muscle']


    ocp = prepare_ocp(biorbd_model_path="arm_Belaise.bioMod", final_time=T, number_shooting_points=Ns, use_SX=True)

    ocp.nlp[0].X_bounds.min[:, 0] = x_0
    ocp.nlp[0].X_bounds.max[:, 0] = x_0

    objectives = ObjectiveList()
    objectives.add(Objective.Lagrange.MINIMIZE_MARKERS, weight=1000, target=Y_i, idx=0)
    objectives.add(Objective.Lagrange.MINIMIZE_STATE, weight=100, target=X0, phase=0, idx=1)

    ocp.update_objectives(new_objectives)

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
    ocp.save(sol, f"solutions/sim_ip_{T*1000}ms_{Ns}sn_ext.bo")

    if use_BO:
        ocp, sol = OptimalControlProgram.load(f"sim_ip_{T*1000}ms_{Ns}sn_ext.bo")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
