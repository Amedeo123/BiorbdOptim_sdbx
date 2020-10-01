import biorbd
from time import time
import numpy as np
from BiorbdViz import BiorbdViz
import os.path

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
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


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, x0, xT, use_SX=False, nb_threads=1):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nbQ = biorbd_model.nbQ()

    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10000, states_idx=np.array(range(0, nbQ)))
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10000, states_idx=np.array(range(nbQ, nbQ*2)))
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, weight=1000000,
                            target=np.tile(xT, (number_shooting_points+1, 1)).T, states_idx=np.array(range(0, nbQ)))

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # State path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = x0

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

    u0 = np.array([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
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

    T = 1.0
    Ns = 100
    motion = 'EXT2'  # 'EXT', 'REACH'
    if motion == 'EXT':
        x0 = np.array([-1, 1, 1, 1, 0, 0, 0, 0])
        xT = np.array([-1, 1, 1, 0.1, 0, 0, 0, 0])
    if motion == 'EXT2':
        x0 = np.array([-2., 1.5, 2.5, 1.3, 0., 0., 0., 0.])
        xT = np.array([-2., 0.8, 2.5, -0.2, 0., 0., 0., 0.])
    if motion == 'REACH':
        x0 = np.array([0., -0.2, 0, 0, 0, 0, 0, 0])
        xT = np.array([0.6, -1, 0, 0.5, 0, 0, 0, 0])
    use_ACADOS = False
    use_IPOPT = False
    use_BO = True

    if use_IPOPT:
        ocp = prepare_ocp(biorbd_model_path="arm_wt_rot_scap.bioMod", final_time=T, number_shooting_points=Ns,
                          x0=x0, xT=xT, use_SX=False, nb_threads=6)

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
        if os.path.isfile(f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}.bo"):
            ocp.save(sol, f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}_1.bo")
        else:
            ocp.save(sol, f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}.bo")

    if use_BO:
        ocp, sol = OptimalControlProgram.load(f"solutions/sim_ip_{int(T*1000)}ms_{Ns}sn_{motion}.bo")

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
