from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix # pylint: disable=import-error, no-name-in-module
from pydrake.multibody.inverse_kinematics import InverseKinematics # pylint: disable=import-error, no-name-in-module

from arm_visualization import create_arm, arm_components_loader

from pydrake.solvers import ( # pylint: disable=import-error, no-name-in-module
    Solve,
)

def compute_inverse_kinematics(q0: Sequence, 
                               endeffector_pose: RigidTransform, 
                               use_rohan_scenario:bool = False, 
                               use_cheap:bool = True

) -> Optional[Sequence]:
    """Return the joint angles for the given end-effector pose.

    Args:
        q0: Initial guess for the joint angles.
        endeffector_pose: The desired pose of the wsg gripper in the world
            frame.

    Returns:
        The joint angles for the given end-effector pose, if a solution
        exist.
    """

    arm_components = arm_components_loader(use_rohan_scenario=use_rohan_scenario, use_cheap=use_cheap, use_meshcat=False,)
    plant = arm_components.plant
    
    plant_context = plant.CreateDefaultContext()
    gripper_frame = plant.GetBodyByName("body").body_frame()
    ik = InverseKinematics(plant, plant_context)  # type: ignore
    ik.AddPositionConstraint(
        gripper_frame,
        [0, 0, 0],
        plant.world_frame(),
        endeffector_pose.translation(),
        endeffector_pose.translation(),
    )

    ik.AddOrientationConstraint(
        gripper_frame,
        RotationMatrix(),
        plant.world_frame(),
        endeffector_pose.rotation(),
        1e-3,
    )

    prog = ik.get_mutable_prog()
    q = ik.q()

    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    result = Solve(ik.prog())
    if not result.is_success():
        print("IK failed")
        return None
    return result.GetSolution(q)