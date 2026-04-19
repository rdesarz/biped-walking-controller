"""
Whole-Body QP Controller for Biped Walking using TSID + Pinocchio.

This example shows how to replace differential inverse kinematics with a
torque-level whole-body QP controller. The pipeline becomes:

    ZMP Preview Controller -> COM/foot reference trajectories
                           -> TSID (this code)
                           -> joint torques

Decision variables: joint accelerations (q_ddot) and contact forces (f_c)
Equality constraint: M * q_ddot + h = S^T * tau + J_c^T * f_c
Inequality constraints: friction cones, torque limits
Cost: weighted COM tracking + foot tracking + posture regularization

Dependencies:
    pip install tsid pin numpy example-robot-data
    # or: conda install tsid pinocchio example-robot-data -c conda-forge
"""

import argparse
from pathlib import Path

import numpy as np
import tsid
import pinocchio as pin


def set_joint(q, model, joint_name, val):
    jid = model.getJointId(joint_name)
    if jid > 0 and model.joints[jid].nq == 1:
        q[model.joints[jid].idx_q] = val


def make_initial_configuration(model):
    q = pin.neutral(model)

    # Talos standing pose. Missing joints are ignored so this still works with
    # another URDF, where the neutral configuration is kept for unknown joints.
    set_joint(q, model, "leg_left_1_joint", 0.0)
    set_joint(q, model, "leg_left_2_joint", 0.0)
    set_joint(q, model, "leg_left_3_joint", -0.5)
    set_joint(q, model, "leg_left_4_joint", 1.0)
    set_joint(q, model, "leg_left_5_joint", -0.6)
    set_joint(q, model, "leg_right_1_joint", 0.0)
    set_joint(q, model, "leg_right_2_joint", 0.0)
    set_joint(q, model, "leg_right_3_joint", -0.5)
    set_joint(q, model, "leg_right_4_joint", 1.0)
    set_joint(q, model, "leg_right_5_joint", -0.6)
    set_joint(q, model, "arm_right_4_joint", -1.5)
    set_joint(q, model, "arm_left_4_joint", -1.5)

    return q


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/home/rdesarz/projects"),
        help="Folder used as the package root for meshes and relative URDF paths.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("talos_data/urdf/talos_full.urdf"),
        help="URDF path. Relative paths are resolved from --model-root.",
    )
    return parser.parse_args()


# ============================================================================
# 1. ROBOT SETUP
# ============================================================================

args = parse_args()
model_root = args.model_root.expanduser().resolve()
urdf_path = args.urdf.expanduser()
if not urdf_path.is_absolute():
    urdf_path = model_root / urdf_path
urdf_path = urdf_path.resolve()

if not urdf_path.is_file():
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")

# Load a humanoid URDF (Talos by default, or pass --urdf for yours).
full_model, full_col_model, full_vis_model = pin.buildModelsFromUrdf(
    str(urdf_path), str(model_root), pin.JointModelFreeFlyer()
)

# TSID RobotWrapper: loads URDF with a free-flyer joint (floating base)
robot = tsid.RobotWrapper(str(urdf_path), [str(model_root)], pin.JointModelFreeFlyer(), False)
model = robot.model()
data = robot.data()

# Frame names for Talos (adapt these to your URDF)
LEFT_FOOT_FRAME = "left_sole_link"
RIGHT_FOOT_FRAME = "right_sole_link"

# Initial configuration: half-sitting / standing pose
# Use a Talos standing pose when these joint names exist, otherwise neutral.
q0 = make_initial_configuration(model)
v0 = np.zeros(robot.nv)

# ============================================================================
# 2. SOLVER SETUP
# ============================================================================

dt = 1e-3  # Control timestep (1 kHz)

# Create the inverse dynamics QP formulation
# This sets up: min ||tasks||^2 s.t. dynamics + contacts
invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
invdyn.computeProblemData(0.0, q0, v0)

# ============================================================================
# 3. CONTACT SETUP (both feet on ground in double support)
# ============================================================================

# Contact parameters
mu = 0.5            # Friction coefficient
f_min = 1.0         # Minimum normal force (N) -- prevents "pulling" on ground
f_max = 1000.0      # Maximum normal force (N)
contact_normal = np.array([0.0, 0.0, 1.0])  # Ground normal

# Contact surface corners (rectangular foot, in local foot frame)
# Adapt these dimensions to your robot's foot geometry
lx = 0.10  # half-length along x
ly = 0.05  # half-width along y
contact_points = np.array([
    [ lx,  ly, 0.0],
    [ lx, -ly, 0.0],
    [-lx, -ly, 0.0],
    [-lx,  ly, 0.0],
]).T  # Shape: (3, 4)

# Gains for contact stabilization (keeps feet from drifting)
kp_contact = 30.0
kd_contact = 2.0 * np.sqrt(kp_contact)

# Weight for contact force regularization in the cost
w_force_reg = 1e-5

# --- Left foot contact ---
contact_left = tsid.Contact6d(
    "contact_left", robot, LEFT_FOOT_FRAME,
    contact_points, contact_normal, mu, f_min, f_max
)
contact_left.setKp(kp_contact * np.ones(6))
contact_left.setKd(kd_contact * np.ones(6))
H_left_ref = robot.framePosition(invdyn.data(), model.getFrameId(LEFT_FOOT_FRAME))
contact_left.setReference(H_left_ref)
invdyn.addRigidContact(contact_left, w_force_reg, 1.0, 1)

# --- Right foot contact ---
contact_right = tsid.Contact6d(
    "contact_right", robot, RIGHT_FOOT_FRAME,
    contact_points, contact_normal, mu, f_min, f_max
)
contact_right.setKp(kp_contact * np.ones(6))
contact_right.setKd(kd_contact * np.ones(6))
H_right_ref = robot.framePosition(invdyn.data(), model.getFrameId(RIGHT_FOOT_FRAME))
contact_right.setReference(H_right_ref)
invdyn.addRigidContact(contact_right, w_force_reg, 1.0, 1)

# ============================================================================
# 4. TASK DEFINITIONS
# ============================================================================

# --- COM tracking task ---
# This is where your ZMP preview controller output goes
kp_com = 30.0
kd_com = 2.0 * np.sqrt(kp_com)
w_com = 1.0  # Weight in the cost

com_task = tsid.TaskComEquality("task-com", robot)
com_task.setKp(kp_com * np.ones(3))
com_task.setKd(kd_com * np.ones(3))
invdyn.addMotionTask(com_task, w_com, 1, 0.0)

# Initial COM reference (will be overwritten by your trajectory generator)
com_ref = robot.com(invdyn.data())
com_traj = tsid.TrajectoryEuclidianConstant("traj-com", com_ref)

# --- Swing foot tracking task (SE3) ---
# Only active during single support; add/remove dynamically
kp_foot = 100.0
kd_foot = 2.0 * np.sqrt(kp_foot)
w_foot = 1.0

swing_foot_task = tsid.TaskSE3Equality("task-swing-foot", robot, RIGHT_FOOT_FRAME)
swing_foot_task.setKp(kp_foot * np.ones(6))
swing_foot_task.setKd(kd_foot * np.ones(6))
# Mask: track all 6 DoF (position + orientation). Set mask[3:6]=0 to ignore rotation.
swing_foot_task.setMask(np.ones(6))
# NOTE: Don't add this task yet -- it's only active during single support.

# --- Posture regularization task ---
# Prevents the QP from producing wild joint motions
kp_posture = 1.0
kd_posture = 2.0 * np.sqrt(kp_posture)
w_posture = 0.1  # Low weight: posture is a soft objective

posture_task = tsid.TaskJointPosture("task-posture", robot)
posture_task.setKp(kp_posture * np.ones(robot.nv - 6))  # -6 for free-flyer
posture_task.setKd(kd_posture * np.ones(robot.nv - 6))
invdyn.addMotionTask(posture_task, w_posture, 1, 0.0)

# Posture reference: the standing configuration (minus free-flyer)
q_posture_ref = q0[7:]  # Remove 7 DoF free-flyer (pos + quat)
sample_posture = tsid.TrajectorySample(robot.nv - 6)
sample_posture.pos(q_posture_ref)

# ============================================================================
# 5. QP SOLVER
# ============================================================================

solver = tsid.SolverHQuadProgFast("solver-qp")
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)

# ============================================================================
# 6. SIMULATION LOOP
# ============================================================================

N_STEPS = 5000  # 5 seconds at 1 kHz
q = q0.copy()
v = v0.copy()

# Storage for logging
com_log = np.zeros((N_STEPS, 3))
tau_log = []

for i in range(N_STEPS):
    t = i * dt

    # ------------------------------------------------------------------
    # 6a. SET REFERENCES FROM YOUR TRAJECTORY GENERATOR
    # ------------------------------------------------------------------
    # Replace these with your actual ZMP preview controller outputs:
    #   com_des[t], com_vel_des[t], com_acc_des[t]
    #   swing_foot_pose_des[t], swing_foot_vel_des[t], swing_foot_acc_des[t]

    # Example: constant COM reference (just standing)
    sample_com = com_traj.computeNext()
    com_task.setReference(sample_com)

    # Example: constant posture reference
    posture_task.setReference(sample_posture)

    # ------------------------------------------------------------------
    # 6b. CONTACT SWITCHING (for walking)
    # ------------------------------------------------------------------
    # During single support, you need to:
    #   1. Remove the contact for the swing foot
    #   2. Add the swing foot tracking task
    #
    # Example (not active in this standing demo):
    #
    #   # Entering single support (right foot swings):
    #   invdyn.removeRigidContact(contact_right.name, transition_duration=0.0)
    #   invdyn.addMotionTask(swing_foot_task, w_foot, 1, 0.0)
    #   swing_foot_task.setReference(desired_foot_SE3_sample)
    #
    #   # Entering double support (right foot lands):
    #   invdyn.removeTask(swing_foot_task.name, transition_duration=0.0)
    #   H_right_new = <desired landing pose>
    #   contact_right.setReference(H_right_new)
    #   invdyn.addRigidContact(contact_right, w_force_reg, 1.0, 1)

    # ------------------------------------------------------------------
    # 6c. SOLVE THE QP
    # ------------------------------------------------------------------
    HQP = invdyn.computeProblemData(t, q, v)
    sol = solver.solve(HQP)

    if sol.status != 0:
        print(f"[t={t:.3f}] QP solver failed with status {sol.status}")
        break

    # Extract joint torques from the solution
    tau = invdyn.getActuatorForces(sol)

    # Extract joint accelerations
    dv = invdyn.getAccelerations(sol)

    # ------------------------------------------------------------------
    # 6d. INTEGRATE (simple Euler -- replace with your simulator)
    # ------------------------------------------------------------------
    # In practice, send `tau` to your simulator (PyBullet, MuJoCo, etc.)
    # and read back q, v. Here we do a simple forward integration.
    v_next = v + dv * dt
    q_next = pin.integrate(model, q, v_next * dt)

    q = q_next
    v = v_next

    # ------------------------------------------------------------------
    # 6e. LOG
    # ------------------------------------------------------------------
    com_log[i] = robot.com(invdyn.data())
    tau_log.append(tau.copy())

    if i % 1000 == 0:
        com_pos = robot.com(invdyn.data())
        print(f"[t={t:.3f}] COM: [{com_pos[0]:.4f}, {com_pos[1]:.4f}, {com_pos[2]:.4f}]")


print("\nDone. Final COM:", robot.com(invdyn.data()))
print(f"Torque vector dimension: {tau.shape[0]} (actuated joints)")

# ============================================================================
# 7. HELPER: HOW TO SET SE3 REFERENCES FOR SWING FOOT
# ============================================================================

def make_se3_sample(position, orientation_matrix, velocity=None, acceleration=None):
    """
    Build a TrajectorySample for TaskSE3Equality from position + rotation.

    TSID expects a 12-dim pos vector: [translation(3), rotation_matrix_flat(9)]
    and 6-dim vel/acc vectors: [linear(3), angular(3)]
    """
    sample = tsid.TrajectorySample(12, 6)

    pos = np.zeros(12)
    pos[:3] = position
    pos[3:] = orientation_matrix.flatten()  # Column-major
    sample.pos(pos)

    if velocity is not None:
        sample.vel(velocity)  # 6D: [v_linear, omega]
    if acceleration is not None:
        sample.acc(acceleration)

    return sample


# Example usage for a swing foot trajectory point:
#
#   foot_pos = np.array([0.0, -0.1, 0.05])  # Foot lifted 5cm
#   foot_rot = np.eye(3)                     # Flat orientation
#   foot_vel = np.zeros(6)
#   foot_acc = np.zeros(6)
#   sample = make_se3_sample(foot_pos, foot_rot, foot_vel, foot_acc)
