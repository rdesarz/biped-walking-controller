import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pybullet as pb
import tsid

from biped_walking_controller.model import Talos, q_from_base_and_joints
from biped_walking_controller.simulation import (
    Simulator,
    _compute_base_from_foot_target,
    _snap_feet_to_plane,
)


@dataclass
class GeneralParams:
    dt: float = 1.0 / 500.0
    duration: float = 100.0
    n_solver_iter: int = 1000


def _parse_xyz(text: str | None) -> np.ndarray:
    if text is None:
        return np.zeros(3)
    return np.array([float(v) for v in text.split()], dtype=float)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

    return rot_z @ rot_y @ rot_x


def _origin_to_transform(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if origin is None:
        return transform

    transform[:3, :3] = _rpy_to_matrix(_parse_xyz(origin.attrib.get("rpy")))
    transform[:3, 3] = _parse_xyz(origin.attrib.get("xyz"))
    return transform


def _find_link(root: ET.Element, link_name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.attrib.get("name") == link_name:
            return link
    raise ValueError(f"Link '{link_name}' not found in URDF")


def _find_parent_link_and_transform(root: ET.Element, child_link_name: str):
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or child.attrib.get("link") != child_link_name:
            continue

        parent = joint.find("parent")
        if parent is None:
            break

        return parent.attrib["link"], _origin_to_transform(joint.find("origin"))

    return child_link_name, np.eye(4)


def contact_points_from_urdf_box(urdf_path: Path, contact_frame_name: str) -> np.ndarray:
    """
    Return a rectangular foot contact patch in the requested contact frame.

    TALOS stores the collision box on ``leg_*_6_link`` and attaches the
    ``*_sole_link`` contact frame with a fixed joint. TSID contacts are defined
    in the contact frame, so the box is transformed before extracting its XY
    footprint.
    """
    root = ET.parse(urdf_path).getroot()
    collision_link_name, parent_to_contact = _find_parent_link_and_transform(
        root, contact_frame_name
    )
    collision_link = _find_link(root, collision_link_name)
    contact_to_parent = np.linalg.inv(parent_to_contact)

    best_points = None
    best_area = -np.inf

    for collision in collision_link.findall("collision"):
        box = collision.find("geometry/box")
        if box is None:
            continue

        size = _parse_xyz(box.attrib["size"])
        half_size = 0.5 * size
        parent_to_box = _origin_to_transform(collision.find("origin"))

        local_corners = np.array(
            [
                [sx * half_size[0], sy * half_size[1], sz * half_size[2], 1.0]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        ).T
        contact_corners = (contact_to_parent @ parent_to_box @ local_corners)[:3].T

        min_xy = contact_corners[:, :2].min(axis=0)
        max_xy = contact_corners[:, :2].max(axis=0)
        area = np.prod(max_xy - min_xy)
        if area <= best_area:
            continue

        best_area = area
        best_points = np.array(
            [
                [max_xy[0], max_xy[1], 0.0],
                [max_xy[0], min_xy[1], 0.0],
                [min_xy[0], min_xy[1], 0.0],
                [min_xy[0], max_xy[1], 0.0],
            ]
        ).T

    if best_points is None:
        raise ValueError(
            f"No collision box found for contact frame '{contact_frame_name}' "
            f"through link '{collision_link_name}'"
        )

    return best_points


def _contact_patch_summary(contact_points: np.ndarray) -> str:
    x_min, x_max = contact_points[0].min(), contact_points[0].max()
    y_min, y_max = contact_points[1].min(), contact_points[1].max()
    return (
        f"x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}] "
        f"(half extents {0.5 * (x_max - x_min):.3f}, {0.5 * (y_max - y_min):.3f})"
    )


def posture_gains_from_model(model: pin.Model) -> np.ndarray:
    gains = np.ones(model.nv - 6)
    leg_gains = {
        "1": 10.0,
        "2": 5.0,
        "3": 5.0,
        "4": 1.0,
        "5": 1.0,
        "6": 10.0,
    }

    for joint_id in range(1, model.njoints):
        joint = model.joints[joint_id]
        if joint.nv != 1 or joint.idx_v < 6:
            continue

        name = model.names[joint_id]
        gain = 1.0
        if name.startswith(("leg_left_", "leg_right_")):
            joint_number = name.rsplit("_", 2)[1]
            gain = leg_gains.get(joint_number, 5.0)
        elif name.startswith("torso_"):
            gain = 500.0
        elif name.startswith(("arm_left_", "arm_right_")):
            gain = 50.0 if name.endswith("_1_joint") else 10.0
        elif name.startswith("head_"):
            gain = 100.0
        elif name.startswith("gripper_"):
            gain = 1.0

        gains[joint.idx_v - 6] = gain

    return gains


class TSIDController:
    def __init__(
        self,
        urdf_path: Path,
        package_root: Path,
        q0: np.ndarray,
        left_foot_frame: str,
        right_foot_frame: str,
        left_contact_points: np.ndarray,
        right_contact_points: np.ndarray,
    ):
        self.robot = tsid.RobotWrapper(
            str(urdf_path), [str(package_root)], pin.JointModelFreeFlyer(), False
        )
        self.model = self.robot.model()
        self.v0 = np.zeros(self.robot.nv)
        self.zero_com_derivatives = np.zeros(3)
        self.com_sample = tsid.TrajectorySample(3)

        self.left_foot_frame = left_foot_frame
        self.right_foot_frame = right_foot_frame
        self.left_contact_points = left_contact_points
        self.right_contact_points = right_contact_points
        self.left_foot_id = self._get_frame_id(left_foot_frame)
        self.right_foot_id = self._get_frame_id(right_foot_frame)

        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
        self.invdyn.computeProblemData(0.0, q0, self.v0)
        self.data = self.invdyn.data()

        self._add_foot_contacts()
        self._add_com_task()
        self._add_posture_task(q0)

        # Use EiquadprogFast: dynamic matrix sizes (memory allocation performed only when resizing)
        self.solver = tsid.SolverHQuadProgFast("solver-qp")

        # Resize the solver to fit the number of variables, equality and inequality constraints
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    def _get_frame_id(self, frame_name: str) -> int:
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{frame_name}' does not exist in the TSID model")
        return frame_id

    def _add_foot_contacts(self):
        mu = 0.3
        f_min = 1.0
        f_max = 1000.0
        contact_normal = np.array([0.0, 0.0, 1.0])

        kp_contact = 30.0
        kd_contact = 2.0 * math.sqrt(kp_contact)

        # Weight for contact force regularization in the cost
        w_force_reg = 1e-5

        # Left foot
        self.contact_left = tsid.Contact6d(
            "contact-left",
            self.robot,
            self.left_foot_frame,
            self.left_contact_points,
            contact_normal,
            mu,
            f_min,
            f_max,
        )
        self.contact_left.setKp(kp_contact * np.ones(6))
        self.contact_left.setKd(kd_contact * np.ones(6))
        self.contact_left.setReference(self.robot.framePosition(self.data, self.left_foot_id))
        self.invdyn.addRigidContact(self.contact_left, w_force_reg, 1.0, 0)

        self.contact_right = tsid.Contact6d(
            "contact-right",
            self.robot,
            self.right_foot_frame,
            self.right_contact_points,
            contact_normal,
            mu,
            f_min,
            f_max,
        )
        self.contact_right.setKp(kp_contact * np.ones(6))
        self.contact_right.setKd(kd_contact * np.ones(6))
        self.contact_right.setReference(self.robot.framePosition(self.data, self.right_foot_id))
        self.invdyn.addRigidContact(self.contact_right, w_force_reg, 1.0, 0)

    def _add_com_task(self):
        kp_com = 20.0
        kd_com = 2.0 * math.sqrt(kp_com)
        w_com = 1.0

        self.com_task = tsid.TaskComEquality("task-com", self.robot)
        self.com_task.setKp(kp_com * np.ones(3))
        self.com_task.setKd(kd_com * np.ones(3))
        self.invdyn.addMotionTask(self.com_task, w_com, 1, 0.0)

        com_ref = self.data.com[0]  # Initial value of the CoM
        self.traj_com = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)

    def _add_posture_task(self, q0: np.ndarray):
        kp_posture = posture_gains_from_model(self.model)
        w_posture = 0.1

        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)
        self.posture_task.setKp(kp_posture)
        self.posture_task.setKd(2.0 * np.sqrt(kp_posture))
        self.invdyn.addMotionTask(self.posture_task, w_posture, 1, 0.0)

        self.traj_posture = tsid.TrajectoryEuclidianConstant("traj_joint", q0[7:])

    def compute(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        com_ref: np.ndarray,
    ) -> np.ndarray:
        self.com_sample.value(com_ref)
        self.com_sample.derivative(self.zero_com_derivatives)
        self.com_sample.second_derivative(self.zero_com_derivatives)
        self.com_task.setReference(self.com_sample)

        posture_ref = self.traj_posture.computeNext()
        self.posture_task.setReference(posture_ref)

        hqp = self.invdyn.computeProblemData(t, q, v)
        sol = self.solver.solve(hqp)
        if sol.status != 0:
            raise RuntimeError(f"TSID QP solver failed at t={t:.3f} with status {sol.status}")
        
        tau = self.invdyn.getActuatorForces(sol)
        contact_forces = self.invdyn.getContactForces(sol)

        print("solver status:", sol.status)
        print("tau max:", np.max(np.abs(tau)))
        print("com:", pin.centerOfMass(self.model, self.data, q))
        print("com err:", pin.centerOfMass(self.model, self.data, q) - com_ref)

        for name in ["contact-left", "contact-right"]:
            if self.invdyn.checkContact(name, sol):
                f = self.invdyn.getContactForce(name, sol)
                print(name, "fz:", f[2], "force norm:", np.linalg.norm(f[:3]))

        return tau


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-talos-data",
        type=Path,
        default=Path("/home/rdesarz/projects"),
        help="Path containing the talos_data folder.",
    )
    parser.add_argument("--plot-results", action="store_true")
    parser.add_argument("--launch-gui", action="store_false")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--duration", type=float, default=GeneralParams.duration)
    return parser.parse_args()


def main():
    args = parse_args()
    params = GeneralParams(duration=args.duration)

    np.set_printoptions(suppress=True, precision=3)

    package_root = args.path_talos_data.expanduser().resolve()
    urdf_path = package_root / "talos_data" / "urdf" / "talos_full.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    talos = Talos(path_to_model=package_root, reduced=False)
    q_init = talos.set_and_get_default_pose()

    simulator = Simulator(
        dt=params.dt,
        path_to_robot_urdf=urdf_path,
        model=talos,
        launch_gui=args.launch_gui,
        n_solver_iter=params.n_solver_iter,
    )

    oMf_rf0 = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf0 = talos.data.oMf[talos.left_foot_id].copy()
    oMf_lf_tgt, oMf_rf_tgt = _snap_feet_to_plane(oMf_lf0, oMf_rf0)

    oMb_init = _compute_base_from_foot_target(
        talos.model, talos.data, q_init, talos.left_foot_id, oMf_lf_tgt
    )
    q_init = q_from_base_and_joints(q_init, oMb_init)

    pin.forwardKinematics(talos.model, talos.data, q_init)
    pin.updateFramePlacements(talos.model, talos.data)
    pin.centerOfMass(talos.model, talos.data, q_init)

    simulator.reset_robot_configuration(q_init)
    simulator.disable_joint_motors()

    left_contact_points = contact_points_from_urdf_box(urdf_path, "left_sole_link")
    right_contact_points = contact_points_from_urdf_box(urdf_path, "right_sole_link")
    print(f"Left TSID contact patch from URDF: {_contact_patch_summary(left_contact_points)}")
    print(f"Right TSID contact patch from URDF: {_contact_patch_summary(right_contact_points)}")

    controller = TSIDController(
        urdf_path=urdf_path,
        package_root=package_root,
        q0=q_init,
        left_foot_frame="left_sole_link",
        right_foot_frame="right_sole_link",
        left_contact_points=left_contact_points,
        right_contact_points=right_contact_points,
    )

    com0 = pin.centerOfMass(talos.model, talos.data, q_init)
    feet_mid = 0.5 * (oMf_lf_tgt.translation + oMf_rf_tgt.translation)
    # com_ref = np.array([feet_mid[0], feet_mid[1], com0[2]])

    com_ref = com0.copy()
    print(f"Fixed COM reference: [{com_ref[0]:.3f}, {com_ref[1]:.3f}, {com_ref[2]:.3f}]")

    pb.setGravity(0, 0, 0)
    for _ in range(10):
        simulator.step()
    pb.setGravity(0, 0, -9.81)

    n_steps = int(math.ceil(params.duration / params.dt))
    time = np.arange(n_steps) * params.dt
    com_ref_log = np.zeros((n_steps, 3))
    com_pin_log = np.zeros((n_steps, 3))
    com_pb_log = np.zeros((n_steps, 3))
    tau_max = np.zeros(n_steps)

    for k, t in enumerate(time):
        q = simulator.get_q(talos.model.nq)
        v = simulator.get_v(talos.model.nv)

        try:
            tau = controller.compute(t, q, v, com_ref)

            
        except RuntimeError as exc:
            print(exc)
            break

        simulator.apply_joint_torques(tau)
        # simulator.reset_robot_configuration(q_init)
        # simulator.update_camera_to_follow_pos(com_ref[0], com_ref[1], 0.0)
        simulator.step()

        pin.forwardKinematics(talos.model, talos.data, q)
        pin.updateFramePlacements(talos.model, talos.data)

        com_ref_log[k] = com_ref
        com_pin_log[k] = pin.centerOfMass(talos.model, talos.data, q)
        com_pb_log[k] = simulator.get_robot_com_position()
        tau_max[k] = np.max(np.abs(tau))

        if k % max(1, int(round(1.0 / params.dt))) == 0:
            print(
                f"[t={t:.2f}] com_ref_y={com_ref[1]:.3f} "
                f"com_pb_y={com_pb_log[k, 1]:.3f} tau_max={tau_max[k]:.1f}"
            )

    if pb.isConnected(simulator.cid):
        pb.disconnect(simulator.cid)


if __name__ == "__main__":
    main()
