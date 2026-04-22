import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pybullet as pb
import tsid
from matplotlib import pyplot as plt

from biped_walking_controller.model import Talos, q_from_base_and_joints
from biped_walking_controller.simulation import (
    Simulator,
    _compute_base_from_foot_target,
    _snap_feet_to_plane,
)


@dataclass
class GeneralParams:
    dt: float = 1.0 / 500.0
    duration: float = 8.0
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

    xyz = _parse_xyz(origin.attrib.get("xyz"))
    rpy = _parse_xyz(origin.attrib.get("rpy"))
    transform[:3, :3] = _rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
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
    Read the rectangular foot contact patch from a URDF collision box.

    Talos stores the foot collision box on ``leg_*_6_link`` and attaches the
    ``*_sole_link`` contact frame with a fixed joint. This function transforms
    that box into the sole frame and returns its XY footprint at z=0.
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


class TSIDFixedComController:
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

        self.left_foot_frame = left_foot_frame
        self.right_foot_frame = right_foot_frame
        self.left_contact_points = left_contact_points
        self.right_contact_points = right_contact_points
        self.left_foot_id = self._get_frame_id(left_foot_frame)
        self.right_foot_id = self._get_frame_id(right_foot_frame)

        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid-fixed-com", self.robot, False)
        self.invdyn.computeProblemData(0.0, q0, self.v0)

        self._add_foot_contacts()
        self._add_com_task()
        self._add_posture_task(q0)

        self.solver = tsid.SolverHQuadProgFast("solver-qp")
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    def _get_frame_id(self, frame_name: str) -> int:
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"Frame '{frame_name}' does not exist in the TSID model")
        return frame_id

    def _add_foot_contacts(self):
        mu = 0.5
        f_min = 1.0
        f_max = 2000.0
        contact_normal = np.array([0.0, 0.0, 1.0])

        kp_contact = 50.0
        kd_contact = 2.0 * math.sqrt(kp_contact)
        w_force_reg = 1e-5

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
        self.contact_left.setReference(
            self.robot.framePosition(self.invdyn.data(), self.left_foot_id)
        )
        self.invdyn.addRigidContact(self.contact_left, w_force_reg, 1.0, 1)

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
        self.contact_right.setReference(
            self.robot.framePosition(self.invdyn.data(), self.right_foot_id)
        )
        self.invdyn.addRigidContact(self.contact_right, w_force_reg, 1.0, 1)

    def _add_com_task(self):
        kp_com = 20.0
        kd_com = 2.0 * math.sqrt(kp_com)
        w_com = 1.0

        self.com_task = tsid.TaskComEquality("task-com", self.robot)
        self.com_task.setKp(kp_com * np.ones(3))
        self.com_task.setKd(kd_com * np.ones(3))
        self.invdyn.addMotionTask(self.com_task, w_com, 1, 0.0)

    def _add_waist_task(self):
        kp_waist = 500.0
        kd_waist = 2.0 * math.sqrt(kp_com)
        w_waist = 1.0

        self.waist_task = tsid.TaskSE3Equality("task-waist", self.robot, "root_joint")
        self.waist_task.setKp(kp_waist * np.ones(6))
        self.waist_task.setKd(kd_waist * np.ones(3))
        self.invdyn.addMotionTask(self.waist_task, w_waist, 1, 0.0)

    def _add_posture_task(self, q0: np.ndarray):
        kp_posture = 2.0
        kd_posture = 2.0 * math.sqrt(kp_posture)
        w_posture = 0.05

        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)
        self.posture_task.setKp(kp_posture * np.ones(self.robot.nv - 6))
        self.posture_task.setKd(kd_posture * np.ones(self.robot.nv - 6))
        self.invdyn.addMotionTask(self.posture_task, w_posture, 1, 0.0)

        self.posture_sample = tsid.TrajectorySample(self.robot.nv - 6)
        self.posture_sample.pos(q0[7:])

    def compute(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        com_ref: np.ndarray,
    ) -> np.ndarray:
        com_sample = tsid.TrajectorySample(3)
        com_sample.pos(com_ref)
        com_sample.vel(np.zeros(3))
        com_sample.acc(np.zeros(3))

        self.com_task.setReference(com_sample)
        self.posture_task.setReference(self.posture_sample)

        hqp = self.invdyn.computeProblemData(t, q, v)
        sol = self.solver.solve(hqp)
        if sol.status != 0:
            raise RuntimeError(f"TSID QP solver failed at t={t:.3f} with status {sol.status}")

        return self.invdyn.getActuatorForces(sol)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-talos-data",
        type=Path,
        default=Path("/home/rdesarz/projects"),
        help="Path containing the talos_data folder.",
    )
    parser.add_argument("--plot-results", action="store_true")
    parser.add_argument("--launch-gui", action="store_true")
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

    controller = TSIDFixedComController(
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
    com_ref = np.array([feet_mid[0], feet_mid[1], com0[2]])
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
    rf_forces = np.zeros(n_steps)
    lf_forces = np.zeros(n_steps)
    tau_max = np.zeros(n_steps)

    if args.record_video:
        simulator.start_video_record()

    last_log = -1
    for k, t in enumerate(time):
        q = simulator.get_q(talos.model.nq)
        v = simulator.get_v(talos.model.nv)

        try:
            tau = controller.compute(t, q, v, com_ref)
        except RuntimeError as exc:
            print(exc)
            break

        simulator.apply_joint_torques(tau)
        simulator.update_camera_to_follow_pos(com_ref[0], com_ref[1], 0.0)
        simulator.step()

        pin.forwardKinematics(talos.model, talos.data, q)
        pin.updateFramePlacements(talos.model, talos.data)

        com_ref_log[k] = com_ref
        com_pin_log[k] = pin.centerOfMass(talos.model, talos.data, q)
        com_pb_log[k] = simulator.get_robot_com_position()
        rf_forces[k], lf_forces[k] = simulator.get_contact_forces()
        tau_max[k] = np.max(np.abs(tau))
        last_log = k

        if k % max(1, int(round(1.0 / params.dt))) == 0:
            print(
                f"[t={t:.2f}] com_ref_y={com_ref[1]:.3f} "
                f"com_pb_y={com_pb_log[k, 1]:.3f} tau_max={tau_max[k]:.1f}"
            )

    if args.record_video:
        simulator.stop_video_record(duration=min(params.duration, time[max(last_log, 0)]))

    if args.plot_results and last_log >= 0:
        plot_time = time[: last_log + 1]
        fig, axs = plt.subplots(3, 1, sharex=True)

        axs[0].plot(plot_time, com_ref_log[: last_log + 1, 1], label="COM ref y")
        axs[0].plot(plot_time, com_pb_log[: last_log + 1, 1], label="COM PyBullet y")
        axs[0].plot(plot_time, com_pin_log[: last_log + 1, 1], label="COM Pinocchio y")
        axs[0].set_ylabel("COM y [m]")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(plot_time, rf_forces[: last_log + 1], label="Right foot")
        axs[1].plot(plot_time, lf_forces[: last_log + 1], label="Left foot")
        axs[1].set_ylabel("Normal force [N]")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(plot_time, tau_max[: last_log + 1])
        axs[2].set_xlabel("time [s]")
        axs[2].set_ylabel("max |tau| [Nm]")
        axs[2].grid(True)

        plt.show()


if __name__ == "__main__":
    main()
