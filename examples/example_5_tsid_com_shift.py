"""
TSID torque-control example: fixed-feet COM transfer in PyBullet.

This is the first inverse-dynamics replacement for the IK loop in
example_4_physics_simulation.py. Both feet stay rigidly constrained on the
ground, and the COM reference moves laterally so the load shifts from one foot
to the other. No foot swing or contact switching is performed yet.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

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
    period: float = 4.0
    settle_duration: float = 1.0
    com_shift_ratio: float = 0.75
    n_solver_iter: int = 1000


class TSIDComShiftController:
    def __init__(
        self,
        urdf_path: Path,
        package_root: Path,
        q0: np.ndarray,
        left_foot_frame: str,
        right_foot_frame: str,
    ):
        self.robot = tsid.RobotWrapper(
            str(urdf_path), [str(package_root)], pin.JointModelFreeFlyer(), False
        )
        self.model = self.robot.model()
        self.v0 = np.zeros(self.robot.nv)

        self.left_foot_frame = left_foot_frame
        self.right_foot_frame = right_foot_frame
        self.left_foot_id = self._get_frame_id(left_foot_frame)
        self.right_foot_id = self._get_frame_id(right_foot_frame)

        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid-com-shift", self.robot, False)
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

        lx = 0.10
        ly = 0.05
        contact_points = np.array(
            [
                [lx, ly, 0.0],
                [lx, -ly, 0.0],
                [-lx, -ly, 0.0],
                [-lx, ly, 0.0],
            ]
        ).T

        kp_contact = 50.0
        kd_contact = 2.0 * math.sqrt(kp_contact)
        w_force_reg = 1e-5

        self.contact_left = tsid.Contact6d(
            "contact-left",
            self.robot,
            self.left_foot_frame,
            contact_points,
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
            contact_points,
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
        kp_com = 40.0
        kd_com = 2.0 * math.sqrt(kp_com)
        w_com = 1.0

        self.com_task = tsid.TaskComEquality("task-com", self.robot)
        self.com_task.setKp(kp_com * np.ones(3))
        self.com_task.setKd(kd_com * np.ones(3))
        self.invdyn.addMotionTask(self.com_task, w_com, 1, 0.0)

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
        com_vel_ref: np.ndarray,
        com_acc_ref: np.ndarray,
    ) -> np.ndarray:
        com_sample = tsid.TrajectorySample(3)
        com_sample.pos(com_ref)
        com_sample.vel(com_vel_ref)
        com_sample.acc(com_acc_ref)

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
    parser.add_argument("--period", type=float, default=GeneralParams.period)
    parser.add_argument("--com-shift-ratio", type=float, default=GeneralParams.com_shift_ratio)
    return parser.parse_args()


def make_com_reference(
    t: float,
    params: GeneralParams,
    com_center: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
):
    com_ref = com_center.copy()
    com_vel_ref = np.zeros(3)
    com_acc_ref = np.zeros(3)

    if t < params.settle_duration:
        return com_ref, com_vel_ref, com_acc_ref

    half_step = 0.5 * abs(left_foot_pos[1] - right_foot_pos[1])
    amplitude = params.com_shift_ratio * half_step
    omega = 2.0 * math.pi / params.period
    phase = omega * (t - params.settle_duration)

    com_ref[1] = com_center[1] + amplitude * math.sin(phase)
    com_vel_ref[1] = amplitude * omega * math.cos(phase)
    com_acc_ref[1] = -amplitude * omega * omega * math.sin(phase)

    return com_ref, com_vel_ref, com_acc_ref


def main():
    args = parse_args()
    params = GeneralParams(
        duration=args.duration,
        period=args.period,
        com_shift_ratio=args.com_shift_ratio,
    )

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

    controller = TSIDComShiftController(
        urdf_path=urdf_path,
        package_root=package_root,
        q0=q_init,
        left_foot_frame="left_sole_link",
        right_foot_frame="right_sole_link",
    )

    com0 = pin.centerOfMass(talos.model, talos.data, q_init)
    feet_mid = 0.5 * (oMf_lf_tgt.translation + oMf_rf_tgt.translation)
    com_center = np.array([feet_mid[0], feet_mid[1], com0[2]])

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

        com_ref, com_vel_ref, com_acc_ref = make_com_reference(
            t,
            params,
            com_center,
            oMf_lf_tgt.translation,
            oMf_rf_tgt.translation,
        )

        try:
            tau = controller.compute(t, q, v, com_ref, com_vel_ref, com_acc_ref)
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
