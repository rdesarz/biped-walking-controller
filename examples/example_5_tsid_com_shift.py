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


class TSIDController:
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

        self.left_foot_frame = "leg_left_6_joint"
        self.right_foot_frame = "leg_right_6_joint"
        self.left_foot_id = self._get_frame_id(left_foot_frame)
        self.right_foot_id = self._get_frame_id(right_foot_frame)

        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
        self.invdyn.computeProblemData(0.0, q0, self.v0)
        self.data = self.invdyn.data()

        self._add_foot_contacts()
        self._add_com_task()
        # self._add_waist_task()
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

        lxp = 0.1  # foot length in positive x direction
        lxn = 0.11  # foot length in negative x direction
        lyp = 0.069  # foot length in positive y direction
        lyn = 0.069  # foot length in negative y direction
        lz = 0.107  # foot sole height with respect to ankle joint

        contact_point = np.ones((3, 4)) * lz
        contact_point[0, :] = [-lxn, -lxn, lxp, lxp]
        contact_point[1, :] = [-lyn, lyp, -lyn, lyp]

        kp_contact = 30.0
        kd_contact = 2.0 * math.sqrt(kp_contact)

        # Weight for contact force regularization in the cost
        w_force_reg = 1e-5

        # Left foot
        self.contact_left = tsid.Contact6d(
            "contact-left",
            self.robot,
            self.left_foot_frame,
            contact_point,
            contact_normal,
            mu,
            f_min,
            f_max,
        )
        self.contact_left.setKp(kp_contact * np.ones(6))
        self.contact_left.setKd(kd_contact * np.ones(6))
        self.contact_left.setReference(
            self.robot.position(self.data, self.model.getJointId(self.left_foot_frame))
        )
        self.invdyn.addRigidContact(self.contact_left, w_force_reg, 1.0, 1)

        self.contact_right = tsid.Contact6d(
            "contact-right",
            self.robot,
            self.right_foot_frame,
            contact_point,
            contact_normal,
            mu,
            f_min,
            f_max,
        )
        self.contact_right.setKp(kp_contact * np.ones(6))
        self.contact_right.setKd(kd_contact * np.ones(6))
        self.contact_right.setReference(
            self.robot.position(self.data, self.model.getJointId(self.right_foot_frame))
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
        kd_waist = 2.0 * math.sqrt(kp_waist)
        w_waist = 1.0

        self.waist_task = tsid.TaskSE3Equality("task-waist", self.robot, "root_joint")
        self.waist_task.setKp(kp_waist * np.ones(6))
        self.waist_task.setKd(kd_waist * np.ones(6))

        # Add a Mask to the task which will select the vector dimensions on which the task will act.
        # In this case the waist configuration is a vector 6d (position and orientation -> SE3)
        # Here we set a mask = [0 0 0 1 1 1] so the task on the waist will act on the orientation of the robot
        mask = np.ones(6)
        mask[:3] = 0.0
        self.waist_task.setMask(mask)
        # Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
        self.invdyn.addMotionTask(self.waist_task, w_waist, 1, 0.0)

    def _add_posture_task(self, q0: np.ndarray):
        kp_posture = np.array(  # proportional gain of joint posture task
            [
                10.0,
                5.0,
                5.0,
                1.0,
                1.0,
                10.0,  # lleg, low gain on axis along y and knee
                10.0,
                5.0,
                5.0,
                1.0,
                1.0,
                10.0,  # rleg
                500.0,
                500.0,  # chest
                50.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,  # larm
                50.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,  # rarm
                100.0,
                100.0,
            ]  # head
        )

        w_posture = 0.1

        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)
        self.posture_task.setKp(kp_posture)
        self.posture_task.setKd(2.0 * kp_posture)
        self.invdyn.addMotionTask(self.posture_task, w_posture, 1, 0.0)

        self.posture_sample = tsid.TrajectorySample(self.robot.nv - 6)
        self.posture_sample.value(q0[7:])

    def compute(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        com_ref: np.ndarray,
    ) -> np.ndarray:
        com_sample = tsid.TrajectorySample(3)
        com_sample.value(com_ref)
        com_sample.derivative(np.zeros(3))
        com_sample.second_derivative(np.zeros(3))

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
    urdf_path = package_root / "talos_data" / "urdf" / "talos_reduced.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    talos = Talos(path_to_model=package_root, reduced=False)
    q_init = talos.set_and_get_default_pose()

    simulator = Simulator(
        dt=params.dt,
        path_to_robot_urdf=urdf_path,
        model=talos,
        launch_gui=True,
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

    controller = TSIDController(
        urdf_path=urdf_path,
        package_root=package_root,
        q0=q_init,
        left_foot_frame="left_sole_link",
        right_foot_frame="right_sole_link",
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
