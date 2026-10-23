#!/usr/bin/env python3
"""OpenArm CAN teleoperation with the deployed mid IK tuning and Adamo XR input.

Hold both inner grips to move; release either to pause and reposition your hands.
Triggers control the grippers independently. B returns home; R resets calibration.
"""

import argparse
from contextlib import ExitStack
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Optional

import numpy as np

import adamo

from .can_motor import OpenArmCAN
from .kinematics import OpenArmKinematics
from .vr_input import VRControllerState, VRInput

# ── 1 Euro Filter ────────────────────────────────────────────────────


def _smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


class OneEuroFilter:
    def __init__(self, x0: np.ndarray, min_cutoff=2.25, beta=0.006, d_cutoff=1.0):
        self.min_cutoff = np.full(x0.shape, min_cutoff)
        self.beta = np.full(x0.shape, beta)
        self.d_cutoff = np.full(x0.shape, d_cutoff)
        self.x_prev = x0.astype(np.float64)
        self.dx_prev = np.zeros(x0.shape, dtype=np.float64)
        self.t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x.astype(np.float64)
            return x
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        t_e = np.full(x.shape, t_e)
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ── Arm IK Controller (Damped Least-Squares) ─────────────────────────

# VR origins captured dynamically at calibration time

# Bent-elbow home configs (j4 = elbow bend forward, j2 = 0 keeps arm at side)
HOME_JOINTS_RIGHT = np.array([0, 0, 0, 1.2, 0, 0, 0])
HOME_JOINTS_LEFT = np.array([0, 0, 0, 1.2, 0, 0, 0])

# Joint limits from URDF [lower, upper] for each of 7 joints
JOINT_LIMITS_RIGHT = np.array([
    [-1.396263, 3.490659],   # j1: base yaw
    [-0.174533, 3.316125],   # j2: shoulder pitch
    [-1.570796, 1.570796],   # j3: upper arm rotation
    [0.0, 2.443461],         # j4: elbow (bends one way only)
    [-1.570796, 1.570796],   # j5: forearm rotation
    [-0.785398, 0.785398],   # j6: wrist pitch
    [-1.570796, 1.570796],   # j7: wrist roll
])
JOINT_LIMITS_LEFT = np.array([
    [-3.490659, 1.396263],
    [-3.316125, 0.174533],
    [-1.570796, 1.570796],
    [0.0, 2.443461],
    [-1.570796, 1.570796],
    [-0.785398, 0.785398],
    [-1.570796, 1.570796],
])

# ── MID TUNING: halfway between normal and pong ──
DLS_LAMBDA_MAX = 0.035     # Mid damping (normal 0.05, pong 0.02)
DLS_SIGMA_THRESH = 0.04    # Mid singularity threshold (normal 0.05, pong 0.03)
NULL_SPACE_GAIN = 0.15     # Mid pull toward home (normal 0.5, pong 0.2)
IK_SUB_ITERS = 3           # Sub-iterations per control cycle
ORIENT_WEIGHT = 0.5      # Mid orientation priority (normal 0.3, pong 0.15)
JOINT_LIMIT_K = 10.0       # Joint-limit avoidance gain (unchanged)
MANIP_GAIN = 0.0           # Disabled (unchanged)


def _quat_to_yaw(q):
    """Extract yaw (rotation around WebXR Y axis) from quaternion [w, x, y, z]."""
    w, x, y, z = q
    forward_x = -2.0 * (x * z + w * y)
    forward_z = 2.0 * (x * x + y * y) - 1.0
    return np.arctan2(forward_x, -forward_z)


def _to_calibrated_space(p_controller, calib_yaw):
    """Transform controller position from local-floor to calibrated space.

    De-rotates by the head yaw captured at calibration time so that the
    user's initial facing direction always maps to -Z (WebXR forward).
    No head position subtraction — controller positions are already
    room-fixed via the WebXR local-floor reference space.
    """
    c, s = np.cos(calib_yaw), np.sin(calib_yaw)
    R_inv = np.array([[ c, 0,  s],
                      [ 0, 1,  0],
                      [-s, 0,  c]])
    return R_inv @ p_controller


def _quat_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _skew(v):
    """Skew-symmetric matrix for cross product: skew(a) @ b = a x b."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def _ee_jacobian(J_full, ee_pos):
    """Convert spatial Jacobian to body Jacobian at EE (6xN).

    Placo's frame_jacobian("world") returns the spatial Jacobian where
    rows 0:3 are linear velocity at the world origin. To get linear
    velocity at the EE position: J_pos = J_lin - skew(p) @ J_ang.
    Angular rows (3:6) are unchanged.

    Returns 6xN: [position; orientation].
    """
    J_pos = J_full[:3, :] - _skew(ee_pos) @ J_full[3:, :]
    J_ang = J_full[3:, :]
    return np.vstack([J_pos, J_ang])


def _orientation_error(R_desired, R_current):
    """Compute orientation error as a 3-vector (axis-angle-like).

    Uses the vee map of the skew-symmetric part of Rd @ Rc^T:
      e = 0.5 * vee(Rd @ Rc^T - Rc @ Rd^T)
    """
    R_err = R_desired @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def _vr_orientation_to_robot(vr_quat, calib_yaw):
    """Transform VR controller orientation to robot frame.

    1. De-rotate by calibration yaw (fixed at calibration time)
    2. Apply WebXR->robot frame rotation

    WebXR: +X=right, +Y=up,  -Z=forward
    Robot: +X=forward, +Y=left, +Z=up

    Returns 3x3 rotation matrix in robot frame.
    """
    R_ctrl = _quat_to_rotation_matrix(vr_quat)
    c, s = np.cos(calib_yaw), np.sin(calib_yaw)
    R_calib_inv = np.array([[ c, 0,  s],
                             [ 0, 1,  0],
                             [-s, 0,  c]])
    R_hfs = R_calib_inv @ R_ctrl
    R_webxr_to_robot = np.array([[ 0, 0, -1],
                                  [-1, 0,  0],
                                  [ 0, 1,  0]], dtype=np.float64)
    return R_webxr_to_robot @ R_hfs


def _joint_limit_weights(q, joint_limits, k=JOINT_LIMIT_K):
    """Compute per-joint weights that increase near joint limits.

    w_i = 1 + k * ((q_i - q_mid) / (q_range/2))^4
    Exponent 4 = mostly flat in center, steep near limits.
    """
    q_mid = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
    q_half_range = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])
    q_half_range = np.maximum(q_half_range, 1e-6)
    normalized = (q - q_mid) / q_half_range
    normalized = np.clip(normalized, -1.0, 1.0)
    return 1.0 + k * normalized**4


class ArmIKController:
    """Damped least-squares IK controller with orientation tracking.

    Improvements over basic DLS:
    - Full 6-DOF task (position + orientation) using VR controller pose
    - Adaptive damping based on smallest singular value of Jacobian
    - Weighted joint-limit avoidance (joints near limits get higher weight)
    - Manipulability gradient in null-space (pushes away from singularities)
    - Factored pseudoinverse (single solve for task + null-space)
    """

    def __init__(
        self,
        kinematics: OpenArmKinematics,
        side: str = "left",
        position_scale: float = 1.0,
        max_joint_velocity: float = 2.0,
    ):
        self.kin = kinematics
        self.side = side
        self.position_scale = position_scale
        self.max_joint_velocity = max_joint_velocity
        self.vr_origin: Optional[np.ndarray] = None
        self.home_joints = HOME_JOINTS_LEFT if side == "left" else HOME_JOINTS_RIGHT
        self.joint_limits = JOINT_LIMITS_LEFT if side == "left" else JOINT_LIMITS_RIGHT
        self._home_ee_pos: Optional[np.ndarray] = None
        self._home_ee_rot: Optional[np.ndarray] = None
        self._vr_orient_ref: Optional[np.ndarray] = None
        self._calib_yaw: Optional[float] = None
        self._current_joints: Optional[np.ndarray] = None
        self._debug_count = 0

    def calibrate(self, joint_pos: np.ndarray, vr_position: np.ndarray,
                  vr_orientation: Optional[np.ndarray] = None,
                  calib_yaw: Optional[float] = None):
        self._current_joints = joint_pos.copy()
        self.vr_origin = vr_position.copy()
        self._calib_yaw = calib_yaw
        ee_pose = self.kin.forward_kinematics(joint_pos)
        self._home_ee_pos = ee_pose[:3, 3].copy()
        self._home_ee_rot = ee_pose[:3, :3].copy()
        if vr_orientation is not None and calib_yaw is not None:
            self._vr_orient_ref = _vr_orientation_to_robot(vr_orientation, calib_yaw)
        else:
            self._vr_orient_ref = None
        self._debug_count = 0
        print(f"  [{self.side}] VR origin: {np.round(vr_position, 4)}")
        print(f"  [{self.side}] Calibrated joints: {np.round(joint_pos, 3)}")
        print(f"  [{self.side}] Calibrated EE pos: ({self._home_ee_pos[0]:.4f}, "
              f"{self._home_ee_pos[1]:.4f}, {self._home_ee_pos[2]:.4f})")
        print(f"  [{self.side}] Orientation tracking: "
              f"{'ON' if self._vr_orient_ref is not None else 'OFF'}")

    def compute(self, vr_position: np.ndarray, dt: float,
                vr_orientation: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self._home_ee_pos is None or vr_position is None:
            return None

        # -- Target pose --
        vr_delta = (vr_position - self.vr_origin) * self.position_scale
        robot_delta = np.array([-vr_delta[2], -vr_delta[0], vr_delta[1]])
        target_pos = self._home_ee_pos + robot_delta

        has_orient = (vr_orientation is not None and self._calib_yaw is not None
                      and self._vr_orient_ref is not None)
        R_target = None
        if has_orient:
            R_vr_now = _vr_orientation_to_robot(vr_orientation, self._calib_yaw)
            R_delta_world = R_vr_now @ self._vr_orient_ref.T
            R_target = R_delta_world @ self._home_ee_rot

        # -- Sub-iterations: multiple IK solves per control cycle --
        sub_dt = dt / IK_SUB_ITERS
        for _ in range(IK_SUB_ITERS):
            current_ee = self.kin.forward_kinematics(self._current_joints)
            current_pos = current_ee[:3, 3]
            current_rot = current_ee[:3, :3]

            J_full = self.kin.get_jacobian(self._current_joints)
            pos_error = target_pos - current_pos

            if has_orient:
                orient_error = _orientation_error(R_target, current_rot)
                error = np.concatenate([pos_error, ORIENT_WEIGHT * orient_error])
                J = _ee_jacobian(J_full, current_pos)
                J[3:, :] *= ORIENT_WEIGHT
                task_dim = 6
            else:
                error = pos_error
                J = _ee_jacobian(J_full, current_pos)[:3, :]
                task_dim = 3

            # Adaptive damping
            svs = np.linalg.svd(J, compute_uv=False)
            sigma_min = svs[-1] if len(svs) > 0 else 0.0
            if sigma_min < DLS_SIGMA_THRESH:
                lam = DLS_LAMBDA_MAX * (1.0 - (sigma_min / DLS_SIGMA_THRESH)**2)
            else:
                lam = 0.0

            # Joint-limit weighting
            w = _joint_limit_weights(self._current_joints, self.joint_limits)
            w_inv = 1.0 / w

            # Weighted DLS
            JWinv = J * w_inv[np.newaxis, :]  # broadcast instead of diag multiply
            JWinvJT = JWinv @ J.T
            reg = lam**2 * np.eye(task_dim) if lam > 0 else 1e-10 * np.eye(task_dim)
            A_inv = np.linalg.solve(JWinvJT + reg, np.eye(task_dim))
            J_pinv = (J.T @ A_inv) * w_inv[:, np.newaxis]  # W_inv @ J^T @ A_inv
            dq_task = J_pinv @ error

            # Error-gated null-space: suppress near-home bias during large errors
            err_norm = np.linalg.norm(pos_error)
            ns_scale = np.exp(-50.0 * err_norm**2)  # Error is in metres
            null_proj = np.eye(7) - J_pinv @ J
            dq_null_home = ns_scale * NULL_SPACE_GAIN * (self.home_joints - self._current_joints)

            # Manipulability gradient (disabled by default: MANIP_GAIN = 0)
            manip_base = 0.0
            dq_null_manip = np.zeros(7)
            if MANIP_GAIN > 0:
                manip_base = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
                eps = 1e-4
                for i in range(7):
                    q_pert = self._current_joints.copy()
                    q_pert[i] += eps
                    J_pert_full = self.kin.get_jacobian(q_pert)
                    if has_orient:
                        J_pert = _ee_jacobian(J_pert_full, current_pos)
                        J_pert[3:, :] *= ORIENT_WEIGHT
                    else:
                        J_pert = _ee_jacobian(J_pert_full, current_pos)[:3, :]
                    manip_pert = np.sqrt(max(np.linalg.det(J_pert @ J_pert.T), 0.0))
                    dq_null_manip[i] = (manip_pert - manip_base) / eps

            dq = dq_task + null_proj @ (dq_null_home + MANIP_GAIN * dq_null_manip)

            # Velocity limit per sub-iteration
            max_step = self.max_joint_velocity * sub_dt
            max_abs = np.max(np.abs(dq))
            if max_abs > max_step:
                dq *= max_step / max_abs

            self._current_joints = self._current_joints + dq

            # Hard clamp
            self._current_joints = np.clip(
                self._current_joints,
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )

        # -- Debug logging --
        self._debug_count += 1
        if self._debug_count <= 10 or self._debug_count % 60 == 0:
            cond = svs[0] / svs[-1] if svs[-1] > 1e-10 else float('inf')
            print(f"  [{self.side}] #{self._debug_count}: "
                  f"err_pos={np.round(pos_error, 4)} "
                  f"sigma_min={sigma_min:.4f} lam={lam:.4f} "
                  f"ns={ns_scale:.3f} cond={cond:.1f} "
                  f"joints={np.round(self._current_joints, 3)}",
                  flush=True)

        return self._current_joints.copy()

    def reset(self, joint_pos: np.ndarray, vr_position: np.ndarray,
              vr_orientation: Optional[np.ndarray] = None,
              calib_yaw: Optional[float] = None):
        self.calibrate(joint_pos, vr_position, vr_orientation, calib_yaw)


# ── Main loop ────────────────────────────────────────────────────────

class KeyboardResetListener:
    """Non-blocking terminal key reader for runtime IK reset."""

    def __init__(self):
        self._stdin = sys.stdin
        self._fd = None
        self._original_attrs = None
        self.enabled = False

        if not self._stdin.isatty():
            return

        try:
            self._fd = self._stdin.fileno()
            self._original_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
            print("  Keyboard reset: press R to reset IK", flush=True)
        except termios.error as exc:
            print(f"  Keyboard reset disabled: {exc}", flush=True)

    def consume_reset(self) -> bool:
        if not self.enabled:
            return False

        reset_requested = False
        while True:
            readable, _, _ = select.select([self._stdin], [], [], 0)
            if not readable:
                break
            ch = self._stdin.read(1)
            if not ch:
                break
            if ch in ("r", "R"):
                reset_requested = True
        return reset_requested

    def close(self):
        if self.enabled and self._original_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_attrs)
            except termios.error:
                pass
        self.enabled = False


def _latest_joint_positions_for_reset(
    arm: Optional[OpenArmCAN],
    fallback: Optional[np.ndarray],
    side: str,
) -> Optional[np.ndarray]:
    if arm is None:
        return None

    try:
        for state in arm.states[:7]:
            state.valid = False
        arm.refresh()
        n_valid = sum(1 for state in arm.states[:7] if state.valid)
        if n_valid == 7:
            return arm.get_positions()
        if fallback is not None:
            print(f"  [{side}] Reset using last commanded joints "
                  f"({n_valid}/7 current motor states valid)", flush=True)
            return fallback.copy()
        print(f"  [{side}] Reset cannot read joints "
              f"({n_valid}/7 current motor states valid)", flush=True)
        return None
    except Exception as exc:
        if fallback is not None:
            print(f"  [{side}] Reset joint refresh failed ({exc}); "
                  "using last commanded joints", flush=True)
            return fallback.copy()
        print(f"  [{side}] Reset joint refresh failed: {exc}", flush=True)
        return None

GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenArm CAN VR Teleop — MID (balanced damping)")
    parser.add_argument("--robot-name", default="orin")
    parser.add_argument("--urdf-path", default="")
    parser.add_argument("--right-can", default="can0", help="CAN interface for right arm")
    parser.add_argument("--left-can", default="can1", help="CAN interface for left arm")
    parser.add_argument("--side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--max-joint-velocity", type=float, default=6.5)
    parser.add_argument("--control-rate", type=float, default=60.0)
    parser.add_argument("--motor-kp", type=float, default=35.0,
                        help="Motor position gain (0-500)")
    parser.add_argument("--motor-kd", type=float, default=1.0,
                        help="Motor damping gain (0-5)")
    parser.add_argument("--protocol", choices=["quic", "tcp", "udp"], default="quic")
    parser.add_argument("--input-timeout", type=float, default=0.25,
                        help="Pause on missing controller poses/buttons after this many seconds")
    parser.add_argument("--check", action="store_true",
                        help="Check the URDF, FK and IK locally without connecting or opening CAN")
    args = parser.parse_args(argv)
    for name in ("control_rate", "max_joint_velocity", "position_scale", "input_timeout"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name, maximum in (("motor_kp", 500), ("motor_kd", 5)):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= maximum:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and {maximum}")
    if args.side == "both" and args.left_can == args.right_can:
        parser.error("left and right arms must use different CAN interfaces")
    try:
        with ExitStack() as resources:
            _run(args, resources)
    except KeyboardInterrupt:
        print("\nStopped", flush=True)
    except Exception as exc:
        print(f"OpenArm teleop failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


def _run(args, resources):

    urdf_path = args.urdf_path or str(
        Path(__file__).parent / "urdf" / "openarm_bimanual.urdf"
    )

    use_right = args.side in ("right", "both")
    use_left = args.side in ("left", "both")

    print("OpenArm CAN VR Teleop — MID MODE (balanced damping)", flush=True)
    if use_right:
        print(f"  Right arm: {args.right_can}", flush=True)
    if use_left:
        print(f"  Left arm:  {args.left_can}", flush=True)
    print(f"  URDF: {urdf_path}", flush=True)

    # Load kinematics (Placo)
    print("Loading kinematics...", flush=True)
    right_kin = OpenArmKinematics(urdf_path, "openarm_right_") if use_right else None
    left_kin = OpenArmKinematics(urdf_path, "openarm_left_") if use_left else None
    print("  Kinematics ready", flush=True)

    if args.check:
        for side, kin in (("right", right_kin), ("left", left_kin)):
            if kin is None:
                continue
            ik = ArmIKController(kin, side=side, max_joint_velocity=args.max_joint_velocity)
            q = ik.home_joints.copy().astype(float)
            ik.calibrate(q, np.zeros(3), np.array([1., 0., 0., 0.]), 0.0)
            result = ik.compute(np.array([0., 0., -0.01]), 1.0 / args.control_rate,
                                np.array([1., 0., 0., 0.]))
            if result is None or not np.all(np.isfinite(result)):
                raise RuntimeError(f"{side} IK did not produce finite joint positions")
            print(f"  [{side}] FK/Jacobian/IK check passed", flush=True)
        return

    # Authenticate and subscribe before enabling motors. Robot owns the SDK session.
    print("Connecting to Adamo...", flush=True)
    robot = resources.enter_context(adamo.Robot(name=args.robot_name, protocol=args.protocol))
    vr = VRInput(robot, timeout=args.input_timeout)
    resources.callback(vr.close)

    # Initialize CAN
    right_arm = left_arm = None
    right_pos = left_pos = None

    if use_right:
        print(f"Initializing right arm ({args.right_can})...", flush=True)
        right_arm = OpenArmCAN(args.right_can)
        resources.callback(right_arm.close)
        right_arm.enable_all()
        time.sleep(0.05)
        right_arm.refresh()
        right_pos = right_arm.get_positions()
        n_ok = sum(1 for s in right_arm.states[:7] if s.valid)
        print(f"  Right motors: {n_ok}/7  pos={np.round(right_pos, 3)}", flush=True)
        if n_ok != 7:
            raise RuntimeError(f"Right arm: {n_ok}/7 motors responded; check CAN and motor power")

    if use_left:
        print(f"Initializing left arm ({args.left_can})...", flush=True)
        left_arm = OpenArmCAN(args.left_can)
        resources.callback(left_arm.close)
        left_arm.enable_all()
        time.sleep(0.05)
        left_arm.refresh()
        left_pos = left_arm.get_positions()
        n_ok = sum(1 for s in left_arm.states[:7] if s.valid)
        print(f"  Left motors:  {n_ok}/7  pos={np.round(left_pos, 3)}", flush=True)
        if n_ok != 7:
            raise RuntimeError(f"Left arm: {n_ok}/7 motors responded; check CAN and motor power")

    # IK controllers
    right_ik = ArmIKController(
        right_kin, side="right",
        position_scale=args.position_scale,
        max_joint_velocity=args.max_joint_velocity,
    ) if use_right else None

    left_ik = ArmIKController(
        left_kin, side="left",
        position_scale=args.position_scale,
        max_joint_velocity=args.max_joint_velocity,
    ) if use_left else None

    dt = 1.0 / args.control_rate
    right_joints = right_pos.copy() if right_pos is not None else None
    left_joints = left_pos.copy() if left_pos is not None else None

    # ── Smooth homing: move arms from current position to bent-elbow home ──
    print("Homing to bent-elbow position...", flush=True)
    if right_ik:
        T = right_kin.forward_kinematics(right_joints)
        print(f"  [right] Current EE: ({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})")
        T = right_kin.forward_kinematics(HOME_JOINTS_RIGHT)
        print(f"  [right] Home EE:    ({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})")
    if left_ik:
        T = left_kin.forward_kinematics(left_joints)
        print(f"  [left]  Current EE: ({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})")
        T = left_kin.forward_kinematics(HOME_JOINTS_LEFT)
        print(f"  [left]  Home EE:    ({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})")

    HOMING_DURATION = 2.0  # seconds
    HOMING_RATE = 60.0
    homing_dt = 1.0 / HOMING_RATE
    homing_steps = int(HOMING_DURATION * HOMING_RATE)

    right_start = right_joints.copy() if right_joints is not None else None
    left_start = left_joints.copy() if left_joints is not None else None

    for step in range(homing_steps):
        t_frac = (step + 1) / homing_steps
        # Smooth ease-in-out (cosine interpolation)
        alpha = 0.5 * (1.0 - np.cos(np.pi * t_frac))

        if right_arm and right_start is not None:
            right_joints = right_start + alpha * (HOME_JOINTS_RIGHT - right_start)
            right_arm.set_joint_positions(
                right_joints, kp=args.motor_kp * 0.5, kd=args.motor_kd,
                process_responses=False)
            right_arm._process_responses()

        if left_arm and left_start is not None:
            left_joints = left_start + alpha * (HOME_JOINTS_LEFT - left_start)
            left_arm.set_joint_positions(
                left_joints, kp=args.motor_kp * 0.5, kd=args.motor_kd,
                process_responses=False)
            left_arm._process_responses()

        time.sleep(homing_dt)

    right_joints = HOME_JOINTS_RIGHT.copy() if use_right else None
    left_joints = HOME_JOINTS_LEFT.copy() if use_left else None

    # Open grippers
    if right_arm:
        right_arm.set_gripper(GRIPPER_OPEN, kp=args.motor_kp, kd=args.motor_kd)
    if left_arm:
        left_arm.set_gripper(GRIPPER_OPEN, kp=args.motor_kp, kd=args.motor_kd)

    print("  Homing complete", flush=True)

    keyboard = KeyboardResetListener()
    resources.callback(keyboard.close)
    vr.read()  # Discard button edges received during startup homing.

    right_euro: Optional[OneEuroFilter] = None
    left_euro: Optional[OneEuroFilter] = None
    calibrated = False
    calib_yaw: Optional[float] = None
    clutch_active = False
    start_time = time.monotonic()
    last_debug = 0

    def rebuild_ik_solver(
        reason: str,
        left_vr: VRControllerState,
        right_vr: VRControllerState,
        head_quat: Optional[np.ndarray],
    ) -> bool:
        nonlocal right_ik, left_ik
        nonlocal right_joints, left_joints
        nonlocal right_euro, left_euro
        nonlocal calibrated, calib_yaw

        print(f"Reset requested ({reason}) — rebuilding IK solver", flush=True)

        right_euro = None
        left_euro = None
        calibrated = False

        if use_right:
            right_joints = _latest_joint_positions_for_reset(
                right_arm, right_joints, "right")
            right_ik = ArmIKController(
                right_kin, side="right",
                position_scale=args.position_scale,
                max_joint_velocity=args.max_joint_velocity,
            )
        if use_left:
            left_joints = _latest_joint_positions_for_reset(
                left_arm, left_joints, "left")
            left_ik = ArmIKController(
                left_kin, side="left",
                position_scale=args.position_scale,
                max_joint_velocity=args.max_joint_velocity,
            )

        got_right = (not use_right or
                     (right_joints is not None and right_vr.position is not None))
        got_left = (not use_left or
                    (left_joints is not None and left_vr.position is not None))
        if head_quat is None or not got_right or not got_left:
            print("  Reset pending: waiting for current head/controller pose",
                  flush=True)
            return False

        calib_yaw = _quat_to_yaw(head_quat)

        if right_ik and right_vr.position is not None:
            right_cs = _to_calibrated_space(right_vr.position, calib_yaw)
            right_euro = OneEuroFilter(right_cs)
            right_ik.calibrate(
                right_joints, right_cs, right_vr.orientation, calib_yaw)

        if left_ik and left_vr.position is not None:
            left_cs = _to_calibrated_space(left_vr.position, calib_yaw)
            left_euro = OneEuroFilter(left_cs)
            left_ik.calibrate(
                left_joints, left_cs, left_vr.orientation, calib_yaw)

        calibrated = True
        print(f"  IK solver reset complete "
              f"(calib yaw={np.degrees(calib_yaw):.1f}°)", flush=True)
        return True

    def engage_clutch(
        left_vr: VRControllerState,
        right_vr: VRControllerState,
    ) -> bool:
        """Rebase both IK solvers at the controller and robot poses now."""
        nonlocal right_euro, left_euro

        if calib_yaw is None:
            return False
        if use_right and (right_vr.position is None or right_joints is None):
            return False
        if use_left and (left_vr.position is None or left_joints is None):
            return False

        if right_ik and right_vr.position is not None:
            right_cs = _to_calibrated_space(right_vr.position, calib_yaw)
            right_euro = OneEuroFilter(right_cs)
            right_ik.calibrate(
                right_joints, right_cs, right_vr.orientation, calib_yaw)
        if left_ik and left_vr.position is not None:
            left_cs = _to_calibrated_space(left_vr.position, calib_yaw)
            left_euro = OneEuroFilter(left_cs)
            left_ik.calibrate(
                left_joints, left_cs, left_vr.orientation, calib_yaw)

        print("Both inner grips held — relative IK movement enabled", flush=True)
        robot.log("Both inner grips held — arm movement enabled")
        return True

    def return_home(
        left_vr: VRControllerState,
        right_vr: VRControllerState,
        head_quat: Optional[np.ndarray],
    ):
        """Smoothly return both enabled arms home, then rebase the IK."""
        nonlocal right_joints, left_joints, clutch_active

        print("B pressed — returning to home", flush=True)
        clutch_active = False
        right_start = _latest_joint_positions_for_reset(
            right_arm, right_joints, "right") if use_right else None
        left_start = _latest_joint_positions_for_reset(
            left_arm, left_joints, "left") if use_left else None

        for step in range(homing_steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step + 1) / homing_steps))
            if right_arm and right_start is not None:
                right_joints = right_start + alpha * (HOME_JOINTS_RIGHT - right_start)
                right_arm.set_joint_positions(
                    right_joints, kp=args.motor_kp * 0.5, kd=args.motor_kd,
                    process_responses=False)
                right_arm._process_responses()
            if left_arm and left_start is not None:
                left_joints = left_start + alpha * (HOME_JOINTS_LEFT - left_start)
                left_arm.set_joint_positions(
                    left_joints, kp=args.motor_kp * 0.5, kd=args.motor_kd,
                    process_responses=False)
                left_arm._process_responses()
            time.sleep(homing_dt)

        if use_right:
            right_joints = HOME_JOINTS_RIGHT.copy()
        if use_left:
            left_joints = HOME_JOINTS_LEFT.copy()
        state = vr.read()
        rebuild_ik_solver("VR B after return home",
                          state.left, state.right, state.head_orientation)
        print("  Return home complete; hold both inner grips to move", flush=True)

    print("Waiting for VR data — hold both inner grips to move", flush=True)
    robot.log("Ready — hold both inner grips to move; release either to pause")
    rate = adamo.Rate(args.control_rate)

    try:
        while True:
            loop_start = time.monotonic()
            t = loop_start - start_time

            state = vr.read()
            left_vr, right_vr = state.left, state.right
            head_quat = state.head_orientation
            if state.rebase_requested and clutch_active:
                clutch_active = False
                robot.log("Movement paused or tracking resumed — rebasing the controller poses")
            keyboard_reset = keyboard.consume_reset()
            if state.home_requested:
                return_home(left_vr, right_vr, head_quat)
                continue
            if keyboard_reset:
                rebuild_ik_solver("keyboard R", left_vr, right_vr, head_quat)
                clutch_active = False

            # Debug: print VR state periodically
            if t - last_debug > 2.0:
                last_debug = t
                r_pos = right_vr.position if right_vr.position is not None else "None"
                l_pos = left_vr.position if left_vr.position is not None else "None"
                print(f"  [VR] t={t:.1f}s  L={l_pos}  R={r_pos}  "
                      f"L.upd={left_vr.updated} R.upd={right_vr.updated}", flush=True)

            if not calibrated:
                got_right = use_right and right_vr.position is not None
                got_left = use_left and left_vr.position is not None
                got_head = head_quat is not None
                need_right = use_right
                need_left = use_left
                if (got_right or not need_right) and (got_left or not need_left) and got_head:
                    calib_yaw = _quat_to_yaw(head_quat)
                    if right_ik and right_vr.position is not None:
                        right_cs = _to_calibrated_space(right_vr.position, calib_yaw)
                        right_euro = OneEuroFilter(right_cs)
                        right_ik.calibrate(right_joints, right_cs, right_vr.orientation, calib_yaw)
                    if left_ik and left_vr.position is not None:
                        left_cs = _to_calibrated_space(left_vr.position, calib_yaw)
                        left_euro = OneEuroFilter(left_cs)
                        left_ik.calibrate(left_joints, left_cs, left_vr.orientation, calib_yaw)
                    calibrated = True
                    print(f"Calibrated — VR teleop active (calib yaw={np.degrees(calib_yaw):.1f}°)", flush=True)
                else:
                    rate.sleep()
                    continue

            both_grips = left_vr.grip_pressed and right_vr.grip_pressed
            if both_grips and not clutch_active:
                clutch_active = engage_clutch(left_vr, right_vr)
            elif not both_grips and clutch_active:
                clutch_active = False
                print("Grip released or tracking lost — IK movement paused", flush=True)
                robot.log("Arm movement paused — hold both inner grips with fresh tracking to move")

            # Right arm
            if (clutch_active and right_ik and right_vr.updated
                    and right_vr.position is not None):
                right_cs = _to_calibrated_space(right_vr.position, calib_yaw)
                if right_euro is not None:
                    right_cs = right_euro(right_cs, t)
                cmd = right_ik.compute(right_cs, dt, right_vr.orientation)
                if cmd is not None:
                    right_arm.set_joint_positions(
                        cmd, kp=args.motor_kp, kd=args.motor_kd,
                        process_responses=False)
                    right_joints = cmd.copy()
                right_arm._process_responses()

            # Left arm
            if (clutch_active and left_ik and left_vr.updated
                    and left_vr.position is not None):
                left_cs = _to_calibrated_space(left_vr.position, calib_yaw)
                if left_euro is not None:
                    left_cs = left_euro(left_cs, t)
                cmd = left_ik.compute(left_cs, dt, left_vr.orientation)
                if cmd is not None:
                    left_arm.set_joint_positions(
                        cmd, kp=args.motor_kp, kd=args.motor_kd,
                        process_responses=False)
                    left_joints = cmd.copy()
                left_arm._process_responses()

            # Grippers (trigger: 0→open, 1→closed)
            if right_arm and right_vr.fresh:
                t_val = np.clip(right_vr.trigger, 0.0, 1.0)
                grip_pos = GRIPPER_OPEN + t_val * (GRIPPER_CLOSED - GRIPPER_OPEN)
                right_arm.set_gripper(grip_pos, kp=args.motor_kp, kd=args.motor_kd)
            if left_arm and left_vr.fresh:
                t_val = np.clip(left_vr.trigger, 0.0, 1.0)
                grip_pos = GRIPPER_OPEN + t_val * (GRIPPER_CLOSED - GRIPPER_OPEN)
                left_arm.set_gripper(grip_pos, kp=args.motor_kp, kd=args.motor_kd)

            # Debug grippers every 2s
            if t - last_debug < 0.05:
                r_trig = right_vr.trigger if use_right else 0
                l_trig = left_vr.trigger if use_left else 0
                r_grip = GRIPPER_OPEN + np.clip(r_trig, 0, 1) * (GRIPPER_CLOSED - GRIPPER_OPEN)
                l_grip = GRIPPER_OPEN + np.clip(l_trig, 0, 1) * (GRIPPER_CLOSED - GRIPPER_OPEN)
                print(f"  [Grip] R.trig={r_trig:.3f}→cmd={r_grip:.4f}  "
                      f"L.trig={l_trig:.3f}→cmd={l_grip:.4f}", flush=True)

            rate.sleep()

    except Exception as exc:
        robot.log(f"Teleoperation stopped: {exc}", level="error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
