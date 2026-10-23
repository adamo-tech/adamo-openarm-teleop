"""
Direct CAN motor controller for Damiao motors.

Bypasses the openarm_can library to work around the CAN FD recv bug:
the library sends canfd_frame (72 bytes) which the firmware transmits
as FD frames, but motors respond with classical frames that the library
can't read.  This module sends classical can_frame (16 bytes) and receives
on an FD-enabled socket (72 bytes) — the only combination that works with
the tymmothy candleLight FD firmware.
"""

import socket
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Motor types and limits ───────────────────────────────────────────

class MotorType(IntEnum):
    DM4310 = 1
    DM4340 = 3
    DM8009 = 7


@dataclass
class MotorLimits:
    p_max: float  # rad
    v_max: float  # rad/s
    t_max: float  # Nm


MOTOR_LIMITS = {
    MotorType.DM4310: MotorLimits(12.5, 30, 10),
    MotorType.DM4340: MotorLimits(12.5, 8, 28),
    MotorType.DM8009: MotorLimits(12.5, 45, 54),
}

# OpenArm v10 motor configuration (7 joints + gripper)
ARM_MOTOR_TYPES = [
    MotorType.DM8009,  # joint 1
    MotorType.DM8009,  # joint 2
    MotorType.DM4340,  # joint 3
    MotorType.DM4340,  # joint 4
    MotorType.DM4310,  # joint 5
    MotorType.DM4310,  # joint 6
    MotorType.DM4310,  # joint 7
]

ARM_SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
ARM_RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18
GRIPPER_MOTOR_TYPE = MotorType.DM4310


# ── Protocol encoding/decoding ───────────────────────────────────────

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def _double_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = _clamp(x, x_min, x_max)
    span = x_max - x_min
    norm = (x - x_min) / span
    return int(norm * ((1 << bits) - 1))


def _uint_to_double(x: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    norm = x / ((1 << bits) - 1)
    return norm * span + x_min


def pack_enable() -> bytes:
    return b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC'


def pack_disable() -> bytes:
    return b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD'


def pack_mit(motor_type: MotorType, kp: float, kd: float,
             q: float = 0.0, dq: float = 0.0, tau: float = 0.0) -> bytes:
    lim = MOTOR_LIMITS[motor_type]
    q_u = _double_to_uint(q, -lim.p_max, lim.p_max, 16)
    dq_u = _double_to_uint(dq, -lim.v_max, lim.v_max, 12)
    kp_u = _double_to_uint(kp, 0, 500, 12)
    kd_u = _double_to_uint(kd, 0, 5, 12)
    tau_u = _double_to_uint(tau, -lim.t_max, lim.t_max, 12)
    return bytes([
        (q_u >> 8) & 0xFF,
        q_u & 0xFF,
        dq_u >> 4,
        ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
        kp_u & 0xFF,
        kd_u >> 4,
        ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF),
        tau_u & 0xFF,
    ])


def pack_posforce(position: float, speed_rad_s: float,
                  current_limit_pu: float) -> bytes:
    """Pack a Damiao position/force command.

    The last field is a per-unit motor-current limit in the range [0, 1].
    """
    speed_u = int(_clamp(speed_rad_s * 100.0, 0.0, 10000.0))
    current_u = int(_clamp(current_limit_pu * 10000.0, 0.0, 10000.0))
    return struct.pack("<fHH", float(position), speed_u, current_u)


@dataclass
class MotorState:
    position: float = 0.0   # rad
    velocity: float = 0.0   # rad/s
    torque: float = 0.0     # Nm
    t_mos: int = 0
    t_rotor: int = 0
    valid: bool = False


def parse_state(motor_type: MotorType, data: bytes) -> MotorState:
    if len(data) < 8:
        return MotorState()
    lim = MOTOR_LIMITS[motor_type]
    q_u = (data[1] << 8) | data[2]
    dq_u = (data[3] << 4) | (data[4] >> 4)
    tau_u = ((data[4] & 0xF) << 8) | data[5]
    return MotorState(
        position=_uint_to_double(q_u, -lim.p_max, lim.p_max, 16),
        velocity=_uint_to_double(dq_u, -lim.v_max, lim.v_max, 12),
        torque=_uint_to_double(tau_u, -lim.t_max, lim.t_max, 12),
        t_mos=data[6],
        t_rotor=data[7],
        valid=True,
    )


# ── CAN Socket (send classical, recv FD) ────────────────────────────

def _make_bus(interface: str):
    """Factory: pick the right CAN backend based on interface name prefix."""
    if interface.startswith("waveshare:"):
        try:
            from .waveshare_can import WaveshareCANBus
        except ImportError:
            from waveshare_can import WaveshareCANBus
        return WaveshareCANBus(interface)
    return CANBus(interface)


class CANBus:
    """Raw CAN socket that sends 16-byte can_frame and receives on FD socket."""

    def __init__(self, interface: str = "can0"):
        self._sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        # Enable FD frames for receiving (motors respond with FD framing)
        self._sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        self._sock.bind((interface,))
        self._sock.settimeout(0.001)  # 1ms default timeout for recv

    def send(self, can_id: int, data: bytes):
        """Send a classical CAN frame (16 bytes)."""
        frame = struct.pack('=IB3x8s', can_id, len(data), data.ljust(8, b'\x00'))
        self._sock.send(frame)

    def recv(self, timeout_s: float = 0.005) -> Optional[Tuple[int, bytes]]:
        """Receive a frame. Returns (can_id, data) or None."""
        self._sock.settimeout(timeout_s)
        try:
            data = self._sock.recv(72)
            can_id = struct.unpack_from('=I', data, 0)[0] & 0x1FFFFFFF
            dlc = data[4]
            payload = data[8:8 + min(dlc, 8)]
            return can_id, payload
        except socket.timeout:
            return None

    def recv_all(self, timeout_s: float = 0.01) -> List[Tuple[int, bytes]]:
        """Receive all pending frames within timeout."""
        frames = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            result = self.recv(timeout_s=min(remaining, 0.002))
            if result is None:
                break
            frames.append(result)
        return frames

    def close(self):
        self._sock.close()


# ── Arm Controller ───────────────────────────────────────────────────

import os as _os
CTRL_MODE = _os.environ.get("OPENARM_CTRL_MODE", "mit").lower()
POSVEL_VEL = float(_os.environ.get("OPENARM_VEL", "5.0"))


class OpenArmCAN:
    """Direct CAN controller for one OpenArm (7 joints + gripper)."""

    def __init__(self, interface: str = "can0", gripper_posforce: bool = False,
                 gripper_speed: float = 5.0,
                 gripper_current_limit: float = 0.15):
        self.bus = _make_bus(interface)
        self._motor_types = ARM_MOTOR_TYPES
        self._send_ids = ARM_SEND_IDS
        self._recv_ids = ARM_RECV_IDS
        self._gripper_posforce = gripper_posforce
        self._gripper_speed = _clamp(gripper_speed, 0.0, 100.0)
        self._gripper_current_limit = _clamp(
            gripper_current_limit, 0.0, 1.0)
        self._recv_to_idx: Dict[int, int] = {
            rid: i for i, rid in enumerate(ARM_RECV_IDS)
        }
        self._recv_to_idx[GRIPPER_RECV_ID] = 7  # gripper is index 7

        # Current motor states (7 joints + 1 gripper)
        self.states: List[MotorState] = [MotorState() for _ in range(8)]

        if self._gripper_posforce:
            self._configure_gripper_posforce()

    def _configure_gripper_posforce(self):
        """Put only the gripper motor in POS_FORCE mode and verify it."""
        # Damiao parameter write: motor ID, 0x55, CTRL_MODE RID 10,
        # POS_FORCE value 4. This is the format used by openarm_can.
        data = bytes([
            GRIPPER_SEND_ID, 0x00, 0x55, 0x0A,
            0x04, 0x00, 0x00, 0x00,
        ])
        self.bus.send(0x7FF, data)
        time.sleep(0.01)
        frames = self.bus.recv_all(timeout_s=0.05)
        confirmed = any(
            can_id == GRIPPER_RECV_ID
            and len(payload) >= 8
            and payload[2] == 0x55
            and payload[3] == 0x0A
            and int.from_bytes(payload[4:8], "little") == 4
            for can_id, payload in frames
        )
        if not confirmed:
            raise RuntimeError(
                "Gripper motor did not confirm POS_FORCE control mode")

    def enable_all(self):
        """Enable all motors."""
        data = pack_enable()
        for sid in self._send_ids:
            self.bus.send(sid, data)
            time.sleep(0.001)
        self.bus.send(GRIPPER_SEND_ID, data)
        time.sleep(0.01)
        self._process_responses()

    def disable_all(self):
        """Disable all motors."""
        data = pack_disable()
        for sid in self._send_ids:
            self.bus.send(sid, data)
            time.sleep(0.001)
        self.bus.send(GRIPPER_SEND_ID, data)
        time.sleep(0.01)
        self._process_responses()

    def set_joint_positions(self, positions: np.ndarray, kp: float = 30.0,
                            kd: float = 1.0, process_responses: bool = True):
        """Send MIT position commands to all 7 arm joints.

        Args:
            positions: array of 7 joint positions in radians
            kp: position gain (0-500)
            kd: damping gain (0-5)
            process_responses: if False, skip blocking recv (for tight loops)
        """
        for i in range(7):
            if CTRL_MODE == "posvel":
                pv = struct.pack("<ff", float(positions[i]), POSVEL_VEL)
                self.bus.send(0x100 + self._send_ids[i], pv)
            else:
                data = pack_mit(self._motor_types[i], kp=kp, kd=kd, q=positions[i])
                self.bus.send(self._send_ids[i], data)
        if process_responses:
            self._process_responses()

    def set_gripper(self, position: float, kp: float = 30.0, kd: float = 1.0):
        """Send position command to gripper motor."""
        if self._gripper_posforce:
            data = pack_posforce(
                position, self._gripper_speed, self._gripper_current_limit)
            self.bus.send(0x300 + GRIPPER_SEND_ID, data)
        elif CTRL_MODE == "posvel":
            data = struct.pack("<ff", float(position), POSVEL_VEL)
            self.bus.send(0x100 + GRIPPER_SEND_ID, data)
        else:
            data = pack_mit(GRIPPER_MOTOR_TYPE, kp=kp, kd=kd, q=position)
            self.bus.send(GRIPPER_SEND_ID, data)
        time.sleep(0.001)
        self._process_responses()

    def get_positions(self) -> np.ndarray:
        """Return current joint positions (7 arm joints)."""
        return np.array([s.position for s in self.states[:7]])

    def get_gripper_position(self) -> float:
        return self.states[7].position

    def refresh(self):
        """Query current state of all motors."""
        for sid in self._send_ids:
            # Send refresh command (read state)
            data = bytes([sid & 0xFF, (sid >> 8) & 0xFF, 0xCC,
                          0x00, 0x00, 0x00, 0x00, 0x00])
            self.bus.send(0x7FF, data)
            time.sleep(0.001)
        # Gripper
        data = bytes([GRIPPER_SEND_ID & 0xFF, 0x00, 0xCC,
                      0x00, 0x00, 0x00, 0x00, 0x00])
        self.bus.send(0x7FF, data)
        time.sleep(0.01)
        self._process_responses()

    def _process_responses(self, timeout_s: float = 0.005):
        """Read all pending CAN frames and update motor states."""
        frames = self.bus.recv_all(timeout_s=timeout_s)
        for can_id, data in frames:
            if can_id in self._recv_to_idx:
                idx = self._recv_to_idx[can_id]
                if idx < 7:
                    mt = self._motor_types[idx]
                else:
                    mt = GRIPPER_MOTOR_TYPE
                self.states[idx] = parse_state(mt, data)

    def close(self):
        try:
            self.disable_all()
        finally:
            self.bus.close()


def test_connection(interface: str = "can0") -> bool:
    """Quick connectivity test — enable and read back all motors."""
    print(f"Testing CAN connection on {interface}...")
    arm = OpenArmCAN(interface)
    arm.enable_all()
    time.sleep(0.05)
    arm.refresh()

    all_ok = True
    for i in range(7):
        s = arm.states[i]
        status = "OK" if s.valid else "NG"
        if not s.valid:
            all_ok = False
        print(f"  Joint {i+1} (0x{ARM_SEND_IDS[i]:02x}): {status}"
              f"  pos={s.position:.3f} rad" if s.valid else
              f"  Joint {i+1} (0x{ARM_SEND_IDS[i]:02x}): {status}")

    s = arm.states[7]
    status = "OK" if s.valid else "NG"
    if not s.valid:
        all_ok = False
    print(f"  Gripper (0x{GRIPPER_SEND_ID:02x}): {status}"
          f"  pos={s.position:.3f} rad" if s.valid else
          f"  Gripper (0x{GRIPPER_SEND_ID:02x}): {status}")

    arm.disable_all()
    arm.close()
    print(f"\n{'ALL OK' if all_ok else 'SOME MOTORS FAILED'}")
    return all_ok


if __name__ == "__main__":
    import sys
    iface = sys.argv[1] if len(sys.argv) > 1 else "can0"
    test_connection(iface)
