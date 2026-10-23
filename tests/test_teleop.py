from pathlib import Path
from types import SimpleNamespace

import adamo
import numpy as np
import pytest

from conftest import ObservedInput, frame
from openarm_vr_teleop import can_vr_teleop_mid as teleop
from openarm_vr_teleop.can_motor import OpenArmCAN
from openarm_vr_teleop.kinematics import OpenArmKinematics


@pytest.fixture(scope="module")
def kinematics():
    urdf = Path(teleop.__file__).parent / "urdf" / "openarm_bimanual.urdf"
    return {hand: OpenArmKinematics(str(urdf), f"openarm_{hand}_") for hand in ("left", "right")}


@pytest.mark.parametrize("hand", ["left", "right"])
def test_real_ik_tracks_target_and_respects_limits(kinematics, hand):
    kin = kinematics[hand]
    ik = teleop.ArmIKController(kin, side=hand, max_joint_velocity=6.5)
    home = ik.home_joints.copy()
    origin = kin.forward_kinematics(home)[:3, 3].copy()
    q = home.copy()
    ik.calibrate(q, np.zeros(3), np.array([1., 0., 0., 0.]), 0.)
    for _ in range(20):
        result = ik.compute(np.array([0., 0., -0.02]), 1/60, np.array([1., 0., 0., 0.]))
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result - q)) <= 6.5/60 + 1e-9
        assert np.all(result >= ik.joint_limits[:, 0])
        assert np.all(result <= ik.joint_limits[:, 1])
        q = result
    error = np.linalg.norm(kin.forward_kinematics(q)[:3, 3] - origin - [0.02, 0., 0.])
    assert error < 0.003


class FakeArm:
    def __init__(self, interface):
        self.interface = interface
        self.states = [SimpleNamespace(valid=True) for _ in range(8)]
        self.q = teleop.HOME_JOINTS_RIGHT.copy()
        self.commands = []
        self.grippers = []
        self.closed = False

    def enable_all(self): pass
    def refresh(self): pass
    def _process_responses(self): pass

    def get_positions(self):
        return self.q.copy()

    def set_joint_positions(self, q, **kwargs):
        self.q = q.copy()
        self.commands.append(q.copy())

    def set_gripper(self, q, **kwargs):
        self.grippers.append(q)

    def close(self):
        self.closed = True


def setup_hardware(monkeypatch, kinematics):
    arms = {}
    def make_arm(interface):
        arms[interface] = FakeArm(interface)
        return arms[interface]
    monkeypatch.setattr(teleop, "OpenArmCAN", make_arm)
    monkeypatch.setattr(teleop, "OpenArmKinematics", lambda path, prefix: kinematics[prefix.split('_')[1]])
    monkeypatch.setattr(teleop.time, "sleep", lambda _: None)
    monkeypatch.setattr(teleop, "KeyboardResetListener", lambda: SimpleNamespace(
        consume_reset=lambda: False, close=lambda: None))
    return arms


def test_control_loop_clutch_rebase_grippers_and_cleanup(monkeypatch, kinematics):
    arms = setup_hardware(monkeypatch, kinematics)
    robot = adamo.Robot.local("openarm-loop-test", port=0)
    monkeypatch.setattr(teleop.adamo, "Robot", lambda **kwargs: robot)
    frames = iter([
        frame(grip=False),  # startup: consume pending button edges
        frame(grip=False, trigger=0.2),
        frame(),           # engage at current pose
        frame(z=-0.01),    # follow controller
        frame(left_grip=False, z=1., trigger=0.8),  # release either -> both stop
        frame(z=1.),       # re-engage at new hand pose, no target jump
        frame(z=2., resumed=True),  # opening/closing operator menu -> rebase
        {**frame(), "left": None, "right": None},  # no stale gripper commands
    ])
    clock = [100.]
    counts, positions = [], []
    class ScriptedInput(ObservedInput):
        def __init__(self, robot, **kwargs):
            super().__init__(robot, clock=lambda: clock[0], **kwargs)

        def read(self):
            try:
                message = next(frames)
            except StopIteration:
                raise KeyboardInterrupt
            clock[0] += 1/60
            self.publish(message)
            return super().read()

    class Rate:
        def __init__(self, hz): pass
        def sleep(self):
            counts.append([(len(a.commands), len(a.grippers)) for a in arms.values()])
            positions.append([a.q.copy() for a in arms.values()])

    monkeypatch.setattr(teleop, "VRInput", ScriptedInput)
    monkeypatch.setattr(teleop.adamo, "Rate", Rate)
    assert teleop.main([]) == 0
    assert all(a.closed for a in arms.values())
    assert len(counts) == 7
    for side in range(2):
        # Both arms stop on either grip release; triggers still operate.
        assert counts[3][side][0] == counts[2][side][0]
        assert counts[3][side][1] == counts[2][side][1] + 1
        # Re-engage and resume each rebase at the current commanded joints.
        np.testing.assert_allclose(positions[4][side], positions[3][side], atol=0.003)
        np.testing.assert_allclose(positions[5][side], positions[4][side], atol=0.003)
        # Missing tracking sends neither arm nor gripper commands.
        assert counts[6][side] == counts[5][side]


def test_partial_motor_startup_closes_both_arms(monkeypatch, kinematics):
    arms = setup_hardware(monkeypatch, kinematics)
    make_arm = teleop.OpenArmCAN
    def fail_left(interface):
        arm = make_arm(interface)
        if interface == "can1":
            arm.states[3].valid = False
        return arm
    robot = adamo.Robot.local("openarm-startup-test", port=0)
    monkeypatch.setattr(teleop.adamo, "Robot", lambda **kwargs: robot)
    monkeypatch.setattr(teleop, "OpenArmCAN", fail_left)
    assert teleop.main([]) == 1
    assert len(arms) == 2 and all(a.closed for a in arms.values())
    assert all(not a.commands for a in arms.values())


def test_bus_closes_even_when_motor_disable_fails():
    arm = OpenArmCAN.__new__(OpenArmCAN)
    closed = []
    arm.bus = SimpleNamespace(close=lambda: closed.append(True))
    def fail():
        raise OSError("CAN disconnected")
    arm.disable_all = fail
    with pytest.raises(OSError, match="CAN disconnected"):
        arm.close()
    assert closed == [True]
