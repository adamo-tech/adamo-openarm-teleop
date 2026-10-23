import json
import threading

import adamo
import pytest

from openarm_vr_teleop.vr_input import VRInput


def pose(z=0.0):
    return {"position": {"x": 0., "y": 0., "z": z},
            "orientation": {"w": 1., "x": 0., "y": 0., "z": 0.}}


def frame(*, grip=True, left_grip=None, trigger=0.0, b=False, z=0.0, resumed=False):
    def controller(pressed):
        return {"pose": pose(z), "axes": [0., 0., 0., 0., trigger],
                "buttons": [0, int(pressed), 0, 0, 0, int(b)]}
    return {"type": "XrFrame", "head": pose(), "resumed": resumed,
            "left": controller(grip if left_grip is None else left_grip),
            "right": controller(grip)}


class ObservedInput(VRInput):
    def __init__(self, robot, **kwargs):
        self.arrived = threading.Event()
        self.robot = robot
        super().__init__(robot, **kwargs)

    def _on_sample(self, sample):
        super()._on_sample(sample)
        self.arrived.set()

    def publish(self, message, channel="json/xr/frame"):
        self.arrived.clear()
        payload = message if isinstance(message, bytes) else json.dumps(message).encode()
        self.robot.session.put(f"{self.robot.name}/control/{channel}", payload)
        assert self.arrived.wait(3), "SDK did not deliver XR input"


@pytest.fixture
def source():
    # Real wheel, local transport: no API key, cloud, headset or motors.
    with adamo.Robot.local("openarm-test", port=0) as robot:
        clock = [100.0]
        source = ObservedInput(robot, clock=lambda: clock[0])
        yield source, clock
        source.close()
