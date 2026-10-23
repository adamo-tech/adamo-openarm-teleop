import json
from pathlib import Path

import numpy as np
import pytest

from conftest import frame, pose


def test_full_frame_and_grip_release(source):
    source, _ = source
    source.publish(frame(trigger=0.7))
    state = source.read()
    assert state.left.grip_pressed and state.right.grip_pressed
    assert state.left.fresh and state.right.fresh
    assert state.right.trigger == pytest.approx(0.7)
    np.testing.assert_equal(state.head_orientation, [1., 0., 0., 0.])
    assert state.left.updated
    assert not source.read().left.updated
    source.publish(frame(left_grip=False))
    state = source.read()
    assert not state.left.grip_pressed and state.right.grip_pressed
    assert state.rebase_requested


def test_tracking_timeout_releases_both_and_stops_gripper_updates(source):
    source, clock = source
    source.publish(frame(trigger=1))
    source.read()
    clock[0] += 0.251
    state = source.read()
    for hand in (state.left, state.right):
        assert not hand.fresh and not hand.grip_pressed and not hand.updated
        assert hand.position is None
    assert state.head_orientation is None
    assert not state.home_requested
    source.publish(frame(z=1))
    assert source.read().rebase_requested


def test_pose_updates_do_not_keep_old_buttons_alive(source):
    source, clock = source
    source.publish({"type": "PoseStamped", **pose()}, "json/xr/left/pose")
    source.publish({"type": "Joy", "axes": [0]*4 + [0.4], "buttons": [0, 1]},
                   "json/xr/left/joy")
    assert source.read().left.grip_pressed
    clock[0] += 0.251
    source.publish({"type": "PoseStamped", **pose(1)}, "json/xr/left/pose")
    assert not source.read().left.fresh
    source.publish({"type": "Joy", "axes": [], "buttons": [1, 1]}, "json/xr/left/joy")
    assert source.read().left.trigger == 1


def test_missing_controller_in_frame_releases_immediately(source):
    source, _ = source
    source.publish(frame())
    source.read()
    source.publish({**frame(), "left": None})
    state = source.read()
    assert state.left.position is None and not state.left.grip_pressed
    assert state.right.fresh and state.rebase_requested


def test_resume_or_tracking_gap_between_reads_requests_rebase(source):
    source, clock = source
    source.publish(frame())
    source.read()
    source.publish(frame(resumed=True, z=2))
    assert source.read().rebase_requested
    clock[0] += 0.3
    source.publish(frame(z=3))
    assert source.read().rebase_requested
    source.publish({"type": "Bool", "data": True}, "json/xr/clutch")
    assert source.read().rebase_requested


def test_release_and_repress_between_control_ticks_is_not_lost(source):
    source, _ = source
    source.publish(frame())
    source.read()
    source.publish(frame(left_grip=False))
    source.publish(frame(z=1))
    state = source.read()
    assert state.left.grip_pressed and state.rebase_requested


def test_b_requests_home_once_per_press_and_expires(source):
    source, clock = source
    source.publish(frame(b=True))
    assert source.read().home_requested
    source.publish(frame(b=True))
    assert not source.read().home_requested
    source.publish(frame(b=False))
    source.publish(frame(b=True))
    clock[0] += 0.3
    assert not source.read().home_requested


def test_invalid_pose_and_zero_quaternion_are_not_used(source):
    source, _ = source
    bad = frame()
    bad["left"]["pose"]["orientation"] = dict.fromkeys(("w", "x", "y", "z"), 0.)
    source.publish(bad)
    assert not source.read().left.fresh


def test_legacy_cdr_is_decoded_by_sdk(source):
    source, _ = source
    fixtures = json.loads((Path(__file__).parent / "legacy-xr-cdr.json").read_text())
    for topic in ("/head_pose", "/controller/left", "/controller/left/joy"):
        packet = next(f for f in fixtures if f["topic"] == topic)
        source.publish(bytes.fromhex(packet["hex"]), "cdr/xr_tracking")
    state = source.read()
    np.testing.assert_allclose(state.left.position, [0.125, 1.5, -0.25])
    assert state.left.grip_pressed and state.left.fresh
    np.testing.assert_equal(state.head_orientation, [1., 0., 0., 0.])
