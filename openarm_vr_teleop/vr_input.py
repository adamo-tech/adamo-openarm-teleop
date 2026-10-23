"""Latest XR state from the Adamo SDK, shared with the CAN control loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np
from adamo import xr

HANDS = ("left", "right")
POSE_CHANNELS = {f"json/xr/{hand}/pose": hand for hand in HANDS}
JOY_CHANNELS = {f"json/xr/{hand}/joy": hand for hand in HANDS}
XR_BUTTON_GRIP = 1
XR_BUTTON_B = 5


@dataclass
class VRControllerState:
    position: np.ndarray | None = None
    orientation: np.ndarray | None = None  # w, x, y, z
    trigger: float = 0.0
    grip_pressed: bool = False
    updated: bool = False
    fresh: bool = False


@dataclass
class VRState:
    left: VRControllerState
    right: VRControllerState
    head_orientation: np.ndarray | None
    home_requested: bool = False
    rebase_requested: bool = False


@dataclass
class _Hand:
    state: VRControllerState = field(default_factory=VRControllerState)
    pose_time: float = float("-inf")
    joy_time: float = float("-inf")


def _pose_arrays(pose):
    p, q = pose.position, pose.orientation
    position = np.array([p.x, p.y, p.z], dtype=np.float64)
    orientation = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
        raise ValueError("XR pose contains non-finite values")
    norm = np.linalg.norm(orientation)
    if norm < 1e-8:
        raise ValueError("XR orientation has zero length")
    return position, orientation / norm


class VRInput:
    """Accept SDK-decoded XrFrame and per-stream messages, including normalized CDR.

    The SDK drops old packets using their transport timestamps. A separate
    monotonic watchdog releases the clutch when packets stop arriving. Poses
    and buttons must both be fresh; pose traffic cannot keep old grips alive.
    """

    def __init__(self, robot, timeout: float = 0.25, *, clock=time.monotonic):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("XR timeout must be a positive finite number")
        self.timeout = timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._hands = {hand: _Hand() for hand in HANDS}
        self._head_orientation = None
        self._head_time = float("-inf")
        self._home_requested = False
        self._b_was_pressed = False
        self._rebase_requested = False
        self._closed = False
        xr.subscribe_xr_control(robot, self._on_sample, max_age=timeout)

    def _invalidate(self, hand):
        self._hands[hand] = _Hand()
        self._rebase_requested = True
        if hand == "right":
            self._home_requested = False
            self._b_was_pressed = False

    def _set_pose(self, hand, pose, now):
        try:
            position, orientation = _pose_arrays(pose)
        except ValueError:
            self._invalidate(hand)
            return
        entry = self._hands[hand]
        if now - entry.pose_time > self.timeout:
            self._rebase_requested = True
        entry.state.position = position
        entry.state.orientation = orientation
        entry.state.updated = True
        entry.pose_time = now

    def _set_joy(self, hand, axes, buttons, now):
        entry = self._hands[hand]
        trigger = axes[4] if len(axes) > 4 else (buttons[0] if buttons else 0.0)
        if not np.isfinite(trigger):
            self._invalidate(hand)
            return
        grip_pressed = len(buttons) > XR_BUTTON_GRIP and bool(buttons[XR_BUTTON_GRIP])
        if (now - entry.joy_time > self.timeout
                or (entry.state.grip_pressed and not grip_pressed)):
            self._rebase_requested = True
        entry.state.trigger = float(np.clip(trigger, 0.0, 1.0))
        entry.state.grip_pressed = grip_pressed
        entry.joy_time = now
        if hand == "right":
            b_pressed = len(buttons) > XR_BUTTON_B and bool(buttons[XR_BUTTON_B])
            if b_pressed and not self._b_was_pressed:
                self._home_requested = True
            self._b_was_pressed = b_pressed

    def _set_head(self, pose, now):
        try:
            _, self._head_orientation = _pose_arrays(pose)
            self._head_time = now
        except ValueError:
            self._head_orientation = None
            self._head_time = float("-inf")

    def _on_sample(self, sample):
        now, message = self._clock(), sample.message
        with self._lock:
            if self._closed:
                return
            if isinstance(message, xr.XrFrame):
                # An omitted controller in a full frame means tracking was lost.
                for hand in HANDS:
                    controller = message.controller(hand)
                    if controller is None:
                        self._invalidate(hand)
                    else:
                        self._set_pose(hand, controller.pose, now)
                        self._set_joy(hand, controller.axes, controller.buttons, now)
                if message.head is not None:
                    self._set_head(message.head, now)
                else:
                    self._head_orientation = None
                    self._head_time = float("-inf")
                self._rebase_requested |= message.resumed
            elif isinstance(message, xr.PoseStamped):
                if sample.channel in POSE_CHANNELS:
                    self._set_pose(POSE_CHANNELS[sample.channel], message, now)
                elif sample.channel == "json/xr/head/pose":
                    self._set_head(message, now)
            elif isinstance(message, xr.Joy) and sample.channel in JOY_CHANNELS:
                self._set_joy(JOY_CHANNELS[sample.channel], message.axes, message.buttons, now)
            elif isinstance(message, xr.Bool) and sample.channel == "json/xr/clutch":
                self._rebase_requested |= message.data

    def read(self) -> VRState:
        """Atomically consume changes; stale or missing controllers are released."""
        now = self._clock()
        with self._lock:
            states = {}
            for hand, entry in self._hands.items():
                fresh = (not self._closed and entry.state.position is not None
                         and now - entry.pose_time <= self.timeout
                         and now - entry.joy_time <= self.timeout)
                if fresh:
                    states[hand] = replace(
                        entry.state, position=entry.state.position.copy(),
                        orientation=entry.state.orientation.copy(), fresh=True)
                else:
                    states[hand] = VRControllerState()
                    # A later packet must rebase even if the control loop missed the gap.
                    self._rebase_requested = True
                entry.state.updated = False
            head = (self._head_orientation.copy()
                    if self._head_orientation is not None and now - self._head_time <= self.timeout
                    else None)
            state = VRState(states["left"], states["right"], head,
                            self._home_requested and states["right"].fresh,
                            self._rebase_requested)
            self._home_requested = False
            self._rebase_requested = False
            return state

    def close(self):
        # Robot.close owns the SDK subscriptions. Ignore callbacks during teardown.
        with self._lock:
            self._closed = True
            for hand in HANDS:
                self._invalidate(hand)
