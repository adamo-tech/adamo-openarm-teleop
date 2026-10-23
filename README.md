# OpenArm teleoperation with Adamo

The deployed OpenArm `can_vr_teleop_mid.py`, migrated to the Rust-backed Adamo
Python SDK. It retains the weighted damped least-squares IK, orientation tracking,
One Euro filtering, CAN commands and mid tuning from the Orin.

## Install and check

Requires Linux and Python 3.10 or newer. Python 3.12 is used for local validation.

```sh
git clone https://github.com/adamo-tech/adamo-openarm-teleop.git
cd adamo-openarm-teleop
uv sync --locked --python 3.12
uv run openarm-teleop --check
```

`--check` loads the bundled URDF and meshes and exercises FK, Jacobians and IK
for both arms. It does not authenticate, connect to a robot, or open CAN.

The project pins Placo to the Orin's version, 0.9.20, and its compatible URDF
libraries. `uv.lock` fixes the complete dependency set, including Adamo 1.0.15.

## Run on the robot computer

Set `ADAMO_API_KEY` to your organization API key. The SDK handles authentication
and connection setup; credentials are not embedded in the source.

For the Orin's dual-channel Waveshare adapter:

```sh
uv run openarm-teleop --robot-name orin \
  --right-can waveshare:0 --left-can waveshare:1 --side both
```

For configured SocketCAN interfaces:

```sh
uv run openarm-teleop --robot-name orin --right-can can0 --left-can can1
```

The original module invocation also works:
`uv run python -m openarm_vr_teleop.can_vr_teleop_mid`.
Use `--help` for gains, velocity limits, control rate and other options.

The application authenticates and checks that all seven motors on each enabled
arm respond, then moves the arms to the original bent-elbow home position and
opens the grippers. Startup homing and a B-button return home do not require the
grips to be held. Normal controller-driven arm motion does.

## Controls

| Input | Behaviour |
| --- | --- |
| Hold **both inner grips** | Enable relative arm movement from the current hand poses |
| Release **either grip** | Pause both arms; reposition your hands before gripping again |
| Triggers | Open/close the corresponding gripper, independently of the movement clutch |
| Right controller **B** | Smoothly return the enabled arms home and rebase the IK |
| Terminal **R** | Refresh joint state and reset IK calibration |
| **Ctrl+C** | Disable motors and close CAN and SDK resources |

Both grips are required even with `--side left` or `--side right`, matching the
deployed controller. Position scaling is 1.0, joint speed limit 6.5 rad/s, motor
gains `kp=35`, `kd=1`, and the default control rate is 60 Hz.

Missing poses or button data pause movement after 0.25 seconds (configurable with
`--input-timeout`). Missing controllers in a full XR frame release immediately.
Stale tracking stops gripper updates too. Returning tracking or an XR resume
event rebases the movement reference before following the hands again. Pausing
stops new targets; it does not disable the motors or act as an emergency stop.

## SDK integration and files

- `openarm_vr_teleop/can_vr_teleop_mid.py`: IK, clutch, homing and CAN control loop.
- `openarm_vr_teleop/vr_input.py`: `adamo.xr.subscribe_xr_control(robot, ...)`,
  typed XR input, fresh snapshots and tracking timeout.
- `openarm_vr_teleop/kinematics.py`: Placo FK and Jacobians.
- `openarm_vr_teleop/can_motor.py`: original Damiao motor protocol and SocketCAN.
- `openarm_vr_teleop/waveshare_can.py`: Waveshare USB-CAN backend.
- `openarm_vr_teleop/urdf/`, `meshes/`: bundled model, with relative mesh paths.

`adamo.Robot` owns authentication and subscriptions. Its control callbacks run
on SDK threads while `adamo.Rate` paces the motor loop. The SDK decodes full
`XrFrame` messages and separate pose/joy streams, including legacy CDR tracking;
there is no application-level CDR parser. Status transitions also appear in the
robot's Adamo console. This project provides arm control; camera streaming can
continue in the existing separate process.

For Waveshare, install the vendor's
[USB-CAN-FD-B Linux SDK](https://www.waveshare.com/wiki/USB-CAN-FD#Linux_Tutorial)
and select its `libcontrolcanfd.so` for your architecture (ARM64 on Jetson):

```sh
export WAVESHARE_CAN_LIBRARY=/absolute/path/to/libcontrolcanfd.so
```

The vendor binary is not distributed in this repository or Python package.
SocketCAN does not require it. USB access and CAN interface setup remain host
prerequisites.

## Validation

```sh
uv run pytest -q
uv build
```

Tests use the real Adamo wheel with a local transport, real Placo kinematics,
and fake motor endpoints. They cover grip release/re-engagement, XR resume,
tracking timeouts, independent triggers, legacy CDR decoding, motor startup
failure cleanup, joint limits and velocity limits. No robot hardware is used.

Ported on 2026-10-22 from the deployed OpenArm `can_vr_teleop_mid.py` controller.
The solver, filtering, coordinate transforms and tuning were checked against
the deployed source. This migration has not been tested with physical arms.
The legacy XR test fixtures came from the
[Adamo SDK](https://github.com/adamo-tech/adamo)'s
`tests/fixtures/legacy-xr-cdr.json`.

## License

[Apache License 2.0](LICENSE). The OpenArm model and meshes originate from
[Enactic's openarm_description](https://github.com/enactic/openarm_description).
See [NOTICE](NOTICE) for attribution and modification details.
