"""Waveshare USB-CAN-FD-B backend for OpenArm CAN motor control.

Exposes WaveshareCANBus with the same .send / .recv / .recv_all / .close
interface as can_motor.CANBus, so it's a drop-in replacement when the user
passes an interface name starting with 'waveshare:' (e.g. waveshare:0 = ch0,
waveshare:1 = ch1).

The Waveshare device is a single dual-channel adapter; a process-wide
singleton owns the device handle and reference-counts channel users.
"""

from __future__ import annotations

import threading
import os
from pathlib import Path
from ctypes import (
    CDLL, Structure, Union, byref,
    c_void_p, c_ulong, c_long, c_uint, c_ubyte, c_ushort, c_ulonglong,
)
from typing import List, Optional, Tuple

_LIB_PATH = os.environ.get(
    "WAVESHARE_CAN_LIBRARY",
    str(Path(__file__).parent / "waveshare" / "libcontrolcanfd.so"),
)

VCI_USBCAN2 = 41
STATUS_OK = 1
TYPE_CAN = 0
TYPE_CANFD = 1


class _CANInit(Structure):
    _fields_ = [("acc_code", c_uint), ("acc_mask", c_uint),
                ("reserved", c_uint), ("filter", c_ubyte),
                ("timing0", c_ubyte), ("timing1", c_ubyte),
                ("mode", c_ubyte)]


class _CANFDInit(Structure):
    _fields_ = [("acc_code", c_uint), ("acc_mask", c_uint),
                ("abit_timing", c_uint), ("dbit_timing", c_uint),
                ("brp", c_uint), ("filter", c_ubyte),
                ("mode", c_ubyte), ("pad", c_ushort), ("reserved", c_uint)]


class _Cfg(Union):
    _fields_ = [("can", _CANInit), ("canfd", _CANFDInit)]


class _ChannelCfg(Structure):
    _fields_ = [("can_type", c_uint), ("config", _Cfg)]


class _CanFrame(Structure):
    _fields_ = [
        ("can_id", c_uint, 29),
        ("err",    c_uint, 1),
        ("rtr",    c_uint, 1),
        ("eff",    c_uint, 1),
        ("can_dlc", c_ubyte),
        ("__pad",   c_ubyte),
        ("__res0",  c_ubyte),
        ("__res1",  c_ubyte),
        ("data",    c_ubyte * 8),
    ]


class _TransmitData(Structure):
    _fields_ = [("frame", _CanFrame), ("transmit_type", c_uint)]


class _ReceiveData(Structure):
    _fields_ = [("frame", _CanFrame), ("timestamp", c_ulonglong)]


class _CanFDFrame(Structure):
    _fields_ = [
        ("can_id", c_uint, 29),
        ("err",    c_uint, 1),
        ("rtr",    c_uint, 1),
        ("eff",    c_uint, 1),
        ("len",    c_ubyte),
        ("brs",    c_ubyte, 1),
        ("esi",    c_ubyte, 1),
        ("__res",  c_ubyte, 6),
        ("__res0", c_ubyte),
        ("__res1", c_ubyte),
        ("data",   c_ubyte * 64),
    ]


class _ReceiveFDData(Structure):
    _fields_ = [("frame", _CanFDFrame), ("timestamp", c_ulonglong)]


def _load():
    try:
        lib = CDLL(_LIB_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load Waveshare CAN library {_LIB_PATH}: {exc}. "
            "Install the Waveshare USB-CAN-FD-B Linux SDK and set "
            "WAVESHARE_CAN_LIBRARY to its libcontrolcanfd.so for your platform, "
            "or select a SocketCAN interface such as can0."
        ) from exc
    lib.ZCAN_OpenDevice.restype = c_void_p
    lib.ZCAN_CloseDevice.argtypes = (c_void_p,)
    lib.ZCAN_SetAbitBaud.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_SetDbitBaud.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_SetCANFDStandard.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_SetResistanceEnable.argtypes = (c_void_p, c_ulong, c_ulong)
    lib.ZCAN_InitCAN.argtypes = (c_void_p, c_ulong, c_void_p)
    lib.ZCAN_InitCAN.restype = c_void_p
    lib.ZCAN_StartCAN.argtypes = (c_void_p,)
    lib.ZCAN_ResetCAN.argtypes = (c_void_p,)
    lib.ZCAN_Transmit.argtypes = (c_void_p, c_void_p, c_ulong)
    lib.ZCAN_Receive.argtypes = (c_void_p, c_void_p, c_ulong, c_long)
    lib.ZCAN_ReceiveFD.argtypes = (c_void_p, c_void_p, c_ulong, c_long)
    lib.ZCAN_GetReceiveNum.argtypes = (c_void_p, c_ulong)
    lib.ZCAN_ClearFilter.argtypes = (c_void_p,)
    lib.ZCAN_AckFilter.argtypes = (c_void_p,)
    lib.ZCAN_ClearBuffer.argtypes = (c_void_p,)
    return lib


_lib = _load()


# ── Device singleton ─────────────────────────────────────────────────

class _Device:
    _instance: Optional["_Device"] = None
    _lock = threading.Lock()

    def __init__(self, abit: int = 1_000_000, dbit: int = 5_000_000,
                 enable_terminator: bool = True):
        self.handle = _lib.ZCAN_OpenDevice(VCI_USBCAN2, 0, 0)
        if not self.handle:
            raise RuntimeError(
                "ZCAN_OpenDevice failed — check Waveshare USB cable, and that "
                "the udev rule grants access (or run with sudo).")
        self._channel_handles: dict[int, c_void_p] = {}
        self._refcount = 0
        for ch in (0, 1):
            if _lib.ZCAN_SetAbitBaud(self.handle, ch, abit) != STATUS_OK:
                raise RuntimeError(f"SetAbitBaud failed on channel {ch}")
            if _lib.ZCAN_SetDbitBaud(self.handle, ch, dbit) != STATUS_OK:
                raise RuntimeError(f"SetDbitBaud failed on channel {ch}")
            if _lib.ZCAN_SetCANFDStandard(self.handle, ch, 0) != STATUS_OK:
                raise RuntimeError(f"SetCANFDStandard failed on channel {ch}")
            if enable_terminator:
                _lib.ZCAN_SetResistanceEnable(self.handle, ch, 1)

    @classmethod
    def acquire(cls, channel: int) -> Tuple["_Device", c_void_p]:
        with cls._lock:
            if cls._instance is None:
                cls._instance = _Device()
            dev = cls._instance
            if channel not in dev._channel_handles:
                cfg = _ChannelCfg()
                cfg.can_type = TYPE_CANFD
                cfg.config.canfd.mode = 0
                ch_handle = _lib.ZCAN_InitCAN(dev.handle, channel, byref(cfg))
                if not ch_handle:
                    raise RuntimeError(f"ZCAN_InitCAN failed on channel {channel}")
                _lib.ZCAN_ClearFilter(ch_handle)
                _lib.ZCAN_AckFilter(ch_handle)
                if _lib.ZCAN_StartCAN(ch_handle) != STATUS_OK:
                    raise RuntimeError(f"ZCAN_StartCAN failed on channel {channel}")
                dev._channel_handles[channel] = ch_handle
            dev._refcount += 1
            return dev, dev._channel_handles[channel]

    @classmethod
    def release(cls, channel: int) -> None:
        with cls._lock:
            if cls._instance is None:
                return
            dev = cls._instance
            ch_handle = dev._channel_handles.get(channel)
            if ch_handle is not None:
                _lib.ZCAN_ResetCAN(ch_handle)
                del dev._channel_handles[channel]
            dev._refcount -= 1
            if dev._refcount <= 0:
                _lib.ZCAN_CloseDevice(dev.handle)
                cls._instance = None


# ── Bus class with the same interface as can_motor.CANBus ────────────

class WaveshareCANBus:
    """Drop-in replacement for can_motor.CANBus, backed by libcontrolcanfd."""

    def __init__(self, interface: str = "waveshare:0"):
        if not interface.startswith("waveshare:"):
            raise ValueError(f"WaveshareCANBus expects 'waveshare:N' interface, got {interface!r}")
        try:
            self._channel = int(interface.split(":", 1)[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid Waveshare interface name: {interface!r}")
        if self._channel not in (0, 1):
            raise ValueError(f"Waveshare channel must be 0 or 1, got {self._channel}")
        self._dev, self._ch_handle = _Device.acquire(self._channel)
        self._closed = False

    def send(self, can_id: int, data: bytes) -> None:
        """Send a classical CAN frame (matches can_motor.CANBus.send signature)."""
        msg = _TransmitData()
        msg.transmit_type = 0  # normal send
        f = msg.frame
        f.can_id = can_id & 0x1FFFFFFF
        f.eff = 1 if can_id > 0x7FF else 0
        f.rtr = 0
        f.err = 0
        dlc = min(len(data), 8)
        f.can_dlc = dlc
        for i in range(dlc):
            f.data[i] = data[i]
        n = _lib.ZCAN_Transmit(self._ch_handle, byref(msg), 1)
        if n != 1:
            raise RuntimeError(f"ZCAN_Transmit returned {n} (expected 1)")

    def _drain_queues(self) -> List[Tuple[int, bytes]]:
        """Non-blocking drain of both classical and FD receive queues.

        Damiao motors on a CAN-FD bus reply with FD frames, so we MUST poll
        both queues — only checking TYPE_CAN silently drops every reply.
        """
        frames: List[Tuple[int, bytes]] = []
        # Classical
        pending = _lib.ZCAN_GetReceiveNum(self._ch_handle, TYPE_CAN)
        if pending > 0:
            n = min(pending, 64)
            buf = (_ReceiveData * n)()
            got = _lib.ZCAN_Receive(self._ch_handle, byref(buf), n, 0)
            for i in range(max(0, got)):
                f = buf[i].frame
                payload = bytes(bytearray(f.data)[: f.can_dlc])
                frames.append((int(f.can_id), payload))
        # FD
        pending = _lib.ZCAN_GetReceiveNum(self._ch_handle, TYPE_CANFD)
        if pending > 0:
            n = min(pending, 64)
            buf = (_ReceiveFDData * n)()
            got = _lib.ZCAN_ReceiveFD(self._ch_handle, byref(buf), n, 0)
            for i in range(max(0, got)):
                f = buf[i].frame
                # Truncate to first 8 bytes — matches the existing CANBus
                # behaviour in can_motor.py, which expects 8-byte payloads.
                payload_len = min(int(f.len), 8)
                payload = bytes(bytearray(f.data)[: payload_len])
                frames.append((int(f.can_id), payload))
        return frames

    def recv(self, timeout_s: float = 0.005) -> Optional[Tuple[int, bytes]]:
        """Receive a single frame within timeout, or None on timeout."""
        import time
        deadline = time.monotonic() + timeout_s
        while True:
            frames = self._drain_queues()
            if frames:
                return frames[0]
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)

    def recv_all(self, timeout_s: float = 0.01) -> List[Tuple[int, bytes]]:
        """Drain all pending frames within the timeout window."""
        import time
        deadline = time.monotonic() + timeout_s
        frames: List[Tuple[int, bytes]] = []
        # Initial non-blocking drain
        frames.extend(self._drain_queues())
        # Poll for any new frames until deadline
        while time.monotonic() < deadline:
            new = self._drain_queues()
            if new:
                frames.extend(new)
            else:
                time.sleep(0.0005)
        return frames

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _Device.release(self._channel)
