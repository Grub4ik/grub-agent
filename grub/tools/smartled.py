"""In-memory SmartLED discovery state and fixed mode commands."""

from __future__ import annotations

import ipaddress
import subprocess
from typing import Any

from .discover_smartled import discover_smartled
from .smartled_audio import (
    SMARTLED_MUSIC_SCHEMA,
    SMARTLED_SPECTRUM_SCHEMA,
    SMARTLED_STOP_AUDIO_SCHEMA,
    SmartLEDAudioModes,
)


SMARTLED_STATIC_SCHEMA = {
    "type": "object",
    "properties": {
        "r": {"type": "integer", "minimum": 0, "maximum": 255},
        "g": {"type": "integer", "minimum": 0, "maximum": 255},
        "b": {"type": "integer", "minimum": 0, "maximum": 255},
    },
    "required": ["r", "g", "b"],
    "additionalProperties": False,
}
SMARTLED_RAINBOW_SCHEMA = {
    "type": "object",
    "properties": {
        "interval": {"type": "integer", "minimum": 100, "maximum": 60000},
    },
    "required": ["interval"],
    "additionalProperties": False,
}

_NETWORK = ipaddress.ip_network("192.168.0.0/24")
_PORT = 4210
_NC_COMMAND = "nc"
_CONNECT_TIMEOUT_SECONDS = 2
_PROCESS_TIMEOUT_SECONDS = 4.0
_MAX_RESPONSE_CHARS = 512


class SmartLEDCommandError(RuntimeError):
    """Raised when Grub cannot safely send the fixed SmartLED command."""


class SmartLEDTools:
    """Keep the last discovered address in memory for this Grub process."""

    def __init__(self) -> None:
        self._ip: str | None = None
        self._audio_modes = SmartLEDAudioModes()

    def discover(self) -> dict[str, Any]:
        """Run discovery, clearing stale state and storing a valid found IP."""
        self._ensure_audio_stopped()
        self._ip = None
        result = discover_smartled()
        if result.get("status") == "not_found":
            return result
        if result.get("status") != "found":
            raise SmartLEDCommandError("Discovery returned an unrecognized status.")

        raw_ip = result.get("ip")
        try:
            address = ipaddress.ip_address(raw_ip)
        except (TypeError, ValueError) as exc:
            raise SmartLEDCommandError("Discovery returned an invalid IP address.") from exc
        if (
            not isinstance(address, ipaddress.IPv4Address)
            or address not in _NETWORK
            or address in {_NETWORK.network_address, _NETWORK.broadcast_address}
        ):
            raise SmartLEDCommandError("Discovery returned an unusable host address in 192.168.0.0/24.")

        self._ip = str(address)
        return {"status": "found", "ip": self._ip, "port": _PORT}

    def set_static(self, r: int, g: int, b: int) -> dict[str, Any]:
        """Send validated RGB values to the address found in this session."""
        self._ensure_audio_stopped()
        for channel, value in (("r", r), ("g", g), ("b", b)):
            # bool is a subclass of int, but is not a valid RGB channel.
            if type(value) is not int or not 0 <= value <= 255:
                raise SmartLEDCommandError(
                    f"RGB channel {channel} must be an integer from 0 to 255."
                )
        command_bytes = f"MODE STATIC {r} {g} {b}\n".encode("ascii")
        return self._send_command(command_bytes, "static", {"rgb": {"r": r, "g": g, "b": b}})

    def set_rainbow(self, interval: int) -> dict[str, Any]:
        """Set rainbow interval in milliseconds on the discovered device."""
        self._ensure_audio_stopped()
        if type(interval) is not int or not 100 <= interval <= 60000:
            raise SmartLEDCommandError("Rainbow interval must be an integer from 100 to 60000 milliseconds.")
        command = f"MODE RAINBOW INTERVAL {interval}\n".encode("ascii")
        return self._send_command(command, "rainbow", {"interval": interval})

    def start_music(self, **settings: Any) -> dict[str, Any]:
        """Start system-audio-reactive Music mode in a background worker."""
        return self._audio_modes.start_music(self._require_ip(), **settings)

    def start_spectrum(self, **settings: Any) -> dict[str, Any]:
        """Start FFT Spectrum mode in a background worker."""
        return self._audio_modes.start_spectrum(self._require_ip(), **settings)

    def stop_audio_mode(self) -> dict[str, Any]:
        """Stop the currently active Music or Spectrum audio stream."""
        return self._audio_modes.stop()

    def close(self) -> None:
        """Stop the background audio mode during Grub shutdown."""
        self._audio_modes.close()

    def _ensure_audio_stopped(self) -> None:
        if self._audio_modes.is_running:
            raise SmartLEDCommandError(
                "A SmartLED audio mode is running; call smartled_stop_audio_mode before changing device mode."
            )

    def _require_ip(self) -> str:
        if self._ip is None:
            raise SmartLEDCommandError(
                "SmartLED has not been discovered in this Grub session; run discover_smartled first."
            )
        try:
            address = ipaddress.ip_address(self._ip)
        except ValueError as exc:
            self._ip = None
            raise SmartLEDCommandError("The stored SmartLED IP address is invalid.") from exc
        if (
            not isinstance(address, ipaddress.IPv4Address)
            or address not in _NETWORK
            or address in {_NETWORK.network_address, _NETWORK.broadcast_address}
        ):
            self._ip = None
            raise SmartLEDCommandError("The stored SmartLED IP is not a usable host in 192.168.0.0/24.")
        return str(address)

    def _send_command(
        self,
        command_bytes: bytes,
        mode: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one application-defined command; model input never supplies it."""
        address = self._require_ip()

        command = [
            _NC_COMMAND,
            "-4",
            "-N",
            "-w",
            str(_CONNECT_TIMEOUT_SECONDS),
            address,
            str(_PORT),
        ]
        try:
            completed = subprocess.run(
                command,
                input=command_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_PROCESS_TIMEOUT_SECONDS,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SmartLEDCommandError(
                "Netcat timed out; the command may have reached the device, but delivery and execution are unknown."
            ) from exc
        except OSError as exc:
            raise SmartLEDCommandError(
                f"Could not run Netcat ({type(exc).__name__}: {exc}); no execution is confirmed."
            ) from exc

        response = _decode_output(completed.stdout)
        if completed.returncode != 0:
            detail = _decode_output(completed.stderr)
            suffix = f" Netcat said: {detail}" if detail else ""
            raise SmartLEDCommandError(
                f"Netcat exited with status {completed.returncode}; the command may have reached "
                f"the device, but execution is unknown.{suffix}"
            )

        # No acknowledgement format is specified for this command. A zero
        # Netcat exit status only describes the local client process.
        result = {
            "status": "sent_unconfirmed",
            "ip": address,
            "port": _PORT,
            "mode": mode,
            "command": command_bytes.decode("ascii").rstrip("\n"),
            "execution_confirmed": False,
            "response_received": bool(response),
            "response": response or None,
        }
        if details:
            result.update(details)
        return result


def _decode_output(value: bytes | str | None) -> str:
    """Decode and bound Netcat output before returning it to the agent."""
    if not value:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text.strip()[:_MAX_RESPONSE_CHARS]
