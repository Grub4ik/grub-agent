"""Find a SmartLED responder using its read-only TCP discovery command."""

from __future__ import annotations

import errno
import ipaddress
import json
import math
import socket
import time
from codecs import getincrementaldecoder
from typing import Any


SMARTLED_DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_NETWORK = ipaddress.ip_network("192.168.0.0/24")
_PORT = 4210
_COMMAND = b"HELP\n"
_DEVICE_NAME = "SmartLED-ESP32"
_DEFAULT_CONNECT_TIMEOUT = 0.1
_DEFAULT_RESPONSE_TIMEOUT = 0.2
_DEFAULT_MAX_RESPONSE_BYTES = 4096
_READ_CHUNK_BYTES = 1024
_FATAL_SOCKET_ERRNOS = {
    errno.EAFNOSUPPORT,
    errno.EMFILE,
    errno.ENETDOWN,
    errno.ENETUNREACH,
    errno.ENFILE,
    errno.ENOBUFS,
    errno.ENOMEM,
    errno.EPROTONOSUPPORT,
}


class SmartLEDDiscoveryError(RuntimeError):
    """Raised when local configuration or a system error prevents discovery."""


class _SmartLEDProtocolError(ValueError):
    """Raised when a peer does not return one bounded UTF-8 JSON document."""


def _read_json_response(connection: Any, max_response_bytes: int, deadline: float) -> Any:
    """Read and decode one JSON document without waiting for peer shutdown."""
    utf8_decoder = getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    text = ""
    received_bytes = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("Timed out while waiting for the SmartLED response.")
        connection.settimeout(remaining)
        chunk = connection.recv(min(_READ_CHUNK_BYTES, max_response_bytes - received_bytes + 1))

        if not chunk:
            if received_bytes == 0:
                raise _SmartLEDProtocolError("The peer returned an empty response.")
            try:
                text += utf8_decoder.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                raise _SmartLEDProtocolError("The response is not valid UTF-8.") from exc
            try:
                document_text = text.lstrip(" \t\r\n")
                document, end = json_decoder.raw_decode(document_text)
            except json.JSONDecodeError as exc:
                raise _SmartLEDProtocolError("The response ended before a complete JSON document.") from exc
            if document_text[end:].strip(" \t\r\n"):
                raise _SmartLEDProtocolError("Unexpected data follows the JSON document.")
            return document

        received_bytes += len(chunk)
        if received_bytes > max_response_bytes:
            raise _SmartLEDProtocolError(
                f"The response exceeded the {max_response_bytes}-byte limit."
            )
        try:
            text += utf8_decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise _SmartLEDProtocolError("The response is not valid UTF-8.") from exc

        document_text = text.lstrip(" \t\r\n")
        try:
            document, end = json_decoder.raw_decode(document_text)
        except json.JSONDecodeError:
            # A TCP recv may end in the middle of a JSON value. Keep reading
            # until it becomes complete, the peer closes, or the deadline expires.
            continue
        if document_text[end:].strip(" \t\r\n"):
            raise _SmartLEDProtocolError("Unexpected data follows the JSON document.")
        if utf8_decoder.getstate()[0]:
            raise _SmartLEDProtocolError("The response ends with an incomplete UTF-8 character.")
        return document


def discover_smartled(
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
    response_timeout: float = _DEFAULT_RESPONSE_TIMEOUT,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
) -> dict[str, str | int | None]:
    """Search 192.168.0.1–192.168.0.254 for the exact SmartLED device name."""
    if isinstance(connect_timeout, bool) or isinstance(response_timeout, bool):
        raise ValueError("Timeouts must be finite positive numbers.")
    try:
        connect_timeout = float(connect_timeout)
        response_timeout = float(response_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timeouts must be finite positive numbers.") from exc
    if not math.isfinite(connect_timeout) or connect_timeout <= 0:
        raise ValueError("connect_timeout must be a finite positive number.")
    if not math.isfinite(response_timeout) or response_timeout <= 0:
        raise ValueError("response_timeout must be a finite positive number.")
    if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 1:
        raise ValueError("max_response_bytes must be a positive integer.")

    for address in _NETWORK.hosts():
        ip = str(address)
        try:
            with socket.create_connection((ip, _PORT), timeout=connect_timeout) as connection:
                deadline = time.monotonic() + response_timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise socket.timeout("Response deadline expired before sending the command.")
                connection.settimeout(remaining)
                connection.sendall(_COMMAND)
                response = _read_json_response(connection, max_response_bytes, deadline)
        except OSError as exc:
            if exc.errno in _FATAL_SOCKET_ERRNOS:
                raise SmartLEDDiscoveryError(
                    f"System networking resources prevent discovery: {exc}"
                ) from exc
            # Refused connections, per-host timeouts, and send/receive errors
            # are local to this address; continue with the next IP.
            continue
        except _SmartLEDProtocolError:
            # Empty, malformed, oversized, or invalidly encoded peer replies
            # do not prevent trying the remaining addresses.
            continue

        if isinstance(response, dict) and response.get("device_name") == _DEVICE_NAME:
            return {"status": "found", "ip": ip, "port": _PORT}

    return {"status": "not_found", "ip": None}
