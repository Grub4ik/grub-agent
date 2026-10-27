from __future__ import annotations

import json
import socket
import unittest
from unittest.mock import patch

from grub.agent import ToolCall, ToolExecutor, ToolRegistry
from grub.tools.discover_smartled import (
    SMARTLED_DISCOVERY_SCHEMA,
    SmartLEDDiscoveryError,
    discover_smartled,
)


class FakeSocket:
    def __init__(self, chunks=(), recv_errors=()):
        self.chunks = list(chunks)
        self.recv_errors = list(recv_errors)
        self.timeout_values = []
        self.sent = []
        self.closed = False
        self.recv_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()

    def close(self):
        self.closed = True

    def settimeout(self, value):
        self.timeout_values.append(value)

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, _size):
        self.recv_count += 1
        if self.recv_errors:
            raise self.recv_errors.pop(0)
        if self.chunks:
            return self.chunks.pop(0)
        return b""


def json_reply(device_name: str) -> bytes:
    return json.dumps({"device_name": device_name}).encode("utf-8")


class SmartLEDDiscoveryTests(unittest.TestCase):
    def test_exact_device_is_found_and_only_help_is_sent(self):
        peer = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer) as connect:
            result = discover_smartled()

        self.assertEqual(result, {"status": "found", "ip": "192.168.0.1", "port": 4210})
        connect.assert_called_once_with(("192.168.0.1", 4210), timeout=0.1)
        self.assertEqual(peer.sent, [b"HELP\n"])
        self.assertTrue(peer.closed)

    def test_connection_failure_continues_to_the_next_address(self):
        peer = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch(
            "grub.tools.discover_smartled.socket.create_connection",
            side_effect=[ConnectionRefusedError("closed"), peer],
        ) as connect:
            result = discover_smartled()

        self.assertEqual(result["ip"], "192.168.0.2")
        self.assertEqual([call.args[0][0] for call in connect.call_args_list], [
            "192.168.0.1", "192.168.0.2"
        ])

    def test_response_timeout_continues_to_the_next_address(self):
        timed_out_peer = FakeSocket(recv_errors=[socket.timeout("late")])
        found_peer = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch(
            "grub.tools.discover_smartled.socket.create_connection",
            side_effect=[timed_out_peer, found_peer],
        ):
            result = discover_smartled(response_timeout=0.5)

        self.assertEqual(result["ip"], "192.168.0.2")
        self.assertTrue(timed_out_peer.closed)
        self.assertTrue(found_peer.closed)
        self.assertTrue(all(timeout > 0 for timeout in timed_out_peer.timeout_values))

    def test_json_may_arrive_in_multiple_tcp_fragments(self):
        peer = FakeSocket([b'{"device_', b'name":"SmartLED-ESP32"}'])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()

        self.assertEqual(result["status"], "found")
        self.assertEqual(peer.recv_count, 2)
        self.assertTrue(peer.closed)

    def test_utf8_character_may_be_split_between_fragments(self):
        response = '{"label":"лампа","device_name":"SmartLED-ESP32"}'.encode("utf-8")
        split = response.index("м".encode("utf-8")) + 1
        peer = FakeSocket([response[:split], response[split:]])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()
        self.assertEqual(result["status"], "found")
        self.assertEqual(peer.recv_count, 2)

    def test_incomplete_utf8_after_json_is_rejected(self):
        peer = FakeSocket([b'{"device_name":"SmartLED-ESP32"}\xc3', b""])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()
        self.assertEqual(result, {"status": "not_found", "ip": None})

    def test_malformed_json_is_ignored_for_that_address(self):
        malformed = FakeSocket([b'{"device_name": nope}', b""])
        found = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch(
            "grub.tools.discover_smartled.socket.create_connection",
            side_effect=[malformed, found],
        ):
            result = discover_smartled()
        self.assertEqual(result["ip"], "192.168.0.2")

    def test_empty_response_is_not_found_and_connection_is_closed(self):
        peer = FakeSocket([b""])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()
        self.assertEqual(result, {"status": "not_found", "ip": None})
        self.assertTrue(peer.closed)

    def test_invalid_utf8_is_ignored(self):
        invalid = FakeSocket([b'{"device_name":"\xff"}', b""])
        found = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch(
            "grub.tools.discover_smartled.socket.create_connection",
            side_effect=[invalid, found],
        ):
            self.assertEqual(discover_smartled()["ip"], "192.168.0.2")

    def test_different_device_name_does_not_match(self):
        peer = FakeSocket([json_reply("OtherDevice")])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()
        self.assertEqual(result, {"status": "not_found", "ip": None})

    def test_missing_device_name_does_not_match(self):
        peer = FakeSocket([b'{"version":1}'])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()
        self.assertEqual(result, {"status": "not_found", "ip": None})

    def test_oversized_response_is_ignored_and_later_host_is_checked(self):
        oversized = FakeSocket([b'{"device_name":"SmartLED-ESP32","extra":"too long"}'])
        found = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch(
            "grub.tools.discover_smartled.socket.create_connection",
            side_effect=[oversized, found],
        ):
            result = discover_smartled(max_response_bytes=40)
        self.assertEqual(result["ip"], "192.168.0.2")
        self.assertTrue(oversized.closed)

    def test_search_stops_at_first_matching_device(self):
        first = FakeSocket([json_reply("SmartLED-ESP32")])
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=first) as connect:
            result = discover_smartled()
        self.assertEqual(result["ip"], "192.168.0.1")
        connect.assert_called_once()

    def test_search_checks_exactly_hosts_one_through_254(self):
        addresses = []

        def refuse(address, timeout):
            addresses.append(address)
            raise ConnectionRefusedError("no listener")

        with patch("grub.tools.discover_smartled.socket.create_connection", side_effect=refuse):
            result = discover_smartled()

        self.assertEqual(result, {"status": "not_found", "ip": None})
        self.assertEqual(addresses, [(f"192.168.0.{octet}", 4210) for octet in range(1, 255)])

    def test_socket_closes_when_send_or_receive_fails(self):
        peer = FakeSocket()
        peer.sendall = unittest.mock.Mock(side_effect=OSError("send failed"))
        with patch("grub.tools.discover_smartled.socket.create_connection", return_value=peer):
            result = discover_smartled()
        self.assertEqual(result, {"status": "not_found", "ip": None})
        self.assertTrue(peer.closed)

    def test_fatal_network_error_is_distinguished_from_not_found(self):
        with patch(
            "grub.tools.discover_smartled.socket.create_connection",
            side_effect=OSError(100, "network down"),
        ) as connect:
            with self.assertRaises(SmartLEDDiscoveryError):
                discover_smartled()
        connect.assert_called_once()

    def test_model_schema_rejects_ip_and_other_arguments(self):
        registry = ToolRegistry()
        registry.register(
            "discover_smartled",
            "Discover a SmartLED device on the local network and return its IP address.",
            SMARTLED_DISCOVERY_SCHEMA,
            lambda: {"status": "not_found", "ip": None},
            permission="read",
        )
        self.assertEqual(registry.schemas()[0]["function"]["parameters"], {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        })
        result = ToolExecutor(registry).execute(
            ToolCall("discover_smartled", {"ip": "192.168.0.42", "port": 4210})
        )
        self.assertFalse(result.success)
        self.assertIn("Invalid arguments", result.error)

    def test_invalid_configuration_is_rejected_before_scanning(self):
        with patch("grub.tools.discover_smartled.socket.create_connection") as connect:
            with self.assertRaises(ValueError):
                discover_smartled(connect_timeout=0)
        connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
