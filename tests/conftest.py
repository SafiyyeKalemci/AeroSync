from __future__ import annotations

import os
import socket


def pytest_configure(config) -> None:
    """Hard-block real sockets only for tests launched by preflight_check."""
    if os.getenv("AEROSYNC_PREFLIGHT_TEST_MODE") != "1":
        return

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def is_loopback(address) -> bool:
        if isinstance(address, tuple) and address:
            return str(address[0]).lower() in {"127.0.0.1", "::1", "localhost"}
        return False

    def guarded_connect(sock, address):
        if is_loopback(address):
            return original_connect(sock, address)
        raise RuntimeError("Real network access is disabled during preflight tests.")

    def guarded_create_connection(address, *args, **kwargs):
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise RuntimeError("Real network access is disabled during preflight tests.")

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create_connection
