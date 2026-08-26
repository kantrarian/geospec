#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A MEASURED no-network sentinel (codex 0509Z item 5).

`http_requests: 0` was a hard-coded ASSERTION in both artifacts: no
code observed whether a request was attempted, so the field proved
nothing. This installs a sentinel that makes an attempt IMPOSSIBLE
and COUNTED -- the reported counter is derived from the measurement,
so a value of 0 now means "nothing tried", not "nobody looked".
"""
import socket

_ATTEMPTS = []


class NetworkAttempted(RuntimeError):
    """Raised the instant an offline operation touches the network."""


class _BlockedSocket(socket.socket):
    def connect(self, *a, **kw):
        _ATTEMPTS.append(a[0] if a else None)
        raise NetworkAttempted(
            "an offline operation attempted a network connection: "
            f"{a[0] if a else '?'}")

    def connect_ex(self, *a, **kw):
        _ATTEMPTS.append(a[0] if a else None)
        raise NetworkAttempted(
            "an offline operation attempted a network connection")


class no_network:
    """Context manager: blocks and COUNTS connection attempts."""

    entered = False

    def __enter__(self):
        self.entered = True
        self._saved = socket.socket
        self._create = socket.create_connection
        self._base = len(_ATTEMPTS)

        def _blocked_create(address, *a, **kw):
            _ATTEMPTS.append(address)
            raise NetworkAttempted(
                f"an offline operation attempted a connection to "
                f"{address}")
        socket.socket = _BlockedSocket
        socket.create_connection = _blocked_create
        return self

    def __exit__(self, *exc):
        socket.socket = self._saved
        socket.create_connection = self._create
        return False

    @property
    def attempts(self):
        return len(_ATTEMPTS) - self._base


def attempts_total():
    return len(_ATTEMPTS)
