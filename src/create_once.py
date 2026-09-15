#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic create-once installation that never replaces a first complete writer.

Proposed runner module (grassmann r2, codex 2026-09-14T17:05Z item 4), scratch only until the owner gives separate
write/apply authority.

A check-for-absence followed by `os.replace` lets another writer install a complete record between the check and the
replace, and then overwrites it. This module writes and fsyncs a same-directory temporary file and hard-links it to the
destination: the link fails if the destination already exists, so an existing record is never replaced, and readers
never see partial bytes at the destination. When the destination exists the caller reopens it and compares. There is
no check-then-replace fallback: if the filesystem cannot link, installation refuses.
"""
import os
import tempfile

TEMP_PREFIX = ".create-once-"


class CreateOnceUnavailable(OSError):
    """The filesystem cannot provide an atomic no-replace install here; nothing was installed."""


def install_no_replace(dst, data):
    """Install `data` at `dst` only if nothing is there. Returns True when installed and False when `dst` already existed
    (its bytes are untouched). The temporary file is always removed; a crash before the link leaves only a stale
    `.create-once-*.tmp` that no reader treats as a record."""
    directory = os.path.dirname(os.path.abspath(dst))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=TEMP_PREFIX, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, dst)
        except FileExistsError:
            return False
        except (OSError, NotImplementedError, AttributeError) as exc:
            raise CreateOnceUnavailable(f"no-replace install unavailable for {dst}: {type(exc).__name__}: {exc}") from None
        return True
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
