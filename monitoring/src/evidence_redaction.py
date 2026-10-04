"""evidence_redaction.py -- remove credential-shaped material from free text before it becomes retained evidence.

METHOD_QUALIFICATION_DELIVERY_PLAN M4 / codex db9a28ff finding 4: provider exception messages are written into the
structured station-attempt records and travel to the export and the hosted bundle, so arbitrary exception text must not
be copied unfiltered. This is a shape filter, not a secret detector: it removes URL user-info, Authorization / Bearer
values, the value of any credential-named key (password, token, api key, secret, session, cookie, HINET_*...) and long
mixed-case alphanumeric runs that look like tokens. Hex digests (lowercase hex) and ordinary messages pass unchanged.
The text is bounded AFTER redaction, so a cut can never expose the tail of a secret the pattern would have matched.
"""
import re

REDACTED = "[REDACTED]"
REDACTED_TOKEN = "[REDACTED_TOKEN]"
DEFAULT_LIMIT = 240

_CREDENTIAL_KEY = (r"(?:pass(?:word|wd)?|pwd|secret|client[_-]?secret|token|access[_-]?token|refresh[_-]?token|"
                   r"api[_-]?key|access[_-]?key|session(?:id)?|cookie|credentials?|auth|"
                   r"[a-z0-9_]*_(?:user|password|token|secret|key))")
_PATTERNS = (
    # scheme://user:password@host -> scheme://[REDACTED]@host
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s@:]+:[^/\s@]*@"), r"\1" + REDACTED + "@"),
    # Authorization: Basic xxx / Proxy-Authorization: Bearer xxx
    (re.compile(r"(?i)\b((?:proxy-)?authorization)\s*[:=]\s*(?:[A-Za-z]+\s+)?[^\s,;]+"), r"\1: " + REDACTED),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer " + REDACTED),
    # key=value / key: value / "key": "value" for credential-named keys
    (re.compile(r"(?i)(\b" + _CREDENTIAL_KEY + r"\b[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&)]+)"), r"\1" + REDACTED),
    # long mixed-case alphanumeric runs (base64 / opaque tokens); lowercase hex digests are not matched
    (re.compile(r"(?<![A-Za-z0-9+/_-])(?=[A-Za-z0-9+/_-]{32,})(?=[^\s]*[A-Z])(?=[^\s]*[a-z])(?=[^\s]*[0-9])"
                r"[A-Za-z0-9+/_-]{32,}={0,2}"), REDACTED_TOKEN),
)


def redact(text, limit=DEFAULT_LIMIT):
    """`text` with credential-shaped material replaced, then bounded to `limit` characters (None = unbounded)."""
    if text is None:
        return None
    out = str(text)
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    if limit is not None and len(out) > limit:
        out = out[:limit] + "..."
    return out
