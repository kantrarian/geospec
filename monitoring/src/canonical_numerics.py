#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r5-canonical-numerics-v1 (codex contract 185751Z; cayley bar a6ce01e; owner-gated).

An explicit information-losing quotient of raw BLAS float outputs so that R5
model/record bytes are identical on every host: `qsig(x, 11)` projects a binary64
value onto 11 significant decimal digits (ROUND_HALF_EVEN, Decimal.from_float —
never Decimal(str(x))). The digest domain serializes every float as a tagged
canonical token so a verifier can recompute `model_sha256` from persisted content
plus this declared policy alone, with no producer runtime. Gates compare canonical
Decimals against exact Decimal thresholds with a ±1 ulp11 ambiguity band that
fails closed to R3 (`R5_NUMERIC_BOUNDARY_AMBIGUITY`) — host noise may never
select a decision branch. This is NOT tolerant equality: verification is exact
equality on the declared canonical domain.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN

POLICY = {"id": "r5-canonical-numerics-v1", "rounding": "ROUND_HALF_EVEN",
          "significant_decimal_digits": 11}

_MODEL_SCHEMA = "r5-model-canonical-v1"
_RECORD_SCHEMA = "r5-record-canonical-v1"


def qsig(x, n: int = 11) -> Decimal:
    """§2: one finite binary64 -> Decimal with n significant decimal digits.
    NaN/inf refuse; either signed zero collapses to Decimal(0)."""
    d = Decimal.from_float(float(x))
    if not d.is_finite():
        raise ValueError(f"non-finite value refused: {x!r}")
    if d == 0:
        return Decimal(0)
    return d.quantize(Decimal(1).scaleb(d.adjusted() - (n - 1)),
                      rounding=ROUND_HALF_EVEN)


def _text_of_decimal(q: Decimal) -> str:
    """Canonical text for an (already-quantized) Decimal: fixed-point, never
    exponent notation; trailing fractional zeroes and trailing point removed;
    zero is exactly '0'."""
    if q == 0:
        return "0"
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def canonical_text(x, n: int = 11) -> str:
    """§2 canonical text of qsig(x, n)."""
    return _text_of_decimal(qsig(x, n))


def parse_canonical_token(s: str, n: int = 11) -> Decimal:
    """Decimal-preserving parse that REFUSES any token not already in canonical
    q-n form (exponent notation, trailing zeroes, more than n significant digits,
    non-finite, or any round-trip mismatch)."""
    if not isinstance(s, str):
        raise ValueError("token must be a string")
    d = Decimal(s)
    if not d.is_finite():
        raise ValueError(f"non-finite token refused: {s!r}")
    if _text_of_decimal(d) != s:
        raise ValueError(f"token not canonical text: {s!r}")
    if d != 0 and len(d.normalize().as_tuple().digits) > n:
        raise ValueError(f"token exceeds {n} significant digits: {s!r}")
    return d


def _canon_view(obj, n: int):
    """§3 recursive digest view: strings/booleans/null/ints preserved (bool checked
    before int); every finite float/Decimal becomes {"$f11": "<canonical text>"};
    arrays keep order; objects require string keys (sorted at serialization);
    unknown types and non-finite numbers refuse."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return {"$f11": canonical_text(obj, n)}
    if isinstance(obj, Decimal):
        return {"$f11": _text_of_decimal(qsig(float(obj), n))}
    if isinstance(obj, (list, tuple)):
        return [_canon_view(v, n) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"non-string key refused: {k!r}")
            out[k] = _canon_view(v, n)
        return out
    raise ValueError(f"unknown type refused: {type(obj).__name__}")


def _policy(n: int) -> dict:
    return {"id": POLICY["id"], "rounding": POLICY["rounding"],
            "significant_decimal_digits": n}


def _envelope_bytes(schema: str, key: str, body, n: int) -> bytes:
    env = {key: _canon_view(body, n), "numeric_policy": _policy(n), "schema": schema}
    return json.dumps(env, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def model_digest(model: dict, n: int = 11) -> str:
    """§3: SHA-256 over the canonical envelope of the COMPLETE model object,
    excluding any digest field itself."""
    body = {k: v for k, v in model.items() if not k.endswith("_sha256")}
    return hashlib.sha256(_envelope_bytes(_MODEL_SCHEMA, "model", body, n)).hexdigest()


def canonical_record_bytes(record: dict, n: int = 11) -> bytes:
    """The record serialized on the same canonical numeric surface."""
    return _envelope_bytes(_RECORD_SCHEMA, "record", record, n)


def gate_compare(x, threshold) -> str:
    """§1: compare qsig(x, 11) with the EXACT Decimal threshold. Within
    ±ulp11(threshold) -> 'ambiguous' (caller fails closed to R3 and emits
    R5_NUMERIC_BOUNDARY_AMBIGUITY); otherwise 'lt' or 'ge'."""
    q = qsig(x, 11)
    t = Decimal.from_float(float(threshold))
    if not t.is_finite():
        raise ValueError("non-finite threshold refused")
    ulp = Decimal(1).scaleb(t.adjusted() - 10)
    if abs(q - t) <= ulp:
        return "ambiguous"
    return "lt" if q < t else "ge"


def canonical_residual(log_ratio, beta, api7, r30, n: int = 11) -> Decimal:
    """C-prime (codex dfad58d ruling; bar REV 2/3): the decision-bearing residual
    as an EXACT function of canonical operands. Every operand (log_ratio, b0, b1,
    b2, api7, r30) is projected to q(n) FIRST; the prediction is evaluated exactly
    in Decimal in the DECLARED operation order `b1*api7 + b2*r30 + b0`, subtracted
    from the projected log_ratio, and the result projected ONCE to q(n). BLAS
    supplies the fitted beta; all downstream rank-grid arithmetic runs on this
    canonical carrier, so grid cells cannot be selected by host-dependent float
    noise."""
    import decimal
    ql = qsig(log_ratio, n)
    b0, b1, b2 = (qsig(b, n) for b in beta)
    qa, qr = qsig(api7, n), qsig(r30, n)
    with decimal.localcontext() as ctx:
        ctx.prec = 60          # exact for products/sums of q11 operands
        resid = ql - (b1 * qa + b2 * qr + b0)
    if resid == 0:
        return Decimal(0)
    return resid.quantize(Decimal(1).scaleb(resid.adjusted() - (n - 1)),
                          rounding=ROUND_HALF_EVEN)


def qfloat(x, n: int = 11) -> float:
    """float carrier of qsig(x, n) — binary64 has ample precision for n <= 15, so
    q(float(q(x))) == q(x): the projection is idempotent through this carrier."""
    return float(qsig(x, n))


def canonical_json_bytes(obj, n: int = 11) -> bytes:
    """Plain-JSON persistence whose float tokens are canonical fixed-point text.
    NOTE the deliberate asymmetry with the tagged digest view: a plain file cannot
    preserve the int/float type class for integral floats (0.0 serializes as `0`),
    so a verifier must NEVER parse-and-recanonicalize these files — it recomputes
    the object and compares THIS writer's bytes against the stored bytes directly.
    Byte equality against the canonical writer is the strictest §3 check: any
    non-canonical token in the file breaks it."""
    def enc(o):
        if isinstance(o, bool):
            return "true" if o else "false"
        if o is None:
            return "null"
        if isinstance(o, str):
            return json.dumps(o, ensure_ascii=False)
        if isinstance(o, int):
            return str(o)
        if isinstance(o, float):
            return canonical_text(o, n)
        if isinstance(o, Decimal):
            return _text_of_decimal(qsig(float(o), n))
        if isinstance(o, (list, tuple)):
            return "[" + ",".join(enc(v) for v in o) + "]"
        if isinstance(o, dict):
            items = []
            for k in sorted(o.keys()):
                if not isinstance(k, str):
                    raise ValueError(f"non-string key refused: {k!r}")
                items.append(json.dumps(k, ensure_ascii=False) + ":" + enc(o[k]))
            return "{" + ",".join(items) + "}"
        raise ValueError(f"unknown type refused: {type(o).__name__}")
    return enc(obj).encode("utf-8")
