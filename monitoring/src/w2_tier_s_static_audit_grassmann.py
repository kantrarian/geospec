#!/usr/bin/env python3
"""Tier-S execution-path STATIC AUDIT (grassmann).

Two complementary static sweeps over the Tier-S execution-path modules,
plus a selftest that proves each sweep detects the class it claims.

  KEY sweep    every literal key read off a PRE carrier, checked
               against the v2 CLOSED field sets. Carriers are found by
               SCOPE-QUALIFIED provenance -- per-function symbols, with
               argument-to-parameter propagation through direct local
               calls -- never a hand-written name list.

  ARITY sweep  intra-module direct calls by bare Name, reported as
               DIRECT_LOCAL_POSITIONAL_ARITY. It does NOT cover
               module-qualified or aliased calls, scope shadowing,
               callbacks, or composition.

Neither sweep substitutes for the real end-to-end execution gate; they
supplement it.

This module NEVER imports the code it audits. The closed field sets are
read out of the target runner's AST, and with --commit every audited
file is read from that commit's blob and the worktree copy must match.

Usage:
  python w2_tier_s_static_audit_grassmann.py --root R [--commit REV]
  python w2_tier_s_static_audit_grassmann.py --root R --selftest
"""
import argparse
import ast
import hashlib
import os
import subprocess
import sys

FILES = [
    "monitoring/src/w2_tier_s_driver_cayley.py",
    "monitoring/src/w2_tier_s_runner_cayley.py",
    "monitoring/src/w2_tier_selector_cayley.py",
    "monitoring/src/w2_cert_runner_cayley.py",
]
RUNNER_REL = "monitoring/src/w2_tier_s_runner_cayley.py"
# element 0 of the returned tuple is the capsule. codex 1914Z: the
# landed driver defines _load_pre_checked(...) -> (pre, sha), and every
# real caller unpacks it, so classifying it as a SCALAR producer left a
# latent false positive -- `both = _load_pre_checked(...)` would have
# been tagged a whole capsule, the exact shape the fire_pre control
# already excluded.
TUPLE_PRODUCERS = {"fire_pre", "_load_pre_checked"}
# the return value IS the capsule
SCALAR_PRODUCERS = {"_load_pre"}
COPY_FUNCS = {"dict", "deepcopy", "copy"}
MODULE_SCOPE = "<module>"


class AuditRefusal(RuntimeError):
    pass


def _const_str(n):
    return n.value if isinstance(n, ast.Constant) and \
        isinstance(n.value, str) else None


# ------------------------------------------------- committed byte source
def _git(root, *args):
    p = subprocess.run(("git", "-C", root) + args, capture_output=True)
    return p.returncode, p.stdout


def resolve_commit(root, rev):
    rc, out = _git(root, "rev-parse", f"{rev}^{{commit}}")
    full = out.decode().strip()
    if rc != 0 or len(full) != 40:
        raise AuditRefusal(
            f"AUDIT_REVISION_UNRESOLVABLE: {rev!r} does not resolve to "
            "a commit in this repository")
    return full


def source_bytes(root, rel, commit):
    """With a commit, the AUDITED BYTES ARE THE COMMITTED BYTES, and a
    divergent worktree copy refuses -- codex 1849Z finding 1: a
    --commit that only prints is a label, not a binding."""
    live = os.path.join(root, *rel.split("/"))
    if commit is None:
        if not os.path.isfile(live):
            return None
        with open(live, "rb") as f:
            return f.read().replace(b"\r\n", b"\n")
    rc, blob = _git(root, "cat-file", "blob", f"{commit}:{rel}")
    if rc != 0:
        raise AuditRefusal(
            f"AUDIT_PATH_ABSENT_AT_COMMIT: {rel} is not present at "
            f"{commit[:12]}")
    if os.path.isfile(live):
        with open(live, "rb") as f:
            got = f.read().replace(b"\r\n", b"\n")
        if got != blob:
            raise AuditRefusal(
                f"AUDIT_WORKTREE_DIVERGENT: {rel} on disk "
                f"({hashlib.sha256(got).hexdigest()[:12]}) is not the "
                f"bytes at {commit[:12]} "
                f"({hashlib.sha256(blob).hexdigest()[:12]})")
    return blob


def load_fields(root, commit=None):
    """codex 1849Z finding 2: the previous version prepended the target
    tree to sys.path and IMPORTED the runner, executing its top level,
    while the docstring claimed it never executes audited code. The
    closed sets now come out of the runner's AST."""
    src = source_bytes(root, RUNNER_REL, commit)
    if src is None:
        raise AuditRefusal(f"AUDIT_RUNNER_ABSENT: {RUNNER_REL}")
    tree = ast.parse(src.decode("utf-8"))
    want = {"PRE_FIELDS": None, "EXECUTION_FIELDS": None,
            "PRE_SCHEMA": None}
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if isinstance(t, ast.Name) and t.id in want:
                try:
                    want[t.id] = ast.literal_eval(n.value)
                except Exception:
                    pass
    if not want["PRE_FIELDS"] or not want["PRE_SCHEMA"]:
        raise AuditRefusal(
            "AUDIT_CLOSED_SETS_UNREADABLE: the runner's PRE_FIELDS / "
            "PRE_SCHEMA are not module-level literals")
    return (set(want["PRE_FIELDS"]),
            set(want["EXECUTION_FIELDS"] or ()), want["PRE_SCHEMA"])


# ------------------------------------------------------- scope machinery
def scope_map(tree):
    """node -> its LEXICAL SCOPE CHAIN, innermost first.

    A chain, not a single name: a capsule bound in an outer function
    and read inside a nested helper is the same object, and my first
    scope-aware version stored only the innermost name, which silently
    dropped `bad_pre = _copy.deepcopy(pre)` in a nested selftest helper
    -- narrowing coverage while the change was sold as precision. A
    sibling scope still cannot see it, so shadowing stays clean."""
    owner, stack = {}, []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            stack.append(n.name)
            for c in ast.iter_child_nodes(n):
                self.visit(c)
            stack.pop()
        visit_AsyncFunctionDef = visit_FunctionDef

        def generic_visit(self, n):
            owner[n] = tuple(reversed(stack)) + (MODULE_SCOPE,)
            super().generic_visit(n)
    V().visit(tree)
    for n in ast.walk(tree):
        owner.setdefault(n, (MODULE_SCOPE,))
    return owner


def _has(aliases, chain, name):
    """Resolve a name against the lexical chain (innermost outward)."""
    return any((s, name) in aliases for s in chain)


def _pre_literal(n, schema):
    if not isinstance(n, ast.Dict):
        return False
    for k, v in zip(n.keys, n.values):
        if _const_str(k) == "schema":
            s = _const_str(v)
            if s and s.startswith(schema.rsplit("-", 1)[0]):
                return True
    return False


def _whole(n, scope, aliases, schema, tuple_elem0=False):
    """Is `n` a WHOLE pre capsule in this scope?  A sub-object
    (`pre["geometry"]`) is not; a copy (`dict(pre)`) is."""
    if isinstance(n, ast.Call):
        f = n.func
        name = f.id if isinstance(f, ast.Name) else \
            (f.attr if isinstance(f, ast.Attribute) else None)
        if name in SCALAR_PRODUCERS:
            return True
        if name in TUPLE_PRODUCERS:
            # only element 0 of the tuple is the capsule
            return bool(tuple_elem0)
        if name in COPY_FUNCS and n.args:
            return _whole(n.args[0], scope, aliases, schema)
        if name in ("loads", "dumps") and n.args:
            return _whole(n.args[0], scope, aliases, schema)
        return False
    if isinstance(n, ast.Name):
        return _has(aliases, scope, n.id)
    if isinstance(n, ast.Dict):
        if _pre_literal(n, schema):
            return True
        for k, v in zip(n.keys, n.values):
            if k is None and _whole(v, scope, aliases, schema):
                return True
    return False


def discover_aliases(tree, schema):
    """Scope-qualified (function, name) pairs, plus argument ->
    parameter propagation through direct local calls, so a capsule
    passed into a renamed parameter is tracked and an unrelated
    shadowed `pre` in another scope is not tainted."""
    owner = scope_map(tree)
    defs = {n.name: n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)}
    aliases = set()
    for _ in range(12):
        grew = False
        for n in ast.walk(tree):
            sc = owner.get(n, (MODULE_SCOPE,))
            inner = sc[0]
            if isinstance(n, ast.Assign):
                tgts, val = n.targets, n.value
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and \
                    n.value:
                tgts, val = [n.target], n.value
            else:
                tgts = None
            if tgts:
                for t in tgts:
                    if isinstance(t, ast.Tuple):
                        for i, el in enumerate(t.elts):
                            if i == 0 and isinstance(el, ast.Name) and \
                                    _whole(val, sc, aliases, schema,
                                           tuple_elem0=True) and \
                                    (inner, el.id) not in aliases:
                                aliases.add((inner, el.id))
                                grew = True
                    elif isinstance(t, ast.Name) and \
                            _whole(val, sc, aliases, schema) and \
                            (inner, t.id) not in aliases:
                        aliases.add((inner, t.id))
                        grew = True
            # argument -> parameter propagation, direct local calls only
            if isinstance(n, ast.Call) and \
                    isinstance(n.func, ast.Name) and \
                    n.func.id in defs:
                callee = defs[n.func.id]
                params = [a.arg for a in callee.args.args]
                for i, a in enumerate(n.args):
                    if i >= len(params):
                        break
                    if _whole(a, sc, aliases, schema) and \
                            (callee.name, params[i]) not in aliases:
                        aliases.add((callee.name, params[i]))
                        grew = True
                for kwn in n.keywords:
                    if kwn.arg in params and \
                            _whole(kwn.value, sc, aliases, schema) and \
                            (callee.name, kwn.arg) not in aliases:
                        aliases.add((callee.name, kwn.arg))
                        grew = True
        if not grew:
            break
    return aliases, owner


def key_sweep(root, pre_fields, exec_fields, files, schema,
              commit=None):
    findings, examined, per_file = [], 0, {}
    for rel in files:
        src = source_bytes(root, rel, commit)
        if src is None:
            continue
        text = src.decode("utf-8")
        tree = ast.parse(text)
        lines = text.split("\n")
        aliases, owner = discover_aliases(tree, schema)
        per_file[rel] = sorted({f"{s}.{v}" for s, v in aliases})
        for n in ast.walk(tree):
            sc = owner.get(n, (MODULE_SCOPE,))
            base = key = shown = None
            if isinstance(n, ast.Subscript) and \
                    isinstance(n.value, ast.Name) and \
                    _has(aliases, sc, n.value.id):
                base, key = pre_fields, _const_str(n.slice)
                shown = f'{n.value.id}["{key}"]'
            elif isinstance(n, ast.Subscript) and \
                    isinstance(n.value, ast.Subscript) and \
                    isinstance(n.value.value, ast.Name) and \
                    _has(aliases, sc, n.value.value.id) and \
                    _const_str(n.value.slice) == "execution":
                base, key = exec_fields, _const_str(n.slice)
                shown = (f'{n.value.value.id}["execution"]'
                         f'["{key}"]')
            elif isinstance(n, ast.Call) and \
                    isinstance(n.func, ast.Attribute) and \
                    n.func.attr == "get" and \
                    isinstance(n.func.value, ast.Name) and \
                    _has(aliases, sc, n.func.value.id) and n.args:
                base, key = pre_fields, _const_str(n.args[0])
                shown = f'{n.func.value.id}.get("{key}")'
            if key is None or not base:
                continue
            examined += 1
            if key not in base:
                findings.append((rel, n.lineno, shown,
                                 lines[n.lineno - 1].strip()))
    return findings, examined, per_file


def arity_sweep(root, files, commit=None):
    findings, calls, ndefs = [], 0, 0
    for rel in files:
        src = source_bytes(root, rel, commit)
        if src is None:
            continue
        text = src.decode("utf-8")
        tree = ast.parse(text)
        lines = text.split("\n")
        defs = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                a = n.args
                pos = len(a.posonlyargs) + len(a.args)
                defs[n.name] = {
                    "min": pos - len(a.defaults),
                    "max": None if a.vararg else pos,
                    "kwreq": {k.arg for k, d in
                              zip(a.kwonlyargs, a.kw_defaults)
                              if d is None},
                    "kwall": {k.arg for k in a.kwonlyargs}}
        ndefs += len(defs)
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call) or \
                    not isinstance(n.func, ast.Name) or \
                    n.func.id not in defs:
                continue
            d = defs[n.func.id]
            if any(isinstance(a, ast.Starred) for a in n.args) or \
                    any(k.arg is None for k in n.keywords):
                continue
            calls += 1
            kw = {k.arg for k in n.keywords}
            supplied = len(n.args) + len(kw - d["kwall"])
            why = None
            if supplied < d["min"]:
                why = f"supplied {supplied}, needs >= {d['min']}"
            elif d["max"] is not None and supplied > d["max"]:
                why = f"supplied {supplied}, takes <= {d['max']}"
            elif d["kwreq"] - kw:
                why = ("missing required keyword-only "
                       f"{sorted(d['kwreq'] - kw)}")
            if why:
                findings.append((rel, n.lineno, n.func.id, why,
                                 lines[n.lineno - 1].strip()))
    return findings, calls, ndefs


def run(root, commit_rev):
    commit = resolve_commit(root, commit_rev) if commit_rev else None
    pre_f, exec_f, schema = load_fields(root, commit)
    present = []
    for rel in FILES:
        try:
            if source_bytes(root, rel, commit) is not None:
                present.append(rel)
        except AuditRefusal:
            raise
    print("W2 TIER-S STATIC AUDIT (grassmann)")
    print(f"  root          {root}")
    shown_commit = commit or "(none -- worktree bytes, NOT commit-bound)"
    print(f"  commit        {shown_commit}")
    print(f"  pre schema    {schema}")
    print(f"  closed fields pre={len(pre_f)} execution={len(exec_f)}")
    for rel in present:
        b = source_bytes(root, rel, commit)
        print(f"    {hashlib.sha256(b).hexdigest()[:16]}  {rel}")
    print()
    kf, kn, al = key_sweep(root, pre_f, exec_f, present, schema,
                           commit)
    print("KEY SWEEP -- scope-qualified provenance-discovered PRE "
          "carriers")
    for rel in present:
        print(f"    {rel.split('/')[-1]:<34} "
              f"{', '.join(al.get(rel, [])) or '(none)'}")
    print(f"  reads examined {kn}")
    print(f"  RESULT         {len(kf)} unknown-key read(s)")
    for rel, ln, shown, text in kf:
        print(f"    {rel}:{ln}  {shown}\n        {text}")
    print()
    af, ac, ad = arity_sweep(root, present, commit)
    print("ARITY SWEEP -- DIRECT_LOCAL_POSITIONAL_ARITY only")
    print("  (not module-qualified/aliased calls, shadowing, "
          "callbacks, or composition)")
    print(f"  local defs {ad}, direct call sites {ac}")
    print(f"  RESULT         {len(af)} mismatch(es)")
    for rel, ln, fn, why, text in af:
        print(f"    {rel}:{ln}  {fn}() {why}\n        {text}")
    print()
    print("SCOPE: supplements the real end-to-end execution gate; "
          "never substitutes for it.")
    return kf, af


# ------------------------------------------------------------- selftest
# fixtures are LINE LISTS joined at run time: embedding escape
# sequences here kept getting mangled in transit, and a fixture
# that does not say what it means is exactly what this audit
# exists to catch.
KEY_MUTS = [
    ("renamed alias, unknown key",
     ['def f(repo):',
      '    pre9 = _load_pre(repo, s)',
      "    return pre9['host']"], True),
    ("alias derived from another alias",
     ['def f(repo):',
      '    pre9 = _load_pre(repo, s)',
      '    later = dict(pre9)',
      "    return later['interpreter']"], True),
    ("pre-shaped literal, unknown execution field",
     ['def f():',
      "    p = {'schema': 'f2g-w2-tier-s-pre-invocation-v2'}",
      "    return p['execution']['hostname']"], True),
    ("RENAMED CALLEE PARAMETER carrying a stale key",
     ['def check(carrier):',
      "    return carrier['host']",
      'def f(repo):',
      '    pre = _load_pre(repo, s)',
      '    return check(pre)'], True),
    ("TUPLE UNPACK element 0 IS the capsule (_load_pre_checked)",
     ['def f(outdir):',
      '    pre, sha = _load_pre_checked(outdir)',
      "    return pre['host']"], True),
    ("TUPLE UNPACK element 0 IS the capsule (fire_pre)",
     ['def f(repo):',
      '    pre, pts = fire_pre(repo)',
      "    return pre['interpreter']"], True),
    ("closure carrier read in a NESTED helper",
     ['def outer(repo):',
      '    pre = _load_pre(repo, s)',
      '    def inner():',
      "        return pre['host']",
      '    return inner'], True),
    ("shadowed sub-object must NOT be a capsule (no false positive)",
     ['def f(repo):',
      '    pre = _load_pre(repo, s)',
      "    return pre['geometry']",
      'def g(other):',
      "    pre = other['geometry']",
      "    return pre['capsule_digest']"], False),
    ("whole tuple from fire_pre is NOT a capsule",
     ['def f(repo):',
      '    both = fire_pre(repo)',
      "    return both['host']"], False),
    ("whole tuple from _load_pre_checked is NOT a capsule",
     ['def f(outdir):',
      '    both = _load_pre_checked(outdir)',
      "    return both['host']"], False),
]
ARITY_MUTS = [
    ("positional omission",
     ['def g(a, b, c):', '    return a',
      'def h():', '    return g(1, 2)'], True),
    ("required keyword-only omission",
     ['def g(a, *, must):', '    return a',
      'def h():', '    return g(1)'], True),
]


def selftest(root):
    import tempfile
    pre_f, exec_f, schema = load_fields(root, None)
    ok = True
    print("SELFTEST -- planted mutations must be detected, and clean "
          "shapes must NOT be")
    for label, lines, expect in KEY_MUTS + ARITY_MUTS:
        code = chr(10).join(lines) + chr(10)
        kind = "key" if any(l is lines for _a, l, _c in
                            KEY_MUTS) else "arity"
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "monitoring", "src"))
            rel = FILES[0]
            with open(os.path.join(td, *rel.split("/")), "w",
                      encoding="utf-8") as f:
                f.write(code)
            if kind == "key":
                found, _n, _a = key_sweep(td, pre_f, exec_f, [rel],
                                          schema)
            else:
                found, _c, _d = arity_sweep(td, [rel])
        hit = bool(found)
        good = (hit == expect)
        ok = ok and good
        want = "detect" if expect else "stay clean"
        print(f"  {'OK  ' if good else 'FAIL'}  ({want}) {label}")
    # commit-binding controls
    try:
        resolve_commit(root, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        print("  FAIL  (refuse) nonexistent revision")
        ok = False
    except AuditRefusal:
        print("  OK    (refuse) nonexistent revision")
    print("SELFTEST:", "ALL PASS" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    try:
        if a.selftest:
            sys.exit(0 if selftest(root) else 1)
        kf, af = run(root, a.commit)
        sys.exit(1 if (kf or af) else 0)
    except AuditRefusal as exc:
        print(f"AUDIT REFUSED: {exc}")
        sys.exit(2)
