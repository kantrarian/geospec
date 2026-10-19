"""Bounded verifier regressions; synthetic content, no publication or network."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from monitoring.src import raw_history_cayley as raw


class CheckoutByteIdentity(unittest.TestCase):
    def test_autocrlf_checkout_preserves_manifest_bound_bytes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="raw-checkout-codex-") as d:
            repo = Path(d, "repo")
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repo), "-c", "core.autocrlf=true", *args],
                                               stderr=subprocess.STDOUT)
            git("init", "-q")
            (repo / ".gitattributes").write_bytes((root / ".gitattributes").read_bytes())
            source = repo / "docs/raw_history"
            source.mkdir(parents=True)
            payload = b'{"value":0.2}\n'
            for name in ("index.json", "hualien.json"):
                (source / name).write_bytes(payload)
            git("add", ".gitattributes", "docs/raw_history")
            dest = Path(d, "checkout")
            dest.mkdir()
            git("checkout-index", "--prefix=" + dest.as_posix() + "/", "--all")
            for name in ("index.json", "hualien.json"):
                self.assertEqual((dest / "docs/raw_history" / name).read_bytes(), payload)


class VerifyReadFailures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="raw-verify-codex-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = {
            "index.json": json.dumps({"build": {"mode": "COMMIT", "commit": "test"},
                                      "preservation": {"equal": 1}}).encode(),
            "hualien.json": b'{"raw_value":0.2}\n',
        }
        for name, data in self.files.items():
            (self.root / name).write_bytes(data)
        self.build = patch.object(raw, "build", return_value=self.files)
        self.build.start()
        self.addCleanup(self.build.stop)

    def verify(self):
        return raw.verify("unused", "test", str(self.root))

    def test_nominal_content_accepts(self):
        self.assertEqual(self.verify(), [])

    def test_region_directory_in_place_of_file_refuses(self):
        region = self.root / "hualien.json"
        region.unlink()
        region.mkdir()
        self.assertTrue(self.verify(), "an unreadable required region is not verified")

    def test_unreadable_index_refuses(self):
        index = self.root / "index.json"
        index.unlink()
        index.mkdir()
        self.assertTrue(self.verify(), "an unreadable index is not verified")

    def test_invalid_json_index_refuses_without_traceback(self):
        (self.root / "index.json").write_bytes(b"broken")
        self.assertTrue(self.verify())

    def test_nonobject_index_refuses_without_traceback(self):
        (self.root / "index.json").write_bytes(b"[]")
        self.assertTrue(self.verify())

    def test_boolean_alias_does_not_verify_as_integer(self):
        changed = json.loads(self.files["index.json"])
        changed["preservation"]["equal"] = True
        (self.root / "index.json").write_text(json.dumps(changed), encoding="utf-8")
        self.assertTrue(self.verify(), "True must not bind as the count 1")

    def test_missing_output_directory_refuses_without_traceback(self):
        self.assertTrue(raw.verify("unused", "test", str(self.root / "absent")))


if __name__ == "__main__":
    unittest.main()
