"""thd-scoped-tls-trust-v1 controls (cayley 2026-10-10; codex b7a29458 correction 2: "retain certificate and hostname
verification, identify the CA source, and test invalid-certificate refusal as well as successful discovery").

Offline: a TLS server on 127.0.0.1 presents a TEST-ONLY certificate for DNS:localhost signed by a TEST-ONLY CA,
both generated for this run only (nothing private is committed). The scoped client trusts exactly that CA bundle:
  * successful service discovery over https://localhost (obspy's own discovery, parsing obspy's sample WADL files);
  * an unrelated CA bundle -> certificate verification refuses, nothing discovered;
  * https://127.0.0.1 with the localhost certificate -> hostname verification refuses;
  * the scoped context is CERT_REQUIRED + check_hostname, built from the named bundle, sha256 recorded;
  * only declared providers are scoped (INGV); undeclared ones keep the default obspy client, untouched;
  * no process-wide trust change (no SSL_CERT_FILE / global opener).
No provider is contacted. Run: <py with obspy> -B -m unittest test_thd_scoped_tls_trust_cayley_20261010 (cwd monitoring/src)
"""
import hashlib
import http.server
import os
import ssl
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thd_provider_routing as TPR  # noqa: E402

FIXTURES = HERE / "fixtures" / "tls_scoped_trust"
WADL = {"/fdsnws/dataselect/1/application.wadl": FIXTURES / "2014-01-07_iris_dataselect.wadl",
        "/fdsnws/station/1/application.wadl": FIXTURES / "2014-01-07_iris_station.wadl"}
# TEST-ONLY TLS material is generated per run into a temporary directory (never committed: no private key in the
# public repository). Needs `cryptography`; without it the TLS tests skip and say so.
FX = None


def setUpModule():
    global FX, _TMP
    try:
        from cryptography import x509  # noqa: F401
    except ImportError:
        return
    import tempfile
    _TMP = tempfile.TemporaryDirectory(prefix="thd-tls-test-")
    FX = Path(_TMP.name)
    _make_tls_material(FX)


def tearDownModule():
    if FX is not None:
        _TMP.cleanup()


def _make_tls_material(out):
    """A test CA, a DNS:localhost server certificate it signs (so 127.0.0.1 is a hostname mismatch), its key, and an
    unrelated CA -- all in `out`, valid only for this run."""
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    now = datetime.datetime.now(datetime.timezone.utc)
    lo, hi = now - datetime.timedelta(days=1), now + datetime.timedelta(days=1)

    def ca(name):
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(lo).not_valid_after(hi)
                .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                             content_commitment=False, key_encipherment=False, data_encipherment=False,
                                             key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
                .sign(key, hashes.SHA256()))
        return key, cert
    ca_key, ca_cert = ca("GeoSpec TEST-ONLY scoped-trust CA")
    _, other = ca("GeoSpec TEST-ONLY unrelated CA")
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
            .issuer_name(ca_cert.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(lo).not_valid_after(hi)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    pem = serialization.Encoding.PEM
    (out / "test_ca.pem").write_bytes(ca_cert.public_bytes(pem))
    (out / "unrelated_ca.pem").write_bytes(other.public_bytes(pem))
    (out / "localhost_cert.pem").write_bytes(cert.public_bytes(pem))
    (out / "localhost_key.pem").write_bytes(key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


class _Handler(http.server.BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):  # noqa: N802
        _Handler.hits.append(self.path)
        path = WADL.get(self.path.split("?")[0])
        if path is None:
            self.send_response(404); self.end_headers(); return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Server:
    def __enter__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(FX / "localhost_cert.pem", FX / "localhost_key.pem")
        self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        _Handler.hits = []
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown(); self.httpd.server_close()


class ScopedTrust(unittest.TestCase):
    def setUp(self):
        if FX is None:
            self.skipTest("cryptography is not installed: the TEST-ONLY TLS material cannot be generated here")

    def context(self, bundle):
        return TPR.trust_context(TPR.CERTIFI_BUNDLE, cafile=str(FX / bundle))

    def test_trusted_ca_discovers_services(self):
        ctx, facts = self.context("test_ca.pem")
        with Server() as srv:
            client = TPR.scoped_trust_client("https://localhost:%d" % srv.port, ssl_context=ctx, timeout=10)
        self.assertIn("dataselect", client.services)
        self.assertIn("station", client.services)
        self.assertTrue(any("dataselect" in hit for hit in _Handler.hits))
        self.assertEqual(facts["bundle_sha256"], hashlib.sha256((FX / "test_ca.pem").read_bytes()).hexdigest())
        self.assertEqual((facts["source"], facts["version"]), (TPR.CERTIFI_BUNDLE, TPR.TLS_TRUST_VERSION))

    def test_untrusted_issuer_refuses_and_discovers_nothing(self):
        ctx, _ = self.context("unrelated_ca.pem")
        with Server() as srv:
            opener = TPR.scoped_trust_client("https://localhost:%d" % srv.port, ssl_context=ctx, timeout=10,
                                             client_class=_NoDiscoveryClient)._url_opener
            with self.assertRaises(urllib.error.URLError) as cm:
                opener.open("https://localhost:%d/fdsnws/station/1/application.wadl" % srv.port, timeout=10)
        self.assertIsInstance(cm.exception.reason, ssl.SSLCertVerificationError)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", str(cm.exception.reason))
        self.assertEqual(_Handler.hits, [], "a refused handshake serves nothing")

    def test_hostname_mismatch_refuses(self):
        ctx, _ = self.context("test_ca.pem")
        with Server() as srv:
            opener = TPR.scoped_trust_client("https://127.0.0.1:%d" % srv.port, ssl_context=ctx, timeout=10,
                                             client_class=_NoDiscoveryClient)._url_opener
            with self.assertRaises(urllib.error.URLError) as cm:
                opener.open("https://127.0.0.1:%d/fdsnws/station/1/application.wadl" % srv.port, timeout=10)
        self.assertIsInstance(cm.exception.reason, ssl.SSLCertVerificationError)
        self.assertIn("match", str(cm.exception.reason).lower())

    def test_untrusted_issuer_through_real_discovery_finds_no_service(self):
        ctx, _ = self.context("unrelated_ca.pem")
        from obspy.clients.fdsn.header import FDSNNoServiceException
        with Server() as srv:
            with self.assertRaises(FDSNNoServiceException):
                TPR.scoped_trust_client("https://localhost:%d" % srv.port, ssl_context=ctx, timeout=10)

    def test_context_keeps_verification_on(self):
        ctx, _ = self.context("test_ca.pem")
        self.assertEqual((ctx.verify_mode, ctx.check_hostname), (ssl.CERT_REQUIRED, True))

    def test_only_the_https_handler_is_replaced(self):
        ctx, _ = self.context("test_ca.pem")
        client = TPR.scoped_trust_client("https://localhost:9443", ssl_context=ctx, timeout=1, client_class=_NoDiscoveryClient)
        https = [h for h in client._url_opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
        self.assertEqual(len(https), 1)
        self.assertIs(https[0]._context, ctx)
        names = {type(h).__name__ for h in client._url_opener.handlers}
        self.assertIn("CustomRedirectHandler", names, "obspy's redirect handling is kept")

    def test_unknown_source_and_missing_bundle_refuse(self):
        with self.assertRaises(ValueError):
            TPR.trust_context("SYSTEM_STORE_GUESS")
        with self.assertRaises((OSError, ssl.SSLError)):
            TPR.trust_context(TPR.CERTIFI_BUNDLE, cafile=str(FX / "absent.pem"))


class Scope(unittest.TestCase):
    def test_only_declared_providers_are_scoped(self):
        # 2026-11-22 (codex 78548eb68): NCEDC joins INGV; the scoped set is exactly these two, every other provider the
        # routes can name keeps the interpreter's default trust
        self.assertEqual(sorted(TPR.TLS_TRUST), ["INGV", "NCEDC"])
        named = sorted({name for entry in TPR.NETWORK_ROUTES.values() for _, name in entry["adapters"]})
        self.assertEqual([p for p in named if TPR.tls_trust_for(p) is not None], ["INGV", "NCEDC"])
        for provider in ("IRIS", "SCEDC", "GEOFON"):
            self.assertIn(provider, named)
            self.assertIsNone(TPR.tls_trust_for(provider))

    def test_the_ingv_entry_is_unchanged(self):
        entry = TPR.TLS_TRUST["INGV"]
        self.assertEqual(sorted(entry), ["basis", "source"])
        self.assertEqual(entry["source"], TPR.CERTIFI_BUNDLE)
        self.assertEqual(hashlib.sha256(entry["basis"].encode("utf-8")).hexdigest(),
                         "5946f672d17f2c0f5e62a8fc25018d897e8cf3c9f960a9645f6cc1363c3d79a5")

    def test_undeclared_provider_gets_the_default_client(self):
        from unittest import mock
        made = []

        class FakeClient:
            def __init__(self, provider, timeout):
                made.append((provider, timeout))
        with mock.patch("obspy.clients.fdsn.Client", FakeClient):
            client, facts = TPR.fdsn_client("IRIS", timeout=7)
        self.assertIsInstance(client, FakeClient)
        self.assertEqual((made, facts), ([("IRIS", 7)], None))

    def test_no_process_wide_trust_change(self):
        if FX is None:
            self.skipTest("cryptography is not installed")
        before = (os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR"), urllib.request._opener)
        TPR.trust_context(TPR.CERTIFI_BUNDLE, cafile=str(FX / "test_ca.pem"))
        TPR.scoped_trust_client("https://localhost:9443", ssl_context=ssl.create_default_context(), timeout=1,
                                client_class=_NoDiscoveryClient)
        self.assertEqual(before, (os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR"), urllib.request._opener))


class NcedcScopedTrust(unittest.TestCase):
    """2026-11-22 (codex 78548eb68): NCEDC is verified against the named bundle, through the same construction site the
    daily run and the recalibration use. No provider is contacted: the client class is a stand-in or the server is local."""

    def test_ncedc_declares_the_named_bundle(self):
        entry = TPR.tls_trust_for("NCEDC")
        self.assertEqual(sorted(entry), ["basis", "source"])
        self.assertEqual(entry["source"], TPR.CERTIFI_BUNDLE)
        for commit in ("99bf4e60d", "fda8ad40f", "45ba008c1"):
            self.assertIn(commit, entry["basis"])
        self.assertTrue(entry["basis"].startswith("RETAINED: "))

    def test_fdsn_client_builds_ncedc_from_the_named_bundle_and_records_its_digest(self):
        try:
            import certifi
        except ImportError:
            self.skipTest("certifi is not installed")
        from unittest import mock
        made = []

        def stand_in(base_url, *, ssl_context, timeout, client_class=None):
            made.append((base_url, ssl_context, timeout))
            return "CLIENT"
        with mock.patch.object(TPR, "scoped_trust_client", stand_in):
            client, facts = TPR.fdsn_client("NCEDC", timeout=120)
        self.assertEqual(client, "CLIENT")
        (base_url, ctx, timeout), = made
        self.assertEqual((base_url, timeout), ("NCEDC", 120))
        self.assertEqual((ctx.verify_mode, ctx.check_hostname), (ssl.CERT_REQUIRED, True))
        bundle = Path(certifi.where()).read_bytes()
        self.assertEqual(facts, dict(version=TPR.TLS_TRUST_VERSION, source=TPR.CERTIFI_BUNDLE,
                                     package_version=certifi.__version__,
                                     bundle_sha256=hashlib.sha256(bundle).hexdigest()))
        # the trust set IS the named bundle: every CA the context holds comes from it, and the system stores were not
        # loaded beside it (the defect is two same-name certificates in the system stores)
        text = bundle.decode("ascii", "ignore")
        in_bundle = {ssl.PEM_cert_to_DER_cert("-----BEGIN CERTIFICATE-----" + block.split("-----END CERTIFICATE-----")[0]
                                              + "-----END CERTIFICATE-----\n")
                     for block in text.split("-----BEGIN CERTIFICATE-----")[1:]}
        held = set(ctx.get_ca_certs(binary_form=True))
        self.assertTrue(held)
        self.assertEqual(held - in_bundle, set(), "a CA outside the named bundle is trusted")
        self.assertEqual(len(held), len(in_bundle))

    def test_fdsn_client_gives_the_real_scoped_client_class_for_ncedc(self):
        try:
            import certifi  # noqa: F401
        except ImportError:
            self.skipTest("certifi is not installed")
        from unittest import mock
        made = []

        class FakeClient:
            def __init__(self, provider, timeout):
                made.append((provider, timeout))
        with mock.patch("obspy.clients.fdsn.Client", FakeClient):
            client, facts = TPR.fdsn_client("NCEDC", timeout=120)
            default, none = TPR.fdsn_client("IRIS", timeout=120)
        self.assertEqual(type(client).__name__, "ScopedTrustClient")
        self.assertIsInstance(client, FakeClient)
        self.assertEqual((client._scoped_ssl_context.verify_mode, client._scoped_ssl_context.check_hostname),
                         (ssl.CERT_REQUIRED, True))
        self.assertEqual(facts["source"], TPR.CERTIFI_BUNDLE)
        self.assertEqual((type(default), none), (FakeClient, None))
        self.assertEqual(made, [("NCEDC", 120), ("IRIS", 120)])

    def test_an_unavailable_bundle_refuses_and_never_falls_back_to_default_trust(self):
        from unittest import mock
        made = []
        with mock.patch.object(TPR, "trust_context", side_effect=OSError("bundle unreadable")), \
                mock.patch("obspy.clients.fdsn.Client", lambda *a, **k: made.append(a)):
            with self.assertRaises(OSError):
                TPR.fdsn_client("NCEDC", timeout=120)
        self.assertEqual(made, [], "no default-trust client is built when the named bundle is unavailable")


class CrossSignedChain(unittest.TestCase):
    """The shape NCEDC serves since 2026-11-17 (grassmann fda8ad40f), on TEST-ONLY certificates: the server presents
    leaf, intermediate and the NEW root cross-signed by an OLD root. A named bundle that holds the new root self-signed
    verifies it whatever else the bundle holds and in whatever order; verification and hostname checking stay on."""

    @classmethod
    def setUpClass(cls):
        cls.dir = None
        try:
            from cryptography import x509  # noqa: F401
        except ImportError:
            return
        import datetime
        import tempfile
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        now = datetime.datetime.now(datetime.timezone.utc)
        lo, hi = now - datetime.timedelta(days=1), now + datetime.timedelta(days=1)

        def name(cn):
            return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

        def cert(subject, key, issuer, issuer_key, ca=True, san=None):
            b = (x509.CertificateBuilder().subject_name(name(subject)).issuer_name(name(issuer))
                 .public_key(key.public_key()).serial_number(x509.random_serial_number())
                 .not_valid_before(lo).not_valid_after(hi)
                 .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
                 .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
                 .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False))
            if san:
                b = b.add_extension(x509.SubjectAlternativeName([x509.DNSName(san)]), critical=False)
            return b.sign(issuer_key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM)
        k = {n: ec.generate_private_key(ec.SECP256R1()) for n in ("absent", "old", "new", "inter", "leaf", "other")}
        old, new, absent = "TEST-ONLY old root", "TEST-ONLY new root", "TEST-ONLY absent root"
        cls._tmp = tempfile.TemporaryDirectory(prefix="thd-tls-chain-test-")
        cls.dir = Path(cls._tmp.name)
        pem = dict(old_self=cert(old, k["old"], old, k["old"]), old_cross=cert(old, k["old"], absent, k["absent"]),
                   new_self=cert(new, k["new"], new, k["new"]), new_cross=cert(new, k["new"], old, k["old"]),
                   inter=cert("TEST-ONLY intermediate", k["inter"], new, k["new"]),
                   leaf=cert("localhost", k["leaf"], "TEST-ONLY intermediate", k["inter"], ca=False, san="localhost"),
                   other=cert("TEST-ONLY unrelated root", k["other"], "TEST-ONLY unrelated root", k["other"]))
        (cls.dir / "chain.pem").write_bytes(pem["leaf"] + pem["inter"] + pem["new_cross"])
        (cls.dir / "key.pem").write_bytes(k["leaf"].private_bytes(serialization.Encoding.PEM,
                                                                  serialization.PrivateFormat.PKCS8,
                                                                  serialization.NoEncryption()))
        for label, parts in dict(new_only=["new_self"], new_then_old=["new_self", "old_self"],
                                 old_then_new=["old_self", "new_self"],
                                 new_then_old_then_cross=["new_self", "old_self", "old_cross"],
                                 cross_then_old_then_new=["old_cross", "old_self", "new_self"],
                                 unrelated=["other"]).items():
            (cls.dir / (label + ".pem")).write_bytes(b"".join(pem[p] for p in parts))

    @classmethod
    def tearDownClass(cls):
        if cls.dir is not None:
            cls._tmp.cleanup()

    def setUp(self):
        if self.dir is None:
            self.skipTest("cryptography is not installed: the TEST-ONLY chain cannot be generated here")

    def serve(self):
        outer = self

        class ChainServer(Server):
            def __enter__(self):
                self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(outer.dir / "chain.pem", outer.dir / "key.pem")
                self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
                self.port = self.httpd.server_address[1]
                self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
                self.thread.start()
                _Handler.hits = []
                return self
        return ChainServer()

    def test_a_bundle_with_the_new_root_discovers_services_in_every_composition_and_order(self):
        for bundle in ("new_only", "new_then_old", "old_then_new", "new_then_old_then_cross", "cross_then_old_then_new"):
            with self.subTest(bundle=bundle):
                ctx, facts = TPR.trust_context(TPR.CERTIFI_BUNDLE, cafile=str(self.dir / (bundle + ".pem")))
                for _ in range(3):
                    with self.serve() as srv:
                        client = TPR.scoped_trust_client("https://localhost:%d" % srv.port, ssl_context=ctx, timeout=10)
                    self.assertIn("dataselect", client.services)
                    self.assertIn("station", client.services)
                self.assertEqual(facts["bundle_sha256"],
                                 hashlib.sha256((self.dir / (bundle + ".pem")).read_bytes()).hexdigest())

    def test_an_unrelated_bundle_fails_closed_on_the_cross_signed_chain(self):
        ctx, _ = TPR.trust_context(TPR.CERTIFI_BUNDLE, cafile=str(self.dir / "unrelated.pem"))
        from obspy.clients.fdsn.header import FDSNNoServiceException
        with self.serve() as srv:
            with self.assertRaises(FDSNNoServiceException):
                TPR.scoped_trust_client("https://localhost:%d" % srv.port, ssl_context=ctx, timeout=10)
            opener = TPR.scoped_trust_client("https://localhost:%d" % srv.port, ssl_context=ctx, timeout=10,
                                             client_class=_NoDiscoveryClient)._url_opener
            with self.assertRaises(urllib.error.URLError) as cm:
                opener.open("https://localhost:%d/fdsnws/station/1/application.wadl" % srv.port, timeout=10)
        self.assertIsInstance(cm.exception.reason, ssl.SSLCertVerificationError)
        self.assertEqual(_Handler.hits, [], "a refused handshake serves nothing")

    def test_a_wrong_hostname_fails_closed_on_the_cross_signed_chain(self):
        ctx, _ = TPR.trust_context(TPR.CERTIFI_BUNDLE, cafile=str(self.dir / "new_only.pem"))
        with self.serve() as srv:
            opener = TPR.scoped_trust_client("https://127.0.0.1:%d" % srv.port, ssl_context=ctx, timeout=10,
                                             client_class=_NoDiscoveryClient)._url_opener
            with self.assertRaises(urllib.error.URLError) as cm:
                opener.open("https://127.0.0.1:%d/fdsnws/station/1/application.wadl" % srv.port, timeout=10)
        self.assertIsInstance(cm.exception.reason, ssl.SSLCertVerificationError)
        self.assertIn("match", str(cm.exception.reason).lower())
        self.assertEqual(_Handler.hits, [])


def _no_discovery():
    from obspy.clients.fdsn import Client

    class NoDiscovery(Client):
        def __init__(self, *args, **kwargs):
            kwargs["_discover_services"] = False
            super().__init__(*args, **kwargs)
    return NoDiscovery


_NoDiscoveryClient = _no_discovery()

if __name__ == "__main__":
    unittest.main()
