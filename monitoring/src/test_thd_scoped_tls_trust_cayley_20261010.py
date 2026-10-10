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
        self.assertEqual(sorted(TPR.TLS_TRUST), ["INGV"])
        self.assertIsNone(TPR.tls_trust_for("IRIS"))
        self.assertIsNone(TPR.tls_trust_for("NCEDC"))

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
