Fixtures for test_thd_scoped_tls_trust_cayley_20261010: the two WADL files are obspy's own test data
(obspy/clients/fdsn/tests/data, LGPL-3.0), served by a local TLS test server so the scoped client's real
service discovery has something to parse. The TLS certificates and key are generated per test run into a
temporary directory and are never committed.
