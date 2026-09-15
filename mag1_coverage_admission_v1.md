# MAG-1 coverage-only admission pass v1 (2026-08-21, post R1 PASS)

Probes are COVERAGE-ONLY (existence + data-flow checks); no measurement values analyzed.

| observatory | role | probe result |
|---|---|---|
| IZN (Iznik, KOERI via INTERMAGNET GIN) | istanbul_marmara FAULT-LOCAL | LIVE (1-min, 2026-08-19 series served) |
| FRN (Fresno, USGS) | socal regional reference | LIVE (1-min XYZF adjusted) |
| TUC (Tucson, USGS) | socal SECOND reference -> enables M3 local-innovation with a real reference | LIVE (1-min XYZF adjusted) |
| East Anatolia (kahramanmaras) | fault-local candidate | NO INTERMAGNET coverage found (ELT guess invalid); AFAD/TUBITAK non-INTERMAGNET networks = deeper-search item; DEFAULT = carrier typed MAG-UNTESTABLE at freeze |

Admission consequence for the freeze candidate: admitted carriers = istanbul_marmara (IZN,
M1/M2 endpoints; M3 only if a registered reference is added) + socal_coachella (FRN local-proxy
+ TUC reference; M1/M2/M3); kahramanmaras = typed mag-untestable unless the deeper search lands
before freeze. Per the v0.2 registry rule, any post-freeze availability change re-enters as a
disclosed amendment.
