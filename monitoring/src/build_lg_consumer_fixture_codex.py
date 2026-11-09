"""Create consumer fixtures using real combiner and archive exporter, synthetic provenance, no I/O providers."""
import json
from pathlib import Path
import sys

from test_lg_integration_candidate import EnsembleIntegrationTests, provenance, DAY
import raw_history_cayley as RH


def build():
    harness = EnsembleIntegrationTests()
    bad = provenance(4)
    bad['measurement']['operator_identity'] = 'd' * 64
    missing = provenance(4)
    del missing['measurement']['observation_day']
    cases = []
    for name, ratio, prov in [('zero-score', .9321286354018504, None), ('nonzero-score', 4, None),
                              ('excluded-identity', 4, bad), ('excluded-clock', 4, missing)]:
        result = harness.compute(ratio, prov)
        report = result.to_dict()
        contributions = {method: report['components'][method]['risk_score'] * weight
                         for method, weight in result.effective_weights.items()}
        assert abs(sum(contributions.values()) - result.combined_risk) < 1e-12
        day = dict(date=DAY.date().isoformat(), source={'report_timestamp': '2026-11-12T12:00:00Z'},
                   methods={m: RH.method_entry(m, c, report) for m, c in report['components'].items()},
                   combined=RH.combined_entry(report))
        cases.append(dict(name=name, report=report, day=day, expected=dict(
            combined=result.combined_risk, weight=result.effective_weights.get('lambda_geo', 0),
            contribution=contributions.get('lambda_geo', 0), included='lambda_geo' in result.method_set['included'])))
    assert cases[1]['expected']['combined'] > cases[0]['expected']['combined']
    return dict(scope='SYNTHETIC_PROVENANCE_REAL_COMBINER_AND_RAW_HISTORY_EXPORTER', cases=cases)


if __name__ == '__main__':
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(build(), handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(output)
