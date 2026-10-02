// Offline check for the report-times label (codex a2a5cfe0): run `node check_report_times_label_cayley_20261001.js`
// from this directory. Loads formatReportTimes from BOTH pages (between the REPORT-TIMES-LABEL markers) and checks
// the cases codex named. Exits non-zero on any failure.
'use strict';
const fs = require('fs');
const path = require('path');

const PAGES = [path.join(__dirname, '..', '..', 'docs', 'index.html'), path.join(__dirname, 'index.html')];
const CASES = [
    // the live case: scored day 2026-09-29, report generated 2026-10-01 (two-day scored-date lag)
    [{date: '2026-09-29', timestamp: '2026-10-01T11:52:42.623236+00:00'},
        'Latest scored day: 2026-09-29 · Report generated: 2026-10-01 11:52:42 +00:00'],
    [{date: '2026-09-29'}, 'Latest scored day: 2026-09-29 · Report generated: unknown (no timestamp in the report)'],
    [{date: '2026-09-29', timestamp: ''}, 'Latest scored day: 2026-09-29 · Report generated: unknown (no timestamp in the report)'],
    [{date: '2026-09-29', timestamp: '2026-10-01T11:52:42'},
        'Latest scored day: 2026-09-29 · Report generated: unknown (timestamp unreadable or without a UTC offset)'],
    [{date: '2026-09-29', timestamp: '2026-10-01T25:52:42+00:00'},
        'Latest scored day: 2026-09-29 · Report generated: unknown (timestamp unreadable or without a UTC offset)'],
    [{date: '2026-09-29', timestamp: 1759319562},
        'Latest scored day: 2026-09-29 · Report generated: unknown (timestamp unreadable or without a UTC offset)'],
    // the stated offset is kept, not converted
    [{date: '2026-09-29', timestamp: '2026-10-01T07:52:42-04:00'},
        'Latest scored day: 2026-09-29 · Report generated: 2026-10-01 07:52:42 -04:00'],
    [{date: '2026-09-29', timestamp: '2026-10-01T11:52:42Z'},
        'Latest scored day: 2026-09-29 · Report generated: 2026-10-01 11:52:42 +00:00'],
    [{date: '2026-02-30', timestamp: '2026-10-01T11:52:42+00:00'},
        'Latest scored day: unknown · Report generated: 2026-10-01 11:52:42 +00:00'],
    [{date: 'Unknown'}, 'Latest scored day: unknown · Report generated: unknown (no timestamp in the report)'],
    [null, 'Latest scored day: unknown · Report generated: unknown (no timestamp in the report)'],
];
const FORBIDDEN = /update|measur|acquir|sensor|observ/i;

let failures = 0;
const bodies = [];
for (const page of PAGES) {
    const html = fs.readFileSync(page, 'utf8').replace(/\r\n/g, '\n');
    const m = /\/\/ REPORT-TIMES-LABEL-BEGIN[\s\S]*?\n([\s\S]*?)\/\/ REPORT-TIMES-LABEL-END/.exec(html);
    if (!m) { console.error('FAIL marker block missing in', page); failures++; continue; }
    bodies.push(m[1]);
    const fn = new Function(m[1] + '\nreturn formatReportTimes;')();
    for (const [input, want] of CASES) {
        const got = fn(input);
        if (got !== want) { console.error('FAIL', path.basename(path.dirname(page)), JSON.stringify(input), '\n got ', got, '\n want', want); failures++; }
        if (FORBIDDEN.test(got)) { console.error('FAIL label implies measurement/update time:', got); failures++; }
    }
    if (html.includes('Last update:')) { console.error('FAIL old "Last update:" label still present in', page); failures++; }
    const calls = html.split('formatReportTimes(ensembleData)').length - 1;
    if (calls !== 1) { console.error('FAIL expected exactly one call site in', page, 'found', calls); failures++; }
}
if (bodies.length === 2 && bodies[0] !== bodies[1]) { console.error('FAIL the two pages carry different functions'); failures++; }
console.log(failures ? `FAILED ${failures}` : `PASS ${CASES.length} cases x ${PAGES.length} pages`);
process.exit(failures ? 1 : 0);
