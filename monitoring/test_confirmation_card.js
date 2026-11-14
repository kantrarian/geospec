'use strict';
// Confirmation on the public region cards and in the raw-history download (cayley 2026-11-14; codex 93df4f774 item 3).
// The page's own inline script runs in a VM up to its startup calls; no network, no browser. The saved cross-view
// fixture (written by monitoring/src/test_confirmation_cross_view_cayley_20261114.py from the issued 2026-11-12 report
// through the real runner functions) drives the card line and the download columns, which must agree with each other
// and with the report's recorded flag, report days, observation counts and reasons.
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'docs', 'index.html'), 'utf8');
const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
assert.equal(inline.length, 1, 'exactly one inline application script');
const START = "console.log('=== GeoSpec Dashboard Initializing ===');";
const cut = inline[0].indexOf(START);
assert.ok(cut > 0, 'startup marker present');
const nodes = {};
const node = () => ({innerHTML: '', textContent: '', style: {}, value: '', hidden: false, classList: {add() {}, remove() {}, toggle() {}},
    addEventListener() {}, setAttribute() {}, appendChild() {}, replaceChildren() {}});
const document = {getElementById: id => nodes[id] ??= node(), createElement: () => node(), querySelectorAll: () => [], querySelector: () => null,
    addEventListener() {}};
const context = vm.createContext({document, window: {}, console: {log() {}, warn() {}, error() {}}, setInterval: () => 0, setTimeout: () => 0,
    fetch: async () => { throw new Error('Unexpected network access'); }, Plotly: {}, URL, Blob: function () {},
    localStorage: {getItem: () => null, setItem() {}}});
vm.runInContext(inline[0].slice(0, cut) + '\nglobalThis.T = {renderCards, confirmationAudit, confirmationCardLine, rawConfirmation, rawTableRows, RAW_COLUMNS};',
    context, {filename: 'index.html:inline'});
const T = context.T;
const FIX = JSON.parse(fs.readFileSync(path.join(__dirname, 'test_fixtures', 'confirmation_cross_view_fixture_cayley_20261114.json'), 'utf8'));
const clone = v => JSON.parse(JSON.stringify(v));
const decode = s => s.replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('PASS ' + name); };

for (const [region, e] of Object.entries(FIX.expected)) {
    const row = FIX.report.regions[region];
    test(region + ': the card line states the recorded flag, report days, observation counts and reason', () => {
        T.renderCards({date: FIX.report.date, earthquake_events: {}, regions: {[region]: row}});
        const h = nodes['status-grid'].innerHTML;
        assert.ok(h.includes('data-status="' + e.card_status + '"'), h);
        const m = /<div class="confirmation-line"[^>]*>([^<]*)<\/div>/.exec(h);
        assert.ok(m, 'confirmation line present');
        assert.equal(decode(m[1]), e.card_text);
    });
    test(region + ': the download columns equal the card audit and the report', () => {
        const d = Object.assign({date: FIX.report.date}, FIX.raw_history_days[region]);
        const c = T.rawConfirmation(d);
        const p = row.persistence, u = p.regime.observation_updates;
        assert.equal(c.confirmation_status, e.card_status);
        assert.equal(c.report_days, String(p.regime.report_days));
        assert.equal(c.thd_observation_updates, u.seismic_thd.observation_updates === null ? '' : String(u.seismic_thd.observation_updates));
        assert.equal(c.lg_observation_updates, u.lambda_geo ? String(u.lambda_geo.observation_updates) : '');
        assert.equal(c.not_confirmed_reasons, p.regime.not_confirmed_reasons.join(' '));
        for (const k of ['confirmation_status', 'report_days', 'lg_observation_updates', 'thd_observation_updates', 'not_confirmed_reasons'])
            assert.ok(T.RAW_COLUMNS.includes(k), k);
        const rows = T.rawTableRows([d], 'seismic_thd');
        assert.equal(rows[0].confirmation_status, e.card_status);
    });
}
test('a report issued before the rule keeps its flag and gets no invented reason', () => {
    const legacy = clone(FIX.report.regions.kaikoura);
    legacy.persistence = {current_tier: 1, consecutive_days: 2, is_confirmed: true, regime: {regime: 'calibration-eligibility-v3|method-comparability-v1'}};
    assert.equal(T.confirmationCardLine(legacy), '');
    const a = T.confirmationAudit({is_confirmed: true, consecutive_days: 2, confirmation_rule: null}, 1);
    assert.equal(a.status, 'CONFIRMED_BEFORE_RULE');
    assert.equal(a.text, null);
});
test('below WATCH no confirmation line is drawn', () => {
    const r = clone(FIX.report.regions.tokyo_kanto); r.tier = 0;
    assert.equal(T.confirmationCardLine(r), '');
});
test('a regime transition names its cause, not an outage', () => {
    const a = T.confirmationAudit({is_confirmed: false, report_days: 1, confirmation_rule: 'observation-update-confirmation-v1',
        not_confirmed_reasons: ['REPORT_DAYS_BELOW_REQUIRED:REGIME_TRANSITION'], observation_updates: {}}, 2);
    assert.equal(decode(a.text), 'Not confirmed: 1 report day — too few report days (rule change: earlier report days are not counted)');
});
test('recorded text is escaped on the card', () => {
    const a = T.confirmationAudit({is_confirmed: false, report_days: 2, confirmation_rule: 'observation-update-confirmation-v1',
        not_confirmed_reasons: ['<img src=x>'], observation_updates: {}}, 1);
    assert.ok(!a.text.includes('<img'), a.text);
});
console.log(JSON.stringify({confirmation_card: 'PASS', passed, browser_rendering: 'NOT_TESTED'}));
