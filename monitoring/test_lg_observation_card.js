'use strict';
// LG observation age on the public region cards and in the raw-history download (cayley 2026-11-11; codex 87d56f8aa,
// 7a98cf3a4). The page's own inline script runs in a VM up to its startup calls; no network, no browser. One
// hand-dated fixture (byte-identical to the hosted check's) drives the card line and the LG audit columns, which
// must agree; a retrieval clock never changes the age.
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
vm.runInContext(inline[0].slice(0, cut) + '\nglobalThis.T = {renderCards, lgCardObservation, rawLgAudit};', context, {filename: 'index.html:inline'});
const T = context.T;
const FIX = JSON.parse(fs.readFileSync(path.join(__dirname, 'test_fixtures', 'lg_observation_age_fixture_cayley_20261111.json'), 'utf8'));
const E = FIX.expected, clone = v => JSON.parse(JSON.stringify(v));
const card = (lg, day = FIX.report_day, extra = {}) => {
    T.renderCards(Object.assign({date: day, earthquake_events: {}, regions: {tokyo_kanto: {date: day + 'T00:00:00', tier: 0, combined_risk: 0,
        confidence: 0.8, methods_available: 1, agreement: 'single', components: lg ? {lambda_geo: lg} : {}}}}, extra));
    return nodes['status-grid'].innerHTML;
};
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('PASS ' + name); };
const LINE = 'LG: GNSS observation ' + E.observation_day + ' · ' + E.observation_age_days + ' d before report day ' + E.report_day +
    ' (support through ' + E.support_through_day + '); the report day is not the observation date';

test('the card states the retained observation day, support-through day and age at the report day', () => {
    const h = card(FIX.component);
    assert.ok(h.includes('data-status="LG_OBSERVATION_AGE"'), h);
    assert.ok(h.includes(LINE), h);
});
test('a retrieval clock never changes the age', () => {
    const k = clone(FIX.component);
    k.baseline_provenance.measurement.retrieval = {files_mtime_utc: {G100: '2026-11-20T00:00:00Z'}, oldest_mtime_utc: '2026-11-20T00:00:00Z'};
    assert.ok(card(k).includes(LINE));
});
test('a later report re-carrying the same observation shows it OLDER, never younger', () => {
    assert.ok(card(FIX.component, '2026-11-20').includes('32 d before report day 2026-11-20'));
});
test('a legacy LG row without a retained measurement says unknown, never blank', () => {
    const k = clone(FIX.component); delete k.baseline_provenance;
    const h = card(k);
    assert.ok(h.includes('data-status="LG_OBSERVATION_TIME_UNKNOWN"') && h.includes('LG: GNSS observation time unknown: not retained by this report'), h);
});
test('inconsistent retained dates are named, not shown as an age', () => {
    const k = clone(FIX.component); k.baseline_provenance.measurement.support_through_day = '2026-11-13';
    assert.ok(card(k).includes('LG: GNSS observation time unknown: the retained dates are inconsistent'));
});
test('unavailable LG, no LG and the CSV fallback carry no observation line', () => {
    const k = clone(FIX.component); k.available = false;
    for (const h of [card(k), card(null), card(FIX.component, FIX.report_day, {fallback: true})]) assert.ok(!h.includes('lg-observation'), h);
});
test('the raw-history download audit gives the same observation day and age as the card', () => {
    const a = T.rawLgAudit({date: FIX.report_day, methods: {lambda_geo: {recorded: FIX.component,
        calibration: {baseline_provenance: FIX.component.baseline_provenance}, admission: {basis: 'METHOD_SET', included: true, weight: 0.4}}}});
    assert.equal(a.observation_day, E.observation_day);
    assert.equal(a.support_through_day, E.support_through_day);
    assert.equal(a.observation_age_days, E.observation_age_days);
});
test('the data-integrity note names the product actually read', () => {
    assert.ok(html.includes('IGS20 <code>.tenv</code>'));
    assert.ok(!html.includes('computed from real NGL GPS .tenv3 files'), 'the old .tenv3-only claim is gone');
});
console.log(JSON.stringify({lg_observation_card: 'PASS', passed, browser_rendering: 'NOT_TESTED'}));
