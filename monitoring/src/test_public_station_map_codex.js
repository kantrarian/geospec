'use strict';
// Offline display controls: no browser, network or producer mutation.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'docs/index.html'), 'utf8');
const map = JSON.parse(fs.readFileSync(path.join(root, 'docs/thd_station_map.json'), 'utf8'));
const report = JSON.parse(fs.readFileSync(path.join(root, 'docs/ensemble_latest.json'), 'utf8'));
const blocks = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]).filter(s => s.trim());
assert(blocks.length > 0);
for (const block of blocks) new vm.Script(block);
const start = html.indexOf('async function renderStationMap(');
const end = html.indexOf('async function loadData()', start);
assert(start >= 0 && end > start);
const nodes = {'station-map-lead': {}, 'station-map-table': {}};
const context = {document: {getElementById: id => nodes[id]}, fetch: async () => ({ok: true, json: async () => map})};
vm.createContext(context);
vm.runInContext(html.slice(start, end), context);
let passed = 0;
async function check(name, body, expected) {
    await context.renderStationMap(body);
    assert(nodes['station-map-lead'].innerHTML.includes(`${expected} of 14 regions have an available numeric THD value`), name);
    passed++;
    console.log('PASS', name);
}
function mutated(available, raw) {
    const body = structuredClone(report);
    for (const row of Object.values(body.regions)) Object.assign(row.components.seismic_thd, {available, raw_value: raw});
    return body;
}
(async () => {
    await check('nominal real report', report, 13);
    await check('unavailable with retained station notes', mutated(false, 0.4), 0);
    await check('null raw value', mutated(true, null), 0);
    await check('string raw value', mutated(true, '0.4'), 0);
    await check('boolean raw value', mutated(true, false), 0);
    await check('nonfinite raw value', mutated(true, NaN), 0);
    await check('genuine numeric zero', mutated(true, 0), 13);
    assert(html.includes('uncalibrated defaults or absolute-threshold fallbacks'));
    assert(html.includes('not a geodesic minimum'));
    assert(map.distance_basis.box_km.includes('not a geodesic minimum'));
    passed++;
    console.log('PASS copy matches baseline and distance scope');
    console.log(`${passed} controls PASS; ${blocks.length} inline script parsed; actual browser rendering NOT_TESTED`);
})().catch(error => { console.error(error); process.exitCode = 1; });
