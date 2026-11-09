'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const R = require('../docs/history-rate.js');
let passed = 0;
const test = (name, fn) => {fn(); passed++; console.log('PASS ' + name);};
const samples = () => [0.1, 0.9, 0.9].map((value, i) => ({date:'2026-11-0'+(i+1),
    source:{kind:'REVISION_CURRENT',sha256:String(i+1).repeat(64)},
    methods:{lambda_geo:{raw:{state:'MEASURED',value},score:{state:'RECORDED',value:0},
        recorded:{frozen:false},calibration:{},admission:{included:false}}}}));
const rows = d => R.rows(d || samples(), 'lambda_geo', 'recorded_raw', 'fixture');
test('numeric LG delta does not require a seismic station', () => {
    const r = rows()[1]; assert.equal(r.change_per_day,0.8); assert.equal(r.status,'KNOWN');
    assert.equal(r.comparability_status,'UNKNOWN'); assert.equal(r.comparability_reason,'GNSS_SUPPORT_UNKNOWN');
    assert.match(r.clamp_note,/CLAMP_BOUND/); assert.equal(r.recorded_unit,null);
});
test('support-qualified legacy view remains separate', () => {
    const r = R.rows(samples(),'lambda_geo','raw');
    assert.ok(r.every(x=>x.change_per_day===null));
});
test('score qualification is not bypassed', () => {
    assert.ok(R.rows(samples(),'lambda_geo','score').every(x=>x.reason==='SCORE_NOT_ADMITTED'));
});
test('held reported value has a real zero difference, not evidence of a new observation', () => {
    assert.equal(rows()[2].change_per_day,0); assert.equal(rows()[2].comparability_status,'UNKNOWN');
});
for (const [name, mutate, reason] of [
    ['placeholder', d=>d[1].methods.lambda_geo.raw.state='UNAVAILABLE_ZERO_PLACEHOLDER','MEASUREMENT_UNKNOWN'],
    ['missing number', d=>d[1].methods.lambda_geo.raw.value=null,'MEASUREMENT_UNKNOWN'],
    ['infinite number', d=>d[1].methods.lambda_geo.raw.value=Infinity,'MEASUREMENT_UNKNOWN'],
    ['frozen', d=>d[1].methods.lambda_geo.recorded.frozen=true,'FROZEN'],
    ['revision', d=>d[1].revision_count=2,'REVISED_DAY_NOT_COMPARABLE'],
    ['unbound report', d=>delete d[1].source.sha256,'REPORT_PROVENANCE_UNKNOWN'],
]) test(name+' cannot become a recorded difference',()=>{
    const d=samples();mutate(d);const r=rows(d);assert.equal(r[1].reason,reason);
    assert.equal(r[1].change_per_day,null);assert.equal(r[2].change_per_day,null);
});
test('gap is not bridged',()=>{const d=samples();d.splice(1,1);assert.equal(rows(d)[1].reason,'MISSING_REPORT_DAY');});
test('CSV and hover disclose unknown support and clamp',()=>{
    assert.match(R.csv(rows()),/comparability_status,comparability_reason,clamp_note/);
    assert.match(R.describe(rows()[1]),/Retained support comparison: UNKNOWN/);
    assert.match(R.describe(rows()[1]),/unclipped change cannot be recovered/);
});
test('reported baseline changes do not become physical continuity',()=>{
    const d=samples();d.forEach(x=>{x.methods.lambda_geo.calibration={baseline_provenance:{
        operator_identity:'a'.repeat(64),calibration_sha256:'b'.repeat(64),baseline_value:2,
        measurement:{operator_identity:'a'.repeat(64),selection_status:'OK'}}};});
    d[1].methods.lambda_geo.calibration.baseline_provenance.calibration_sha256='d'.repeat(64);
    const r=rows(d)[1];assert.equal(r.change_per_day,0.8);
    assert.equal(r.comparability_reason,'SUPPORT_BASELINE_OR_PROCESSING_CHANGE');
});
test('new GNSS support compares its own carrier, not a seismic station',()=>{
    const d=samples();d.forEach(x=>{x.methods.lambda_geo.calibration={baseline_provenance:{
        operator_identity:'a'.repeat(64),calibration_sha256:'b'.repeat(64),baseline_value:2,
        measurement:{operator_identity:'a'.repeat(64),selection_status:'OK'}}};});
    let r=R.rows(d,'lambda_geo','raw')[1];assert.equal(r.status,'KNOWN');assert.equal(r.change_per_day,0.8);
    d[1].methods.lambda_geo.calibration.baseline_provenance.measurement.operator_identity='e'.repeat(64);
    r=R.rows(d,'lambda_geo','raw')[1];assert.equal(r.reason,'GNSS_SUPPORT_UNKNOWN');
});
const receipt=[];
test('retained Tokyo and Kumamoto November 4 steps are displayed exactly',()=>{
    for(const [region, expected] of [['tokyo_kanto',0.8321286354018504],['kumamoto',0.08076795224370515]]) {
        const body=JSON.parse(fs.readFileSync(path.join(__dirname,'../docs/raw_history',region+'.json')));
        const before=JSON.stringify(body), result=R.rows(body.days,'lambda_geo','recorded_raw',region);
        const r=result.find(x=>x.date==='2026-11-04');
        assert.equal(r.status,'KNOWN');assert.ok(Math.abs(r.change_per_day-expected)<1e-12);
        assert.equal(r.comparability_status,'UNKNOWN');assert.equal(JSON.stringify(body),before);
        receipt.push({region,known:result.filter(x=>x.status==='KNOWN').length,total:result.length,step:r});
    }
});
console.log(JSON.stringify({passed,receipt},null,2));
