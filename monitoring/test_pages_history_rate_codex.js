// Offline, synthetic semantic checks plus a consistency check of the retained archive.
'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const R = require('../docs/history-rate.js');
const root = path.resolve(__dirname, '..');
let passed = 0;
function test(name, fn) {fn(); passed++; console.log('PASS ' + name);}
function samples() {
    return [0.6,0.8,0.7,0.2].map((v,i) => ({date:`2026-10-0${i+1}`,
        source:{kind:'REVISION_CURRENT',sha256:String(i+1).repeat(64)},superseded:[],
        methods:{seismic_thd:{raw:{state:'MEASURED',value:v},score:{state:'RECORDED',value:v/2},
            station:{basis:'NOTES_TEXT',value:'IU.TEST'},recorded:{frozen:false},admission:{included:true},
            calibration:{sample_rate_hz:40,baseline:{window:'2026-01-01 to 2026-03-31',mean:0.3,std:0.1}},warnings:[]}}}));
}
const rows = (days=samples(),basis='raw') => R.rows(days,'seismic_thd',basis,'fixture');
function close(a,b) {assert.ok(Math.abs(a-b) < 1e-12, `${a} != ${b}`);}
test('backward differences on later report days',()=>{const r=rows();assert.equal(r[0].reason,'NO_PREVIOUS_DAY');[0.2,-0.1,-0.5].forEach((x,i)=>close(r[i+1].change_per_day,x));assert.equal(r[3].previous_date,'2026-10-03');});
test('not percentage change',()=>close(rows()[1].change_per_day,0.2));
test('genuine zero remains numeric',()=>{const d=samples();d[0].methods.seismic_thd.raw.value=0;close(rows(d)[1].change_per_day,0.8);});
test('unchanged values give known zero',()=>{const d=samples();d[1].methods.seismic_thd.raw.value=0.6;assert.equal(rows(d)[1].change_per_day,0);assert.equal(rows(d)[1].status,'KNOWN');});
test('score basis differs explicitly',()=>close(rows(samples(),'score')[1].change_per_day,0.1));
test('input order does not change results',()=>assert.deepEqual(rows(samples().reverse()),rows()));
test('no input mutation',()=>{const d=samples(),before=JSON.stringify(d);rows(d);assert.equal(JSON.stringify(d),before);});
test('missing day breaks interval',()=>{const d=samples();d.splice(1,1);assert.equal(rows(d)[1].reason,'MISSING_REPORT_DAY');close(rows(d)[2].change_per_day,-0.5);});
test('invalid date is unknown',()=>{const d=samples();d[0].date='2026-02-30';assert.equal(rows(d)[0].reason,'INVALID_REPORT_DAY');});
test('duplicate report days break both sides',()=>{const d=samples();d.push(structuredClone(d[1]));const r=rows(d);assert.equal(r[1].reason,'DUPLICATE_REPORT_DAY');assert.equal(r[2].reason,'DUPLICATE_REPORT_DAY');assert.equal(r[3].reason,'PREVIOUS_DUPLICATE_REPORT_DAY');});
const invalid = [
 ['null',m=>m.raw.value=null,'MEASUREMENT_UNKNOWN'], ['nonfinite',m=>m.raw.value=Infinity,'MEASUREMENT_UNKNOWN'],
 ['numeric string',m=>m.raw.value='0.8','MEASUREMENT_UNKNOWN'], ['placeholder',m=>m.raw.state='UNAVAILABLE_ZERO_PLACEHOLDER','MEASUREMENT_UNKNOWN'],
 ['unconfirmed',m=>m.raw.state='UNCONFIRMED_ZERO','MEASUREMENT_UNKNOWN'], ['frozen flag',m=>m.recorded.frozen=true,'FROZEN'],
 ['frozen warning',m=>m.warnings=['FROZEN'],'FROZEN'], ['unknown station',m=>delete m.station,'STATION_SUPPORT_UNKNOWN'],
 ['unknown processing',m=>delete m.calibration.sample_rate_hz,'PROCESSING_SUPPORT_UNKNOWN'],
 ['unknown baseline',m=>delete m.calibration.baseline,'BASELINE_UNKNOWN']
];
for(const [name,change,reason] of invalid) test(name+' does not become zero',()=>{const d=samples();change(d[1].methods.seismic_thd);const r=rows(d);assert.equal(r[1].reason,reason);assert.equal(r[1].change_per_day,null);assert.equal(r[2].reason,'PREVIOUS_'+reason);});
for(const [name,change] of [
 ['station',m=>m.station.value='IU.OTHER'], ['baseline window',m=>m.calibration.baseline.window='new window'],
 ['baseline mean',m=>m.calibration.baseline.mean=0.5], ['sample rate',m=>m.calibration.sample_rate_hz=20],
 ['unit',m=>m.raw.unit='new unit'], ['method version',m=>m.method_version='v2'], ['regime',m=>m.regime_id='new']
]) test(name+' boundary is unknown',()=>{const d=samples();change(d[1].methods.seismic_thd);assert.equal(rows(d)[1].reason,'SUPPORT_BASELINE_OR_PROCESSING_CHANGE');});
test('daily report hashes are not support changes',()=>assert.equal(rows()[1].status,'KNOWN'));
test('z scores and note text are not support changes',()=>{const d=samples();d[1].methods.seismic_thd.calibration.z_score=8;d[1].methods.seismic_thd.recorded.notes='daily value';assert.equal(rows(d)[1].status,'KNOWN');});
test('object key order is not a boundary',()=>{const d=samples();d[1].methods.seismic_thd.calibration.baseline={std:0.1,mean:0.3,window:'2026-01-01 to 2026-03-31'};assert.equal(rows(d)[1].status,'KNOWN');});
test('revised day stays unknown',()=>{const d=samples();d[1].superseded=[{}];assert.equal(rows(d)[1].reason,'REVISED_DAY_NOT_COMPARABLE');});
test('declared missing revisions stay unknown',()=>{const d=samples();d[1].revision_count=2;assert.equal(rows(d)[1].reason,'REVISED_DAY_NOT_COMPARABLE');});
test('missing report provenance stays unknown',()=>{const d=samples();delete d[1].source.sha256;assert.equal(rows(d)[1].reason,'REPORT_PROVENANCE_UNKNOWN');});
test('excluded score cannot produce a rate',()=>{const d=samples();d[1].methods.seismic_thd.admission.included=false;assert.equal(rows(d,'score')[1].reason,'SCORE_NOT_ADMITTED');assert.equal(rows(d)[1].status,'KNOWN');});
test('overflow stays unknown',()=>{const d=samples();d[0].methods.seismic_thd.raw.value=-Number.MAX_VALUE;d[1].methods.seismic_thd.raw.value=Number.MAX_VALUE;assert.equal(rows(d)[1].reason,'NONFINITE_CHANGE');});
test('unknown units/version not invented',()=>{const r=rows()[1];assert.equal(r.recorded_unit,null);assert.equal(r.recorded_method_version,null);assert.match(R.describe(r),/unit: not retained; method version: not retained/);});
test('CSV contains endpoints, provenance and unknown reasons',()=>{const text=R.csv(rows());assert.match(text,/current_source_sha256/);assert.match(text,/NO_PREVIOUS_DAY/);assert.match(text,/REPORT_DAY_NOT_SENSOR_OR_AVAILABILITY/);assert.equal(text.trim().split('\n').length,5);});
test('invalid basis refuses',()=>assert.throws(()=>rows(samples(),'percent'),/RATE_BASIS_INVALID/));
const archive=[];
test('real archive numerical rates recompute and unknowns remain null',()=>{
 const index=JSON.parse(fs.readFileSync(path.join(root,'docs/raw_history/index.json')));
 for(const region of index.regions) {
  const body=JSON.parse(fs.readFileSync(path.join(root,'docs/raw_history',region+'.json')));
  for(const method of Object.keys(index.methods)) {
   const result=R.rows(body.days,method,'raw',region);
   for(const r of result) {if(r.status==='KNOWN'){assert.equal(r.elapsed_days,1);assert.equal(r.change_per_day,r.current-r.previous);}else assert.equal(r.change_per_day,null);}
   archive.push({region,method,total:result.length,known:result.filter(r=>r.status==='KNOWN').length});
  }
 }
 assert.ok(archive.find(r=>r.region==='hualien'&&r.method==='seismic_thd').known>0);
});
console.log(JSON.stringify({passed,archive},null,2));
