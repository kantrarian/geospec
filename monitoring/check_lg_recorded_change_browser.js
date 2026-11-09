// Local candidate + retained archive. No hosted job or production write.
'use strict';
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {chromium}=require('C:/codex/.tools/environment-browser/node_modules/playwright');
const root=path.resolve(__dirname,'../docs'),out=process.argv[2]&&path.resolve(process.argv[2]);
const integrated=process.argv[3]?JSON.parse(fs.readFileSync(process.argv[3],'utf8')):null;
const server=http.createServer((req,res)=>{
 const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/\/$/,'/index.html'));
 if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 fs.readFile(file,(err,bytes)=>{if(err){res.writeHead(404).end();return;}
  res.setHeader('Content-Type',({'.html':'text/html','.js':'application/javascript','.json':'application/json','.csv':'text/csv'})[path.extname(file)]||'text/plain');res.end(bytes);});
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const checks=[],pass=name=>{checks.push(name);console.log('PASS '+name);};
 try{
  if(out)fs.mkdirSync(out,{recursive:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:'+server.address().port+'/',{waitUntil:'networkidle'});
  await page.waitForFunction(()=>!document.getElementById('raw-export').disabled).catch(async error=>{
    console.log(JSON.stringify({errors,status:await page.locator('#raw-status').innerText(),url:page.url()}));throw error;
  });
  await page.locator('#raw-region').selectOption('tokyo_kanto');
  await page.waitForFunction(()=>rawDisplayedSelection?.region==='tokyo_kanto'&&!document.getElementById('raw-export').disabled);
  await page.locator('#raw-method').selectOption('all');
  await page.locator('#raw-view').selectOption('rate');
  await page.waitForFunction(()=>!document.getElementById('rate-export').disabled&&rateRows.some(r=>r.method==='lambda_geo'));
  assert.equal(await page.locator('#rate-basis').inputValue(),'recorded_raw');
  const step=await page.evaluate(()=>rateRows.find(r=>r.method==='lambda_geo'&&r.date==='2026-11-04'));
  assert.ok(Math.abs(step.change_per_day-0.8321286354018504)<1e-12);
  const lg=await page.evaluate(()=>document.getElementById('raw-rate-chart').data.find(t=>t.name==='LG recorded-number change/day'));
  assert.ok(Math.abs(lg.y[lg.x.indexOf('2026-11-04')]-step.change_per_day)<1e-12);
  assert.match(await page.locator('#rate-summary').innerText(),/UNKNOWN support comparability/);
  pass('Tokyo LG step appears in all-input chart with descriptive-mode label');
  const index=await page.evaluate(()=>rateRows.findIndex(r=>r.method==='lambda_geo'&&r.date==='2026-11-04'));
  await page.locator('#rate-day').selectOption(String(index));
  assert.match(await page.locator('#rate-detail').innerText(),/GNSS_SUPPORT_UNKNOWN/);
  assert.match(await page.locator('#rate-detail').innerText(),/unclipped change cannot be recovered/);
  pass('endpoint inspection discloses GNSS support and clamp');
  assert.equal(await page.locator('#raw-lg-day').inputValue(),'2026-11-04');
  assert.match(await page.locator('#raw-lg-summary').innerText(),/LG EXCLUDED/);
  assert.match(await page.locator('#raw-lg-summary').innerText(),/OBSERVATION_TIME_UNKNOWN/);
  assert.match(await page.locator('#raw-lg-summary').innerText(),/NO_REGISTERED_FRESHNESS_POLICY/);
  pass('retained Tokyo discrepancy explains actual exclusion and unknown clock; rate selection follows report day');
  const pending=page.waitForEvent('download');await page.locator('#rate-export').click();const download=await pending;
  const csv=fs.readFileSync(await download.path(),'utf8');
  assert.match(csv,/comparability_status,comparability_reason,clamp_note/);assert.match(csv,/0.8321286354018504/);
  pass('CSV exports exact delta with provenance and comparability');
  await page.waitForFunction(()=>document.getElementById('raw-rate-chart').layout.shapes.some(s=>s.name?.startsWith('retained-event:')));
  const before=await page.evaluate(()=>JSON.stringify(document.getElementById('raw-rate-chart').data));
  await page.locator('#raw-events-toggle').uncheck();
  await page.waitForFunction(()=>!document.getElementById('raw-rate-chart').layout.shapes.some(s=>s.name?.startsWith('retained-event:')));
  assert.equal(await page.evaluate(()=>JSON.stringify(document.getElementById('raw-rate-chart').data)),before);
  await page.locator('#raw-events-toggle').check();
  pass('earthquake markers survive new view and toggle without changing data');
  if(out)await page.locator('#raw-rate-panel').screenshot({path:path.join(out,'tokyo-lg-recorded-change.png')});
  await page.locator('#rate-basis').selectOption('raw');
  await page.waitForFunction(()=>!document.getElementById('rate-export').disabled&&rateRows.every(r=>r.basis==='raw'));
  assert.ok(await page.evaluate(()=>rateRows.filter(r=>r.method==='lambda_geo').every(r=>r.change_per_day===null)));
  pass('support-matched mode does not invent missing GNSS support');
  await page.locator('#rate-basis').selectOption('recorded_raw');
  await page.locator('#raw-region').selectOption('kumamoto');
  await page.waitForFunction(()=>!document.getElementById('rate-export').disabled&&rateRows.every(r=>r.region==='kumamoto'));
  const k=await page.evaluate(()=>rateRows.find(r=>r.method==='lambda_geo'&&r.date==='2026-11-04'));
  assert.ok(Math.abs(k.change_per_day-0.08076795224370514)<1e-12);
  pass('Kumamoto change survives region switch');
  assert.match(await page.locator('#raw-lg-summary').innerText(),/LG EXCLUDED/);
  pass('Kumamoto audit updates with region, not stale Tokyo values');
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>{const b=document.querySelector('#raw-rate-chart svg.main-svg').getBoundingClientRect();return b.width<=390&&b.right<=391;});
  await page.locator('#raw-rate-panel').scrollIntoViewIfNeeded();
  const box=await page.locator('#raw-rate-chart svg.main-svg').first().boundingBox();
  assert.ok(box.width<=390&&box.x+box.width<=391);
  const controls=await page.locator('#rate-prev').boundingBox();
  assert.ok(controls.y>=box.y+box.height,'controls must be below the chart, not overlaid');
  if(out)await page.locator('#raw-rate-panel').screenshot({path:path.join(out,'kumamoto-lg-mobile.png')});
  pass('mobile chart fits viewport');
  if(integrated)for(const sample of integrated.cases){
   const got=await page.evaluate(s=>{
    const body={region:'tokyo_kanto',days:[s.day]};renderRawLgAudit(body,'lambda_geo');
    return {audit:rawLgAudit(s.day),rows:rawTableRows(body.days,'lambda_geo')};
   },sample);
   const a=got.audit,e=sample.expected;
   assert.equal(a.admission,e.included?'INCLUDED':'EXCLUDED');
   assert.ok(Math.abs(a.effective_weight-e.weight)<1e-12);
   assert.ok(Math.abs(a.weighted_contribution-e.contribution)<1e-12);
   assert.equal(a.observation_age_days,sample.name==='excluded-clock'?null:9);
   assert.ok(Math.abs(Number(got.rows[0].lg_weighted_contribution)-e.contribution)<1e-12);
   assert.equal(got.rows[0].lg_input_sha256,sample.day.methods.lambda_geo.calibration.baseline_provenance.measurement.input_sha256);
   assert.match(await page.locator('#raw-lg-summary').innerText(),new RegExp('to recorded combined '+String(e.combined).replace('.','\\.')));
   if(sample.name==='zero-score')assert.match(await page.locator('#raw-lg-summary').innerText(),/Included zero-score LG/);
   if(out)await page.locator('#raw-lg-audit').screenshot({path:path.join(out,'audit-'+sample.name+'.png')});
   pass('real combiner → archive exporter → UI and CSV: '+sample.name);
  }
  assert.deepEqual(errors,[]);pass('no page JavaScript errors');
  const receipt={utc:new Date().toISOString(),passed:checks.length,browser:browser.version(),checks,errors,
   scope:'Local retained archive and synthetic-provenance real-combiner fixtures; no hosted deployment or efficacy claim'};
  if(out)fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt,null,2));
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
