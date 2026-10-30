// Real page/assets and retained local data. Only Plotly loads from its public CDN.
'use strict';
const fs=require('node:fs'), path=require('node:path'), http=require('node:http'), assert=require('node:assert/strict');
const {chromium}=require(process.env.GEOSPEC_PLAYWRIGHT || 'C:/codex/.tools/environment-browser/node_modules/playwright');
const root=path.resolve(__dirname,'../docs'), out=process.argv[2] && path.resolve(process.argv[2]);
const server=http.createServer((req,res)=>{
 const file=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/\/$/,'/index.html'));
 if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
 fs.readFile(file,(err,data)=>{if(err){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'application/javascript','.json':'application/json','.csv':'text/csv'})[path.extname(file)] || 'text/plain');res.end(data);});
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const checks=[], pass=name=>{checks.push(name);console.log('PASS '+name);};
 try {
  if(out)fs.mkdirSync(out,{recursive:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  let csvMode='abort';
  await page.route('**/data.csv',r=>csvMode==='abort'?r.abort('failed'):csvMode==='503'?r.fulfill({status:503,body:'unavailable'}):csvMode==='malformed'?r.fulfill({status:200,body:'<html>not CSV</html>'}):r.continue());
  await page.goto('http://127.0.0.1:'+server.address().port+'/',{waitUntil:'networkidle'});
  await page.waitForFunction(()=>document.getElementById('history-feedback').dataset.state==='error');
  assert.match(await page.locator('#history-load-status').innerText(),/blank panel is not evidence/);
  assert.equal(await page.locator('#status-grid .region-card').count(),14);
  assert.equal(await page.locator('#history-retry').isVisible(),true);pass('CSV failure visible while cards survive');
  csvMode='ok';await page.locator('#history-retry').click();
  await page.waitForFunction(()=>document.getElementById('history-feedback').dataset.state==='ready');
  assert.ok(await page.evaluate(()=>document.getElementById('history-chart').data.length>0));pass('retry restores history');
  for(const mode of ['503','malformed']) {
   csvMode=mode;await page.evaluate(()=>loadHistory());
   assert.equal(await page.locator('#history-feedback').getAttribute('data-state'),'error');
   assert.match(await page.locator('#history-load-status').innerText(),/last successful chart may be stale/);
   assert.ok(await page.evaluate(()=>document.getElementById('history-chart').data.length>0));pass(mode+' failure identifies stale chart');
  }
  csvMode='ok';
  await page.evaluate(async()=>{const original=Plotly.newPlot;Plotly.newPlot=(id,...args)=>id==='history-chart'?Promise.reject(new Error('test async renderer failure')):original(id,...args);try{await loadHistory();}finally{Plotly.newPlot=original;}});
  assert.match(await page.locator('#history-load-status').innerText(),/test async renderer failure/);pass('asynchronous renderer rejection handled');
  await page.locator('#history-retry').click();await page.waitForFunction(()=>document.getElementById('history-feedback').dataset.state==='ready');
  await page.waitForFunction(()=>!document.getElementById('raw-export').disabled);
  assert.equal(await page.locator('#raw-rate-panel').isVisible(),false);pass('values-only default preserved');
  await page.locator('#open-history-rate').click();await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  assert.match(await page.locator('#rate-summary').innerText(),/214 known \/ 373/);
  assert.deepEqual(await page.evaluate(()=>document.getElementById('raw-rate-chart').layout.xaxis.range),['2026-07-30','2026-10-27']);pass('90-day default range');
  const rawRate=await page.evaluate(()=>document.getElementById('raw-rate-chart').data[0].y.slice());
  assert.ok(rawRate.some(x=>x===null)&&rawRate.some(x=>x<0)&&rawRate.some(x=>x>0));pass('real Hualien signed rates and null gaps');
  assert.match(await page.locator('#rate-detail').innerText(),/unit: not retained; method version: not retained/);pass('missing unit and version explicit');
  await page.locator('#rate-day').focus();await page.keyboard.press('Home');await page.keyboard.press('ArrowDown');
  await page.locator('#rate-next').click();assert.match(await page.locator('#rate-detail').innerText(),/UNKNOWN|report day =/);pass('keyboard endpoint inspection');
  const pending=page.waitForEvent('download');await page.locator('#rate-export').click();const download=await pending;
  const csv=fs.readFileSync(await download.path(),'utf8');assert.match(csv,/current_source_sha256/);assert.match(csv,/NO_PREVIOUS_DAY|MEASUREMENT_UNKNOWN/);assert.equal(csv.trim().split('\n').length,374);pass('full CSV with reasons and source hashes');
  await page.evaluate(()=>Plotly.relayout('raw-chart',{'xaxis.range':['2026-09-01','2026-10-01']}));
  await page.waitForFunction(()=>document.getElementById('raw-rate-chart').layout.xaxis.range[0]==='2026-09-01');pass('value and rate zoom synchronized');
  await page.locator('#rate-window').selectOption('all');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  assert.ok(await page.evaluate(()=>document.getElementById('raw-rate-chart').layout.yaxis.range[1]>5));pass('all-days range restores outlier without clipping');
  await page.locator('#rate-window').selectOption('30');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  assert.deepEqual(await page.evaluate(()=>document.getElementById('raw-rate-chart').layout.xaxis.range),['2026-09-28','2026-10-27']);pass('30-day range');
  await page.locator('#rate-window').selectOption('90');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  await page.locator('#rate-basis').selectOption('score');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  assert.notDeepEqual(await page.evaluate(()=>document.getElementById('raw-rate-chart').data[0].y),rawRate);pass('score basis uses score differences');
  await page.locator('#rate-basis').selectOption('raw');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  if(out){await page.locator('#raw-rate-panel').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'desktop-rate.png')});}
  await page.locator('#raw-method').selectOption('lambda_geo');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  assert.ok(await page.evaluate(()=>document.getElementById('raw-rate-chart').data[0].y.every(x=>x===null)));pass('unknown LG support cannot fabricate rates');
  await page.locator('#raw-method').selectOption('seismic_thd');await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);
  let regionFailure=true;await page.route('**/raw_history/istanbul_marmara.json',r=>regionFailure?r.abort('failed'):r.continue());
  await page.locator('#raw-region').selectOption('istanbul_marmara');await page.waitForFunction(()=>!document.getElementById('raw-retry').hidden);
  assert.equal(await page.locator('#raw-export').isDisabled(),true);assert.equal(await page.locator('#rate-export').isDisabled(),true);
  assert.equal(await page.evaluate(()=>!!document.getElementById('raw-chart').data),false);pass('failed region cannot export or display previous region');
  regionFailure=false;await page.locator('#raw-retry').click();await page.waitForFunction(()=>!document.getElementById('rate-export').disabled);pass('region retry recovers');
  await page.route('**/raw_history/turkey_kahramanmaras.json',async r=>{await new Promise(resolve=>setTimeout(resolve,500));await r.continue();});
  await page.locator('#raw-region').selectOption('turkey_kahramanmaras');await page.locator('#raw-region').selectOption('hualien');
  await page.waitForTimeout(700);assert.match(await page.locator('#raw-status').innerText(),/region hualien/);pass('late region response cannot replace selection');
  await page.evaluate(async()=>{const original=window.fetch;window.fetch=(url,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('aborted','AbortError'))));try{await fetchDashboard('timeout-fixture');}catch(e){window.timeoutResult=e.message;}finally{window.fetch=original;}});
  assert.match(await page.evaluate(()=>window.timeoutResult),/timed out after 15 seconds/);pass('bounded request timeout');
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(600);await page.locator('#raw-rate-panel').scrollIntoViewIfNeeded();
  const plotBox=await page.locator('#raw-rate-chart .svg-container').boundingBox();
  assert.ok(plotBox.x+plotBox.width<=391,JSON.stringify(plotBox));pass('mobile rate plot fits viewport');
  const clipped=await page.locator('#raw-first-section select, #raw-first-section button').evaluateAll(nodes=>nodes.filter(n=>n.getClientRects().length).filter(n=>{const b=n.getBoundingClientRect();return b.left<0||b.right>innerWidth+1;}).map(n=>n.id));
  assert.deepEqual(clipped,[]);pass('mobile controls not clipped');
  if(out)await page.screenshot({path:path.join(out,'mobile-rate.png')});
  await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>document.body.style.zoom='2');await page.locator('#rate-day').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('#rate-day').isVisible(),true);await page.locator('#rate-prev').click();pass('200 percent CSS zoom controls usable');
  assert.deepEqual(errors,[]);pass('no uncaught page errors');
  const report={utc:new Date().toISOString(),browser:browser.version(),checks,passed:checks.length,errors,scope:'Local page and retained data, synthetic request failures; no hosted jobs or data writes'};
  if(out)fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
 } finally {await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
