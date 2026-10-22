// Headless Edge acceptance of the actual Pages chart functions. No production writes.
// Plotly's public CDN is the only network access; chart inputs are pinned local Git fixtures.
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process'), assert = require('assert');
const {chromium} = require(process.env.GEOSPEC_PLAYWRIGHT || 'C:/codex/.tools/environment-browser/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'docs/index.html'), 'utf8');
const names = ['parseCSV', 'historyValue', 'retrospectiveTrace', 'retrospectiveConnectors', 'renderChart', 'detectStressReleaseDrops', 'getDropOverlays'];
const code = names.map(name => html.match(new RegExp('^        (?:async )?function ' + name + '\\(.*?^        \\}', 'ms'))[0]).join('\n');
const csv = cp.execFileSync('git', ['show', '0eca6fab:docs/data.csv'], {cwd: root, encoding: 'utf8'});
const index = JSON.parse(cp.execFileSync('git', ['show', '3ec764e2:docs/raw_history/index.json'], {cwd: root, encoding: 'utf8'}));
(async () => {
    const browser = await chromium.launch({channel: 'msedge', headless: true});
    try {
        const page = await browser.newPage({viewport: {width: 1500, height: 900}}), errors = [];
        page.on('pageerror', e => errors.push(e.message));
        await page.route('**/*', route => route.request().url() === 'https://cdn.plot.ly/plotly-2.27.0.min.js'
            ? route.continue() : route.abort());
        await page.setContent('<body style="background:#15151e;color:white;font-family:sans-serif"><h2>Turkey: peak, first decrease, tier crossing and local low</h2>' +
            '<label><input id="show-retrospective" type="checkbox" checked>Grey retrospective reconstruction</label>' +
            '<p id="retrospective-status"></p><div id="history-chart" style="width:1400px;height:680px"></div>' +
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></body>');
        await page.addScriptTag({content: code});
        await page.evaluate(async ({csv, index}) => {
            window.fetch = async () => ({ok: true, json: async () => index});
            const rows = parseCSV(csv).filter(r => r.region === 'turkey_kahramanmaras' && r.date >= '2026-09-28' && r.date <= '2026-10-09');
            await renderChart(rows, [{region: 'turkey_kahramanmaras', time: '2026-10-04', mag: 5}], detectStressReleaseDrops(rows));
        }, {csv, index});
        await page.waitForFunction(() => document.querySelectorAll('#history-chart .scatterlayer .trace').length >= 3);
        const state = await page.evaluate(() => {
            const chart = document.getElementById('history-chart');
            return {traces: chart.data.map(t => ({name: t.name, x: t.x, y: t.y, line: t.line, text: t.text, visible: t.visible})),
                status: document.getElementById('retrospective-status').textContent, shapes: chart.layout.shapes};
        });
        const grey = state.traces.find(t => t.name === 'Grey reconstruction links (not issued)');
        assert(grey && grey.line.color === '#9ca3af' && grey.line.dash === 'dash');
        assert.deepEqual(grey.x, ['2026-10-06', '2026-10-07', null, '2026-10-07', '2026-10-08', null]);
        const decline = state.traces.find(t => t.name.endsWith('peak-to-low (retrospective)'));
        assert.deepEqual(decline.x, ['2026-10-02', '2026-10-03', '2026-10-04', '2026-10-05']);
        assert(decline.text[0].includes('intraday order unknown'));
        assert(state.shapes.some(s => s.x0 === '2026-10-02' && s.x1 === '2026-10-05'));
        assert(state.shapes.some(s => s.x0 === '2026-10-04' && s.x1 === '2026-10-07' && s.line.dash === 'dot'));
        if (process.argv[2]) await page.locator('body').screenshot({path: process.argv[2]});
        await page.locator('#show-retrospective').uncheck();
        await page.waitForFunction(() => document.getElementById('history-chart').data.filter(t => t.name.startsWith('Retrospective reconstruction') || t.name.startsWith('Grey reconstruction')).every(t => t.visible === false));
        assert.deepEqual(errors, []);
        console.log(JSON.stringify({result: 'BROWSER_PASS', checks: ['grey adjacent-day links', 'peak-to-low trace', 'same-day event wording', 'unmoved comparison window', 'both overlays toggle', 'no page errors'], status: state.status}));
    } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exitCode = 1;});
