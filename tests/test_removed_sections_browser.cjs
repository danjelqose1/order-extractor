// Local removal regression: real frontend, mocked backend, no production traffic.
const {chromium, webkit} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const root = path.resolve(__dirname, '../docs');
const output = process.env.REMOVED_SECTIONS_QA_OUTPUT || '/tmp/removed-sections-browser-qa';
fs.mkdirSync(output, {recursive: true});
const retired = ['workspace', 'beta', 'factoryagent'];
const panels = {
  extract: 'tabExtract', history: 'tabHistory', manual: 'tabManualOrders',
  processing: 'tabProcessing', perfectcut: 'tabPerfectCut', labels: 'tabLabels',
  telegram: 'tabTelegram', pdfeditor: 'tabPdfEditor', scanstudio: 'tabScanStudio',
  analysis: 'tabAnalysis', settings: 'tabSettings',
};
async function run(engine, name, base) {
  const browser = await engine.launch({headless: true});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
    await context.addInitScript(base => {
      localStorage.setItem('loe.apiBase', base);
      // Removing Beta must not resume a previously saved recorder session.
      localStorage.setItem('betaTeachingSessionId', 'removed-section-fixture');
      sessionStorage.setItem(`loe.factory-agent.v1:${base}`, JSON.stringify({sessionId: 'retired-fixture'}));
    }, base);
    const page = await context.newPage(), errors = [], retiredCalls = [], writes = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const request = route.request(), url = new URL(request.url());
      if (/^\/api\/(factory-agent|beta)(\/|$)/.test(url.pathname)) retiredCalls.push(url.pathname);
      if (!['GET', 'HEAD', 'OPTIONS'].includes(request.method())) writes.push(url.pathname);
      if (url.origin !== base) return route.fulfill({status: 200, body: ''});
      if (request.headers().accept?.includes('text/event-stream')) return route.fulfill({status: 204, body: ''});
      if (url.pathname === '/api/features') return route.fulfill({json: {living_dashboard: false, factory_agent: true}});
      if (/^\/(api|orders|manual-orders|events|analysis|telegram-files|healthz)(\/|$)/.test(url.pathname)) {
        return route.fulfill({json: {items: [], groups: {}, counts: {}, orders: [], jobs: [], files: [], rows: []}});
      }
      return route.continue();
    });
    await page.goto(base, {waitUntil: 'networkidle'});
    for (const tab of retired) assert.equal(await page.locator(`[data-tab="${tab}"]`).count(), 0, `${tab} navigation removed`);
    for (const id of ['tabWorkspace', 'tabBeta', 'tabFactoryAgent', 'betaTeachingBar', 'betaDecisionReasonModal']) {
      assert.equal(await page.locator(`#${id}`).count(), 0, `${id} removed`);
    }
    assert.equal(await page.locator('script[src*="factory-agent"]').count(), 0);
    // A stale enabled backend flag cannot reintroduce a removed section.
    assert.equal(await page.evaluate(() => typeof window.FactoryAgentUI), 'undefined');

    for (const [tab, id] of Object.entries(panels)) {
      const button = page.locator(`#primaryNav [data-tab="${tab}"]`);
      assert.equal(await button.count(), 1, `${tab} remains navigable`);
      await button.click();
      assert(await page.locator(`#${id}`).isVisible(), `${tab} panel opens`);
      assert.equal(await page.locator('.tab-panel.active:visible').count(), 1, `${tab} has a single active panel`);
    }
    // Programmatic stale navigation must keep the app usable, never an empty page.
    for (const tab of retired) {
      await page.evaluate(tab => activateTab(tab), tab);
      assert.equal(await page.locator('.tab-panel.active:visible').count(), 1, `${tab} fallback has an active panel`);
      assert(await page.locator('#tabExtract').isVisible(), `${tab} falls back to Overview`);
    }
    // Opening retained modules is read-only and must not restart retired agents.
    assert.deepEqual(retiredCalls, []);
    assert.deepEqual(writes, []);
    assert.deepEqual(errors, []);
    await page.screenshot({path: path.join(output, `${name}-desktop.png`), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#mobileNavToggle').click();
    await page.locator('#primaryNav [data-tab="processing"]').click();
    assert(await page.locator('#tabProcessing').isVisible());
    assert.equal(await page.locator('#mobileNavToggle').getAttribute('aria-expanded'), 'false');
    await page.locator('#mobileNavToggle').click();
    await page.locator('#primaryNav [data-tab="labels"]').click();
    assert(await page.locator('#tabLabels').isVisible());
    assert.deepEqual(retiredCalls, []);
    assert.deepEqual(errors, []);
    await page.screenshot({path: path.join(output, `${name}-mobile.png`), fullPage: true});
    await context.close();
    console.log(`${name}: retained navigation, mobile menu, retired-route fallback and no retired agent requests passed`);
  } finally { await browser.close(); }
}
(async () => {
  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url, 'http://local').pathname;
    const target = path.resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
    if (!target.startsWith(root + path.sep)) return res.writeHead(403).end();
    fs.readFile(target, (error, content) => {
      if (error) return res.writeHead(404).end();
      res.setHeader('Content-Type', target.endsWith('.js') ? 'text/javascript' : target.endsWith('.css') ? 'text/css' : 'text/html');
      res.end(content);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try { await run(chromium, 'chromium', base); await run(webkit, 'webkit', base); }
  finally { await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
