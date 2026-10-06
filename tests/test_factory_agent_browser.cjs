// Isolated UI/lifecycle contract tests. No requests reach production or OpenAI.
const {chromium, webkit} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const root = path.resolve(__dirname, '../docs');
const output = process.env.FACTORY_AGENT_QA_OUTPUT || '/tmp/factory-agent-browser-qa';
fs.mkdirSync(output, {recursive: true});
const config = {
  enabled: true, ready: true, state: 'ready', missing: [], mode: 'inspect_and_prepare', auth: 'app_key',
  orders: [{id: 'fixture:factory-agent-001', label: 'Factory Agent test order · fixture only'}, {id: 'factory:workspace', label: 'Factory workspace · inspect and prepare'}],
  capabilities: ['Read factory orders', 'Prepare change plans for review'],
  limits: {runtime_seconds: 180, max_concurrency: 1},
};
const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';
const initialSession = request => ({
  id: `session-${request.request_id}`, request_id: request.request_id, order_id: request.order_id,
  message: request.message, status: 'running', created_at: '2026-10-06T12:00:00Z',
  activity: [{id: 'event-1', title: 'Hosted browser opened the selected fixture', type: 'browser', status: 'received'}],
  result_text: '', screenshot: null, cleanup_status: 'pending',
});
async function run(engine, name, base) {
  const browser = await engine.launch({headless: true});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 1000}, colorScheme: 'light'});
    await context.addInitScript(base => localStorage.setItem('loe.apiBase', base), base);
    const page = await context.newPage();
    const errors = [], calls = [], submitted = [], productionWrites = [];
    let enabled = false, ready = true, lostPost = false, sessionListDown = false, stopFailure = false, appKeyConfigured = false, authorized = true;
    let session = null;
    let reviewFailure = false;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url());
      if (!['GET', 'HEAD', 'OPTIONS'].includes(req.method()) && !url.pathname.startsWith('/api/factory-agent/')) productionWrites.push(url.pathname);
      if (url.origin !== base) return route.fulfill({status: 200, json: {}});
      if (req.headers().accept?.includes('text/event-stream')) return route.fulfill({status: 204, body: ''});
      if (url.pathname === '/api/features') return route.fulfill({json: {factory_agent: enabled}});
      if (url.pathname.startsWith('/api/factory-agent/')) {
        calls.push({path: url.pathname, method: req.method(), key: req.headers()['x-app-key']});
        if (url.pathname.endsWith('/config') && !appKeyConfigured) return route.fulfill({status: 503, json: {detail: {code: 'setup_required', missing: ['FACTORY_AGENT_ACCESS_KEY'], message: 'Configure Factory Agent access before starting tasks.'}}});
        if (!authorized || req.headers()['x-app-key'] !== 'fixture-access-key') return route.fulfill({status: 401, json: {detail: 'Unauthorized'}});
        if (url.pathname.endsWith('/config')) return route.fulfill({json: {...config, ready, missing: ready ? [] : ['OPENAI_API_KEY'], error: ready ? null : 'Server configuration unavailable.'}});
        if (url.pathname.endsWith('/sessions') && req.method() === 'POST') {
          const body = req.postDataJSON();
          assert(['fixture:factory-agent-001', 'factory:workspace'].includes(body.order_id));
          submitted.push(body);
          assert(body.request_id.match(/^[a-f0-9-]{36}$/));
          session = initialSession(body);
          if (lostPost) return route.abort('failed');
          return route.fulfill({json: session});
        }
        if (url.pathname.endsWith('/sessions')) {
          if (sessionListDown) return route.fulfill({status: 503, json: {detail: 'Session store unavailable'}});
          const summary = session && Object.fromEntries(['id', 'request_id', 'order_id', 'created_at', 'deadline_at', 'status', 'cleanup_status', 'error', 'retired'].map(key => [key, session[key]]));
          return route.fulfill({json: {sessions: summary ? [summary] : []}});
        }
        if (url.pathname.endsWith('/stop')) {
          assert.equal(req.method(), 'POST');
          if (stopFailure) return route.fulfill({status: 502, json: {detail: 'Remote cancellation unavailable'}});
          session.status = 'stopping';
          return route.fulfill({json: session});
        }
        if (url.pathname.endsWith('/review')) {
          assert.equal(req.method(), 'POST');
          const proposalId = decodeURIComponent(url.pathname.split('/').at(-2));
          const body = req.postDataJSON();
          assert.deepEqual(Object.keys(body), ['decision']);
          if (reviewFailure) return route.fulfill({status: 502, json: {detail: 'Review unavailable'}});
          session.proposals.find(proposal => proposal.id === proposalId).status = body.decision;
          return route.fulfill({json: {reviewed: true}});
        }
        if (req.method() === 'DELETE') { session = null; return route.fulfill({json: {deleted: true}}); }
        return route.fulfill({json: session});
      }
      if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/orders') || url.pathname.startsWith('/manual-orders') || url.pathname.startsWith('/events/') || url.pathname.startsWith('/analysis') || url.pathname.startsWith('/telegram-files') || url.pathname === '/healthz') {
        return route.fulfill({json: {items: [], groups: {}, counts: {}}});
      }
      return route.continue();
    });
    await page.goto(base, {waitUntil: 'networkidle'});
    assert(await page.locator('#factoryAgentNav').isHidden(), 'flag off hides section');
    await page.evaluate(() => activateTab('factoryagent'));
    assert(await page.locator('#tabFactoryAgent').isHidden(), 'flag off blocks direct navigation');
    assert.equal(calls.length, 0, 'disabled feature never touches agent endpoints');
    enabled = true;
    await page.reload({waitUntil: 'networkidle'});
    await page.locator('#factoryAgentNav').click();
    assert.equal(await page.locator('#pageTitle').innerText(), 'Factory Agent · Beta');
    await page.locator('#factoryAgentSetup').waitFor({state: 'visible'});
    assert.equal(await page.locator('#factoryAgentMissing').innerText(), 'FACTORY_AGENT_ACCESS_KEY');
    assert.equal(await page.locator('#factoryAgentSetupText').innerText(), 'Configure Factory Agent access before starting tasks.');
    assert(await page.locator('#factoryAgentUnlockForm').isHidden());
    assert(await page.locator('#factoryAgentWorkspace').isHidden());
    assert.equal(calls.filter(call => call.path.includes('/sessions')).length, 0, 'setup preflight never requests session access');
    await page.screenshot({path: path.join(output, `${name}-missing-access-key.png`), fullPage: true});
    if (name === 'chromium') {
      const cleanOutput = path.resolve(__dirname, '../output/factory-agent');
      fs.mkdirSync(cleanOutput, {recursive: true});
      await page.screenshot({path: path.join(cleanOutput, 'setup-required.png'), fullPage: true});
    }
    appKeyConfigured = true;
    await page.locator('#factoryAgentCheckSetup').click();
    await page.locator('#factoryAgentUnlockForm').waitFor({state: 'visible'});
    assert(await page.locator('#factoryAgentSetup').isHidden(), 'normal 401 returns to unlock, not a setup error');
    assert.equal(await page.locator('label[for="factoryAgentKey"]').innerText(), 'Factory Agent access key');
    assert.match(await page.locator('#factoryAgentKeyHint').innerText(), /APP_KEY.*Do not enter an OpenAI API key/);
    await page.locator('#factoryAgentKey').fill('wrong-key');
    await page.locator('#factoryAgentUnlock').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentNotice').textContent.includes('not accepted'));
    assert.equal(await page.locator('#factoryAgentKey').inputValue(), '', 'clear key field immediately');
    assert(await page.locator('#factoryAgentWorkspace').isHidden());
    await page.locator('#factoryAgentKey').fill('fixture-access-key');
    await page.locator('#factoryAgentUnlock').click();
    await page.waitForFunction(() => !document.getElementById('factoryAgentStart').disabled);
    assert.equal(await page.locator('#factoryAgentResult').innerText(), 'No response received yet.');
    assert(await page.locator('#factoryAgentScreenshot').isHidden());
    assert.equal(await page.locator('#factoryAgentActivity li').count(), 0, 'no invented events');
    assert(!(await page.evaluate(() => JSON.stringify({...localStorage, ...sessionStorage}))).includes('fixture-access-key'));
    assert.equal(await page.locator('#factoryAgentOrder').inputValue(), 'factory:workspace', 'factory workspace is the default');
    assert.equal(await page.locator('#factoryAgentMessage').inputValue(), '');
    await page.locator('[data-factory-example="read"]').click();
    assert.match(await page.locator('#factoryAgentMessage').inputValue(), /most recent orders/);
    assert.equal(submitted.length, 0, 'examples only fill an editable request');
    await page.locator('#factoryAgentMessage').fill('');
    await page.locator('#factoryAgentOrder').selectOption('fixture:factory-agent-001');
    assert.match(await page.locator('#factoryAgentMessage').inputValue(), /selected test order/);
    assert.match(await page.locator('#factoryAgentBoundaryText').innerText(), /fixture tool reads only the synthetic order/);
    assert.match(await page.locator('#factoryAgentBoundaryText').innerText(), /including other public pages on that host/);
    assert.match(await page.locator('#factoryAgentBoundaryText').innerText(), /Production API access and operational actions remain blocked/);

    // A failed preflight must not create remote work.
    sessionListDown = true;
    await page.locator('#factoryAgentStart').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentNotice').textContent.includes('could not be verified'));
    assert.equal(calls.filter(call => call.method === 'POST').length, 0);
    sessionListDown = false;
    await page.locator('#factoryAgentStart').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Running');
    assert.equal(calls.filter(call => call.method === 'POST').length, 1);
    assert.equal(submitted[0].order_id, 'fixture:factory-agent-001');
    assert(!('context_session_id' in submitted[0]), 'context is never included by default');
    assert(await page.locator('#factoryAgentStart').isDisabled());
    assert.equal(await page.locator('#factoryAgentActivity li').count(), 1);
    await page.screenshot({path: path.join(output, `${name}-running-light.png`), fullPage: true});

    stopFailure = true;
    await page.locator('#factoryAgentStop').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentNotice').textContent.includes('Cancellation is not confirmed'));
    assert.equal(await page.locator('#factoryAgentStatus').innerText(), 'Running', 'failed cancellation is not shown as success');
    stopFailure = false;
    await page.locator('#factoryAgentStop').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Stopping');
    assert.equal(calls.filter(call => call.path.endsWith('/stop') && call.method === 'POST').length, 2, 'Stop sends cancellation requests');
    assert(await page.locator('#factoryAgentStop').isDisabled());
    assert(await page.locator('#factoryAgentStart').isDisabled(), 'stopping is not cancelled');
    session.status = 'cancelled'; session.cleanup_status = 'pending';
    await page.locator('#factoryAgentReconnect').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Cancelled');
    assert(await page.locator('#factoryAgentStart').isDisabled(), 'terminal turn still reserves a workspace until deletion is confirmed');
    session.cleanup_status = 'deleted';
    await page.waitForFunction(() => !document.getElementById('factoryAgentStart').disabled);

    // Refreshing and re-unlocking recover a session without another POST.
    const postCount = calls.filter(call => call.method === 'POST').length;
    await page.reload({waitUntil: 'networkidle'});
    await page.locator('#factoryAgentNav').click();
    assert(await page.locator('#factoryAgentWorkspace').isHidden());
    await page.locator('#factoryAgentKey').fill('fixture-access-key');
    await page.locator('#factoryAgentUnlock').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Cancelled');
    assert.equal(calls.filter(call => call.method === 'POST').length, postCount);
    await page.locator('#factoryAgentCleanup').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'No session');
    await page.locator('#factoryAgentOrder').selectOption('fixture:factory-agent-001');

    // A dropped POST response is recovered by request id, never replayed.
    lostPost = true;
    await page.locator('#factoryAgentStart').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Awaiting confirmation');
    const unknownPostCount = calls.filter(call => call.method === 'POST').length;
    assert(await page.locator('#factoryAgentStart').isDisabled());
    await page.locator('#factoryAgentReconnect').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Running');
    assert.equal(calls.filter(call => call.method === 'POST').length, unknownPostCount);
    session.status = 'completed'; session.cleanup_status = 'deleted';
    session.activity.push({id: 'event-2', title: '<img src=x onerror="window.bad=1">', type: 'tool', status: 'completed'});
    session.result_text = 'Test fixture response\nClient: Factory Test Client\nGlass: 4F + 16 + 4 LowE\nDimensions: 800 × 1200 mm; quantity: 2; index: 001; position: A1\nAmbiguity: index is missing on the final source row.\n<script>window.bad=1</script>';
    session.screenshot = 'https://example.com/unsafe-image';
    await page.locator('#factoryAgentReconnect').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Completed');
    assert(await page.locator('#factoryAgentScreenshot').isHidden(), 'do not fetch arbitrary remote screenshots');
    assert.equal(await page.locator('#factoryAgentResult script').count(), 0);
    assert.equal(await page.locator('#factoryAgentActivity img').count(), 0);
    assert.equal(await page.evaluate(() => window.bad), undefined);
    session.screenshot = png;
    await page.locator('#factoryAgentReconnect').click();
    await page.locator('#factoryAgentScreenshot').waitFor({state: 'visible'});
    for (const [width, colorScheme] of [[1440, 'light'], [1280, 'light'], [1024, 'dark'], [390, 'dark']]) {
      await page.setViewportSize({width, height: 1000});
      await page.emulateMedia({colorScheme});
      const overflow = await page.locator('#tabFactoryAgent').evaluate(element => ({width: element.clientWidth, scroll: element.scrollWidth, right: element.getBoundingClientRect().right, screen: innerWidth}));
      assert(overflow.scroll <= overflow.width + 1 && overflow.right <= overflow.screen + 1, `responsive fit ${JSON.stringify(overflow)}`);
      await page.screenshot({path: path.join(output, `${name}-${width}-${colorScheme}.png`), fullPage: true});
    }
    await page.setViewportSize({width: 1440, height: 1000});
    await page.locator('#factoryAgentOrder').selectOption('factory:workspace');
    assert(await page.locator('#factoryAgentContinue').isDisabled(), 'fixture context is not silently reused for factory data');
    await page.locator('[data-factory-example="draft"]').click();
    await page.locator('#factoryAgentMessage').fill('Prepare a plan to correct order 123 from 800 mm to 810 mm. Explain the evidence and do not change the order.');
    lostPost = false;
    await page.locator('#factoryAgentStart').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Running');
    assert.equal(submitted.at(-1).order_id, 'factory:workspace');
    assert(!('context_session_id' in submitted.at(-1)));
    assert(await page.locator('#factoryAgentContinue').isDisabled());
    session.status = 'completed'; session.cleanup_status = 'deleted';
    session.result_text = 'Mock browser test response: two proposed plans are ready for operator review. No production records were changed.';
    session.proposals = [
      {id: 'plan-1', title: 'Correct the proposed width', summary: 'Source evidence needs operator review.', order_id: '123', source_version: 'snapshot-123', changes: [{field: 'width_mm', before: 800, after: 810, reason: '<img src=x onerror="window.bad=1">'}], status: 'pending'},
      {id: 'plan-2', title: '<script>window.bad=1</script>', summary: 'Review the missing position.', order_id: '123', changes: [{field: 'position', before: null, after: 'A1'}], status: 'pending'},
    ];
    await page.locator('#factoryAgentReconnect').click();
    await page.waitForFunction(() => document.querySelectorAll('[data-factory-proposal]').length === 4);
    assert.equal(await page.locator('#factoryAgentProposals script, #factoryAgentProposals img').count(), 0);
    assert.match(await page.locator('.factory-agent-change').first().innerText(), /Width \(mm\)[\s\S]*Before[\s\S]*800[\s\S]*Proposed[\s\S]*810/);
    assert.equal(await page.evaluate(() => window.bad), undefined);
    assert.match(await page.locator('.factory-agent-plans').innerText(), /does not change an order/);
    reviewFailure = true;
    await page.locator('[data-factory-proposal="plan-1"][data-factory-decision="accepted"]').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentNotice').textContent.includes('review was not confirmed'));
    assert.equal(await page.locator('.factory-agent-proposal').first().locator('.factory-agent-badge').innerText(), 'Awaiting review');
    assert(await page.locator('[data-factory-proposal="plan-1"][data-factory-decision="accepted"]').isDisabled());
    reviewFailure = false;
    await page.locator('#factoryAgentReconnect').click();
    await page.waitForFunction(() => !document.querySelector('[data-factory-proposal="plan-1"]').disabled);
    await page.locator('[data-factory-proposal="plan-1"][data-factory-decision="accepted"]').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentProposals').textContent.includes('Plan accepted'));
    assert.match(await page.locator('#factoryAgentNotice').innerText(), /No production changes were applied/);
    await page.locator('[data-factory-proposal="plan-2"][data-factory-decision="rejected"]').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentProposals').textContent.includes('Dismissed'));
    assert.equal(await page.locator('#factoryAgentProposals [data-factory-proposal]').count(), 0);
    for (const width of [1440, 390]) {
      await page.setViewportSize({width, height: 1000});
      await page.evaluate(() => window.scrollTo(0, 0));
      assert(await page.locator('#tabFactoryAgent').evaluate(element => element.scrollWidth <= element.clientWidth + 1));
      await page.screenshot({path: path.join(output, `${name}-plans-${width}.png`), fullPage: true});
    }
    await page.setViewportSize({width: 1440, height: 1000});
    const previousSessionId = session.id;
    assert(!(await page.locator('#factoryAgentContinue').isChecked()));
    await page.locator('#factoryAgentContinue').check();
    await page.locator('#factoryAgentMessage').fill('What source evidence supports the accepted plan?');
    await page.locator('#factoryAgentStart').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Running');
    assert.equal(submitted.at(-1).context_session_id, previousSessionId, 'only explicit continuation includes saved context');
    assert(!(await page.locator('#factoryAgentContinue').isChecked()), 'new session does not inherit checked continuation');
    session.status = 'completed'; session.cleanup_status = 'deleted';
    await page.locator('#factoryAgentReconnect').click();
    await page.waitForFunction(() => document.getElementById('factoryAgentStatus').textContent === 'Completed');
    const savedState = await page.evaluate(() => JSON.stringify({...localStorage, ...sessionStorage}));
    assert(!savedState.includes('What source evidence') && !savedState.includes('snapshot-123'));
    assert.deepEqual(productionWrites, [], 'review never calls production mutation routes');
    ready = false;
    await page.locator('#factoryAgentReconnect').click();
    await page.locator('#factoryAgentSetup').waitFor({state: 'visible'});
    assert(await page.locator('#factoryAgentStart').isDisabled());
    assert.match(await page.locator('#factoryAgentMissing').innerText(), /OPENAI_API_KEY/);
    assert.equal(await page.locator('#factoryAgentSetupText').innerText(), 'Server configuration unavailable.');
    await page.screenshot({path: path.join(output, `${name}-setup-required.png`), fullPage: true});
    authorized = false;
    await page.locator('#factoryAgentReconnect').click();
    await page.locator('#factoryAgentUnlockForm').waitFor({state: 'visible'});
    assert(await page.locator('#factoryAgentWorkspace').isHidden());
    assert.equal(await page.locator('#factoryAgentResult').innerText(), 'No response received yet.');
    assert.equal(await page.locator('#factoryAgentActivity li').count(), 0);
    assert.equal(await page.locator('#factoryAgentScreenshot').getAttribute('src'), null);
    assert.equal(await page.locator('#factoryAgentProposals').innerText(), '');
    assert.match(await page.locator('#factoryAgentNotice').innerText(), /Access denied/);
    assert.equal(errors.length, 0, errors.join('\n'));
    await context.close();
    console.log(`${name}: feature flag, auth, setup, session lifecycle, recovery, Stop, cleanup, XSS, screenshots and responsive checks passed`);
  } finally { await browser.close(); }
}
(async () => {
  const server = http.createServer((req, res) => {
    const target = path.resolve(root, `.${new URL(req.url, 'http://local').pathname === '/' ? '/index.html' : new URL(req.url, 'http://local').pathname}`);
    if (!target.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    fs.readFile(target, (error, content) => {
      if (error) { res.writeHead(404).end(); return; }
      res.setHeader('Content-Type', target.endsWith('.js') ? 'text/javascript' : target.endsWith('.css') ? 'text/css' : 'text/html');
      res.end(content);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try { await run(chromium, 'chromium', base); await run(webkit, 'webkit', base); }
  finally { await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
