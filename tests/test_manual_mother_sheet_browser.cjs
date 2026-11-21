// Run with Playwright available and the repository's Python PDF dependencies installed.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const docs = path.resolve(__dirname, '../docs');
const fixture = JSON.parse(execFileSync(path.resolve(__dirname, '../.venv/bin/python'), ['-c',
  "import sys,json,base64;sys.path.insert(0,'tests');from test_manual_mother_sheet import sample_order,build_manual_mother_sheet_pdf;o=sample_order();print(json.dumps({'order':o,'pdf':base64.b64encode(build_manual_mother_sheet_pdf(o)).decode()}))"
], {cwd:path.resolve(__dirname, '..'), encoding:'utf8'}));
const order = fixture.order;
order.id = 31;
const pdf = Buffer.from(fixture.pdf, 'base64');
const server = http.createServer((req, res) => {
  const file = path.join(docs, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  fs.readFile(file, (error, data) => {
    res.writeHead(error ? 404 : 200, {'Content-Type': ({'.html':'text/html', '.js':'application/javascript', '.css':'text/css'})[path.extname(file)] || 'application/octet-stream'});
    res.end(error ? '' : data);
  });
});
server.listen(0, '127.0.0.1', async () => {
  let browser;
  try {
    browser = await chromium.launch({headless:true, executablePath:'/usr/bin/chromium', args:['--no-sandbox']});
    const page = await browser.newPage();
    const base = `http://127.0.0.1:${server.address().port}/`;
    const requests = [];
    await page.route('**/*', route => {
      if (route.request().url().startsWith(base)) return route.continue();
      const url = new URL(route.request().url());
      if (url.pathname.startsWith('/manual-orders')) requests.push([route.request().method(), url.pathname, url.search]);
      if (url.pathname.endsWith('/mother-sheet.pdf')) return route.fulfill({contentType:'application/pdf', body:pdf});
      if (url.pathname === '/manual-orders/31') return route.fulfill({json:order});
      if (url.pathname === '/manual-orders') return route.fulfill({json:{items:[order], has_more:false}});
      return route.fulfill({json:{}});
    });
    await page.goto(base, {waitUntil:'networkidle'});
    await page.evaluate(async () => {activateTab('manual'); await openManualOrder(31, true);});
    const before = await page.evaluate(() => JSON.stringify(manualOrdersState.rows));
    // Print must keep original items even when dimension grouping is selected.
    await page.locator('#manualOrderGroupDimensions').click();
    const popupEvent = page.waitForEvent('popup');
    await page.locator('#manualRedIndexPrint').click();
    const popup = await popupEvent;
    await popup.waitForURL('blob:**');
    assert.deepEqual(requests.filter(r => r[1].endsWith('.pdf')), [['GET', '/manual-orders/31/mother-sheet.pdf', '']]);
    assert(requests.every(r => r[0] === 'GET'));
    assert.equal(await page.evaluate(() => JSON.stringify(manualOrdersState.rows)), before);
    assert.equal(await page.locator('#manualOrderError').isVisible(), false);
    console.log('Manual Orders Print opens the multi-spec Mother Sheet PDF; original rows, grouping isolation and GET-only behavior verified.');
  } catch (error) {console.error(error); process.exitCode = 1;}
  finally {if (browser) await browser.close(); server.close();}
});
