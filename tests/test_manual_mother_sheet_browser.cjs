// Run with Playwright available and the repository's Python PDF dependencies installed.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const docs = path.resolve(__dirname, '../docs');
const fixture = JSON.parse(execFileSync(path.resolve(__dirname, '../.venv/bin/python'), ['-c',
  "import sys,json,base64;sys.path.insert(0,'tests');from test_manual_mother_sheet import sample_order,build_manual_mother_sheet_pdf;o=sample_order();print(json.dumps({'order':o,'pdfs':{layout:base64.b64encode(build_manual_mother_sheet_pdf(o,layout)).decode() for layout in ['a4_portrait','a4_portrait_2up','a4_landscape_2up']}}))"
], {cwd:path.resolve(__dirname, '..'), encoding:'utf8'}));
const order = fixture.order;
order.id = 31;
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
    let oversized = false;
    let twoCopySupport = true;
    await page.route('**/*', route => {
      if (route.request().url().startsWith(base)) return route.continue();
      const url = new URL(route.request().url());
      if (url.pathname.startsWith('/manual-orders')) requests.push([route.request().method(), url.pathname, url.search]);
      if (url.pathname.endsWith('/mother-sheet.pdf')) {
        if (oversized) return route.fulfill({status:400, json:{detail:'This order is too long for two complete copies on one page. Choose 1 copy or try the other orientation.'}});
        return route.fulfill({contentType:'application/pdf',
          headers:twoCopySupport ? {'X-Manual-Mother-Sheet-Layout':url.searchParams.get('layout'),
            'Access-Control-Expose-Headers':'X-Manual-Mother-Sheet-Layout'} : {},
          body:Buffer.from(fixture.pdfs[url.searchParams.get('layout')], 'base64')});
      }
      if (url.pathname === '/manual-orders/31') return route.fulfill({json:order});
      if (url.pathname === '/manual-orders') return route.fulfill({json:{items:[order], has_more:false}});
      return route.fulfill({json:{}});
    });
    await page.goto(base, {waitUntil:'networkidle'});
    await page.evaluate(async () => {activateTab('manual'); await openManualOrder(31, true);});
    const before = await page.evaluate(() => JSON.stringify(manualOrdersState.rows));
    // Print must keep original items even when dimension grouping is selected.
    await page.locator('#manualOrderGroupDimensions').click();
    await page.locator('#manualRedIndexPrint').click();
    await page.keyboard.press('Escape');
    assert(await page.locator('#manualMotherSheetLayoutModal').isHidden());
    assert.equal(requests.filter(r => r[1].endsWith('.pdf')).length, 0);
    const layouts = ['a4_portrait', 'a4_landscape_2up', 'a4_portrait_2up'];
    for (const layout of layouts) {
      await page.locator('#manualRedIndexPrint').click();
      const popupEvent = page.waitForEvent('popup');
      await page.locator(`[data-manual-mother-sheet-layout="${layout}"]`).click();
      const popup = await popupEvent;
      await popup.waitForURL('blob:**');
      assert(await page.locator('#manualMotherSheetLayoutModal').isHidden());
      await popup.close();
    }
    assert.deepEqual(requests.filter(r => r[1].endsWith('.pdf')), layouts.map(layout => ['GET', '/manual-orders/31/mother-sheet.pdf', `?layout=${layout}`]));
    assert(requests.every(r => r[0] === 'GET'));
    assert.equal(await page.evaluate(() => JSON.stringify(manualOrdersState.rows)), before);
    assert.equal(await page.locator('#manualOrderError').isVisible(), false);
    oversized = false;
    twoCopySupport = false;
    await page.locator('#manualRedIndexPrint').click();
    await page.locator('[data-manual-mother-sheet-layout="a4_landscape_2up"]').click();
    await page.waitForFunction(() => document.getElementById('manualOrderError').textContent.includes('not available yet'));
    assert.equal(await page.locator('#manualOrderError').isVisible(), true);
    oversized = true;
    await page.locator('#manualRedIndexPrint').click();
    await page.locator('[data-manual-mother-sheet-layout="a4_portrait_2up"]').click();
    await page.waitForFunction(() => document.getElementById('manualOrderError').textContent.includes('too long'));
    assert.equal(await page.evaluate(() => JSON.stringify(manualOrdersState.rows)), before);
    console.log('Mother Sheet: all three layouts, chooser cancellation, oversized-order message, original rows, grouping isolation and GET-only behavior verified.');
  } catch (error) {console.error(error); process.exitCode = 1;}
  finally {if (browser) await browser.close(); server.close();}
});
