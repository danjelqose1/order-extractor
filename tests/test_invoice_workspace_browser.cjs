// Exercise the real frontend with an isolated in-memory invoice API.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const root = path.resolve(__dirname, '../docs');
const output = process.env.INVOICE_QA_OUTPUT || '/tmp/invoice-workspace-qa';
fs.mkdirSync(output, { recursive: true });
const manualOrder = { id: 7, order_number: 'M-007', client_name: 'North Glass', status: 'approved', order_date: '2026-10-16', rows: [{ glass_type: '4F', quantity: 2, width_mm: 1000, height_mm: 500, final_area_m2: 1 }], total_quantity: 2, total_area_m2: 1 };
const pdfOrder = { id: 12, order_number: 'P-012', order_numbers: ['P-012'], client_name: 'City Windows', status: 'approved', created_at: '2026-10-16T12:00:00Z', rows: [{ type: '4F', quantity: 2, area: .5, dimension: '1000x500', order_number: 'P-012' }] };
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
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', args: ['--no-sandbox'] });
  let savedJobs = [], failSave = false, failLoad = false, writes = 0;
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1024 }, colorScheme: 'light' });
    await context.addInitScript(base => localStorage.setItem('loe.apiBase', base), base);
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    await page.route('**/*', route => {
      const req = route.request(), url = new URL(req.url());
      if (url.hostname === 'cdn.jsdelivr.net' && url.pathname.includes('pdf-lib')) return route.fulfill({ contentType: 'text/javascript', body: fs.readFileSync(require.resolve('pdf-lib/dist/pdf-lib.min.js')) });
      if (url.origin !== base) return route.fulfill({ body: '' });
      if (url.pathname === '/api/invoices') {
        if (req.method() === 'POST') {
          writes++;
          if (failSave) return route.fulfill({ status: 503, json: { error: 'Fixture offline' } });
          savedJobs = req.postDataJSON().jobs;
          return route.fulfill({ json: { ok: true } });
        }
        return route.fulfill({ status: failLoad ? 503 : 200, json: { jobs: savedJobs } });
      }
      if (url.pathname === '/api/prices') return route.fulfill({ json: { glass: { '4F': 100 }, spacer: {}, componentAliases: {}, typeCorrections: {} } });
      if (url.pathname === '/orders') return route.fulfill({ json: { items: [pdfOrder], has_more: false } });
      if (url.pathname === '/orders/12') return route.fulfill({ json: pdfOrder });
      if (url.pathname === '/manual-orders') return route.fulfill({ json: { items: [manualOrder], has_more: false } });
      if (url.pathname === '/manual-orders/7') return route.fulfill({ json: manualOrder });
      if (url.pathname === '/api/features') return route.fulfill({ json: { living_dashboard: false } });
      if (req.headers().accept?.includes('text/event-stream')) return route.fulfill({ status: 204, body: '' });
      if (/^\/(api|orders|manual-orders|events|analysis|telegram-files|healthz)(\/|$)/.test(url.pathname)) return route.fulfill({ json: { items: [], jobs: [], counts: {}, groups: {}, files: [], rows: [] } });
      return route.continue();
    });
    const enter = async () => {
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.locator('[data-tab="invoices"]').click();
      await page.waitForFunction(() => window.InvoiceWorkspace && invoiceRuntime.jobsLoaded);
    };
    await enter();
    assert.equal(writes, 0, 'Opening the invoice section must not write');
    assert(await page.getByText('Your next invoice starts here').isVisible());
    await page.screenshot({ path: path.join(output, 'empty-desktop.png'), fullPage: true });
    await page.locator('#invoiceNew').click();
    await page.getByRole('button', { name: /Start a blank invoice/ }).click();
    await page.locator('#invoiceBusinessName').fill('Alba Glass Studio');
    await page.locator('#invoiceBusinessDetails').fill('Rruga e Dibrës 25\nTiranë, Albania\naccounts@example.com');
    await page.locator('#invoiceBusinessTaxId').fill('AL-123456789');
    await page.locator('#invoiceClientName').fill('Harbor Architecture');
    await page.locator('#invoiceClientDetails').fill('24 Riverside Avenue\nProject office');
    await page.locator('#invoiceIssueDate').fill('2026-10-17');
    await page.locator('#invoiceDueDate').fill('2026-11-16');
    await page.getByLabel('Item 1 description', { exact: true }).fill('Custom glass panels');
    await page.getByLabel('Item 1 quantity', { exact: true }).fill('3');
    await page.getByLabel('Item 1 unit price', { exact: true }).fill('1250.50');
    await page.locator('#invoiceNotes').fill('Bank transfer within 30 days.\nThank you for choosing Alba Glass Studio.');
    await page.locator('#invoiceDiscountValue').fill('10');
    await page.locator('#invoiceVatValue').fill('20');
    await page.locator('#invoiceSaveDocument').click();
    await page.waitForFunction(() => document.getElementById('invoiceStatus').textContent === 'Invoice saved.');
    assert.equal(savedJobs.length, 1);
    assert.equal(savedJobs[0].source, 'invoice');
    assert.equal(savedJobs[0].document.clientName, 'Harbor Architecture');
    let calc = await page.evaluate(() => getActiveInvoiceJob().calculated);
    assert.equal(calc.subtotal, 3751.5);
    assert.equal(calc.discountValue, 375.15);
    assert.equal(calc.vatValue, 675.27);
    assert.equal(calc.total, 4051.62);
    await page.screenshot({ path: path.join(output, 'editor-desktop.png'), fullPage: true });
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#invoiceGeneratePdf').click();
    const download = await downloadPromise;
    await download.saveAs(path.join(output, 'invoice.pdf'));
    assert(fs.statSync(path.join(output, 'invoice.pdf')).size > 1000);
    const originalNumber = savedJobs[0].invoiceNumber;
    await page.locator('#invoiceStatusSelect').selectOption('finalized');
    await page.waitForFunction(() => getActiveInvoiceJob().status === 'finalized');
    assert.equal(savedJobs[0].finalizedSnapshot.total, 4051.62);
    assert(await page.locator('#invoiceBusinessName').isDisabled());
    await page.locator('#manualInvoiceClose').click();
    await page.screenshot({ path: path.join(output, 'register-desktop.png'), fullPage: true });
    await enter();
    assert(await page.getByText(originalNumber, { exact: true }).isVisible());
    assert.equal(await page.evaluate(() => appState.invoices.jobs[0].calculated.total), 4051.62);
    // Copies receive a fresh number and can be edited without changing the original.
    await page.getByRole('button', { name: `Duplicate ${originalNumber}`, exact: true }).click();
    await page.waitForFunction(() => getActiveInvoiceJob()?.invoiceNumber && getActiveInvoiceJob()?.status === 'draft');
    assert.equal(savedJobs.length, 2);
    assert.notEqual(savedJobs[0].invoiceNumber, originalNumber);
    assert.equal(savedJobs[1].status, 'finalized');
    await page.locator('#invoiceDueDate').fill('2020-01-01');
    await page.locator('#invoiceSaveDocument').click();
    assert.match(await page.locator('#invoiceStatus').textContent(), /due date cannot be before/);
    await page.locator('#invoiceDueDate').fill('2026-12-01');
    await page.locator('#invoiceDocumentNumber').fill(originalNumber);
    await page.locator('#invoiceSaveDocument').click();
    assert.match(await page.locator('#invoiceStatus').textContent(), /already in use/);
    await page.locator('#invoiceDocumentNumber').fill('INV-RETRY');
    failSave = true;
    await page.locator('#invoiceSaveDocument').click();
    await page.waitForFunction(() => document.getElementById('invoiceStatus').textContent.includes('Could not save'));
    assert.match(await page.locator('#invoiceStatus').textContent(), /Could not save/);
    assert(savedJobs.every(job => job.invoiceNumber !== 'INV-RETRY'));
    failSave = false;
    await page.locator('#invoiceSaveDocument').click();
    await page.waitForFunction(() => document.getElementById('invoiceStatus').textContent === 'Invoice saved.');
    assert(savedJobs.some(job => job.invoiceNumber === 'INV-RETRY'));
    await page.locator('#manualInvoiceClose').click();
    await page.locator('#invoiceFilterOrder').fill('INV-RETRY');
    assert.equal(await page.locator('#invoiceJobsWrap tbody tr').count(), 1);
    await page.locator('#invoiceFilterClient').fill('Nobody');
    assert(await page.getByText('No matching invoices', { exact: true }).isVisible());
    await page.locator('#invoiceClearFilters').click();
    // Both existing order sources produce drafts and preserve their original data.
    for (const client of ['North Glass', 'City Windows']) {
      await page.locator('#invoiceNew').click();
      await page.getByRole('button', { name: new RegExp(client) }).click();
      await page.waitForFunction(name => getActiveInvoiceJob()?.client === name, client);
      await page.waitForFunction(() => document.getElementById('invoiceStatus').textContent.includes('Draft created from order'));
      assert(await page.locator('#invoiceGeneratePdf').isEnabled());
      assert.equal(await page.evaluate(() => getActiveInvoiceJob().calculated.total), 100);
      await page.locator('#manualInvoiceClose').click();
    }
    const count = savedJobs.length;
    await page.locator('#invoiceNew').click();
    await page.getByRole('button', { name: /North Glass/ }).click();
    assert.equal(savedJobs.length, count, 'Reusing an order opens the existing invoice');
    await page.locator('#manualInvoiceClose').click();
    // Long content and many rows paginate without losing items.
    const previousActiveId = await page.evaluate(() => getActiveInvoiceJob().id);
    await page.getByRole('button', { name: 'Duplicate INV-RETRY', exact: true }).click();
    await page.waitForFunction(previous => getActiveInvoiceJob()?.id !== previous && getActiveInvoiceJob()?.invoiceNumber, previousActiveId);
    await page.evaluate(async () => {
      const job = getActiveInvoiceJob();
      job.items = Array.from({ length: 65 }, (_, i) => ({ description: `Item ${i + 1}: Custom laminated glass panels with a detailed description for the architectural project, protective treatment and finishing.`, quantity: 2, rate: 100 }));
      InvoiceWorkspace.calculate(job); updateInvoicesUI();
      await InvoiceWorkspace.prepareExport(job);
    });
    assert.equal(await page.evaluate(() => getActiveInvoiceJob().calculated.lines.length), 65);
    const longPdf = await page.evaluate(async () => Array.from(await InvoicePDF.build(getActiveInvoiceJob())));
    fs.writeFileSync(path.join(output, 'invoice-multipage.pdf'), Buffer.from(longPdf));
    await page.locator('#manualInvoiceClose').click();
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.screenshot({ path: path.join(output, 'register-dark.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('#tabInvoices').isVisible());
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Mobile page has no horizontal overflow');
    await page.screenshot({ path: path.join(output, 'register-mobile.png'), fullPage: true });
    await page.locator('#invoiceNew').click();
    await page.screenshot({ path: path.join(output, 'create-mobile.png'), fullPage: true });
    await page.locator('#invoiceCreateClose').click();
    failLoad = true;
    const writesBeforeFailure = writes;
    await page.locator('#invoiceRefresh').click();
    await page.waitForFunction(() => document.getElementById('invoiceWorkspaceStatus').textContent.includes('Could not load invoices'));
    assert.match(await page.locator('#invoiceWorkspaceStatus').textContent(), /Could not load invoices/);
    assert.equal(writes, writesBeforeFailure, 'A failed load cannot replace existing invoices');
    assert(await page.locator('#invoiceNew').isDisabled());
    failLoad = false;
    await page.locator('#invoiceRefresh').click();
    await page.waitForFunction(() => !document.getElementById('invoiceNew').disabled);
    assert(await page.locator('#invoiceNew').isEnabled());
    const safeguards = await page.evaluate(async () => {
      const legacy = { id: 'legacy-fixture', status: 'finalized', totalAmount: 1234.56, totalUnits: 2, totalArea: 1, rows: [] };
      await recalcInvoiceJob(legacy, { allowAi: false });
      const small = { source: 'invoice', items: [{ description: 'Small item', quantity: 1, rate: .03 }], discountMode: 'percent', discountValue: 15, vatRate: 20 };
      InvoiceWorkspace.calculate(small);
      const frozen = appState.invoices.jobs.find(job => job.status === 'finalized');
      const before = frozen.calculated.total;
      appState.invoices.priceLists.glass = { '4f': 99999 };
      await recalcAllInvoices({ allowAi: false });
      return { legacyTotal: legacy.calculated.total, legacyBlocked: legacy.calculated.missing.reason, smallTotal: small.calculated.total, frozenUnchanged: frozen.calculated.total === before };
    });
    assert.deepEqual(safeguards, { legacyTotal: 1234.56, legacyBlocked: 'legacy_snapshot', smallTotal: .04, frozenUnchanged: true });
    assert.deepEqual(errors, []);
    console.log('PASS: blank/order invoices, totals, save/reload, PDF, snapshots, duplication, validation, filters, failures, mobile and dark mode');
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
