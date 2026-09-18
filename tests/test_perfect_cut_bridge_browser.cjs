// Run with NODE_PATH pointing to an existing Playwright installation. No app dependencies added.
const { chromium, webkit } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const fixture = require('./fixtures/perfect_cut_order.json');
const docs = path.resolve(__dirname,'../docs');
const expected = 'quantity,width,height\r\n2,315,790\r\n2,365,1260\r\n2,375,390\r\n2,535,1030\r\n4,535,1240\r\n2,535,1250\r\n2,535,1280\r\n6,575,530\r\n';
async function run(engine,name,base){
  const browser = await engine.launch({headless:true});
  try{
    const page = await browser.newPage({viewport:{width:1366,height:900},acceptDownloads:true});
    const manualFixture=require('./fixtures/perfect_cut_manual_order.json');
    const manualOrders={31:structuredClone(manualFixture),32:{...structuredClone(manualFixture),id:32,order_number:'M-032',status:'processing'}};
    const manualRequests=[];
    const errors=[]; page.on('pageerror',error=>errors.push(error.message));
    // Isolated UI fixture: no calls to deployed services or production data.
    await page.route('**/*',route=>{
      const url=route.request().url();
      if(url.startsWith(base)) return route.continue();
      const parsed=new URL(url);
      if(parsed.pathname.startsWith('/manual-orders')){
        manualRequests.push(route.request().method());
        if(parsed.pathname==='/manual-orders'){
          const query=parsed.searchParams.get('query');
          if(query==='failed') return route.fulfill({status:500,json:{detail:'Fixture request failed'}});
          const items=query==='missing'?[]:parsed.searchParams.get('offset')==='50'?[manualOrders[32]]:[manualOrders[31],{...manualFixture,id:33,order_number:'M-DRAFT',status:'draft'}];
          return route.fulfill({status:200,json:{items:items.map(o=>({...o,rows:undefined,row_count:o.rows.length,total_quantity:6})),has_more:!query && parsed.searchParams.get('offset')==='0'}});
        }
        const order=manualOrders[Number(parsed.pathname.split('/').at(-1))];
        return route.fulfill({status:order?200:404,json:order||{detail:'Not found'}});
      }
      if(url.includes('/events/')) return route.fulfill({status:204,body:''});
      return route.fulfill({status:200,contentType:'application/json',body:'{}'});
    });
    await page.goto(base,{waitUntil:'networkidle'});
    await page.locator('[data-tab="perfectcut"]').click();
    await page.locator('#bridgeAdd').click();
    assert.match(await page.locator('#bridgePickerBody').innerText(),/Processing is empty/);
    assert(await page.locator('#bridgePickerAdd').isDisabled());
    await page.locator('#bridgePickerCancel').click();
    await page.evaluate(order=>addOrderToProcessing(order),fixture);
    const sourceBefore=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
    await page.locator('#bridgeAdd').evaluate(button=>{button.click();button.click();});
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),8);
    assert.match(await page.locator('#bridgeSummary').innerText(),/10.0013/);
    assert.match(await page.locator('#bridgeSourceSummary').innerText(),/10.080/);
    assert.equal(await page.locator('#bridgePass').count(),0);
    assert(!await page.locator('#bridgeDownload').isDisabled());
    const downloadPromise=page.waitForEvent('download');
    await page.locator('#bridgeDownload').click();
    const download=await downloadPromise;
    assert.equal(download.suggestedFilename(),'job.csv');
    assert.equal(fs.readFileSync(await download.path(),'utf8'),expected);
    assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),sourceBefore);
    await page.screenshot({path:`/tmp/perfect-cut-${name}-desktop.png`,fullPage:true});
    await page.locator('#bridgeAdd').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),8);
    // Source edit leaves snapshot unchanged, blocks export and supports explicit replacement.
    await page.evaluate(()=>{appState.processing.cart[0].rows[0].width=320; appState.processing.cart[0].rows[0].widthDisplay='320'; rebuildProcessingRows(); PerfectCutBridgeUI.render();});
    assert.match(await page.locator('#bridgeErrors').innerText(),/changed/);
    assert(await page.locator('#bridgeDownload').isDisabled());
    assert.equal(await page.locator('#bridgeRows tbody tr').first().locator('td').nth(4).innerText(),'315');
    await page.locator('#bridgeRefresh').click();
    assert.equal(await page.locator('[data-bridge-select]:checked').count(),8);
    await page.locator('#bridgePickerAdd').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').first().locator('td').nth(4).innerText(),'320');
    await page.locator('[data-bridge-remove="0"]').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),7);
    // Changes during an open picker are caught at the commit boundary.
    await page.locator('#bridgeRefresh').click();
    await page.locator('[data-bridge-select="0"]').check();
    await page.evaluate(()=>{appState.processing.cart[0].rows[0].width=325; rebuildProcessingRows();});
    await page.locator('#bridgePickerAdd').click();
    assert.match(await page.locator('#bridgePickerErrors').innerText(),/changed while this review was open/);
    await page.locator('#bridgePickerCancel').click();
    // Busy async preparation cannot open a picker or export.
    await page.evaluate(()=>processingBridgeBusy++);
    await page.locator('#bridgeAdd').click();
    assert(!await page.locator('#bridgePicker').isVisible());
    assert.match(await page.locator('#bridgeStatus').innerText(),/busy/);
    await page.evaluate(()=>processingBridgeBusy--);
    const beforeClear=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
    page.once('dialog',dialog=>dialog.dismiss()); await page.locator('#bridgeClear').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),7);
    page.once('dialog',dialog=>dialog.accept()); await page.locator('#bridgeClear').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),0);
    assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),beforeClear);
    // Multiple orders and sections, plus explicit deselection of invalid rows.
    const second=structuredClone(fixture); second.id=827; second.order_number='R-26-0827'; second.client_name='Second client'; second.rows[0].shape='triangle'; second.rows[1].type='6F';
    await page.evaluate(order=>addOrderToProcessing(order),second);
    await page.locator('#bridgeAdd').click();
    assert.equal(await page.locator('#bridgePickerSection').count(),0);
    assert.equal(await page.locator('[data-bridge-select]').count(),16);
    await page.locator('[data-bridge-order="1"]').check();
    assert(await page.locator('#bridgePickerAdd').isDisabled());
    assert.match(await page.locator('#bridgePickerErrors').innerText(),/triangle/);
    await page.locator('[data-bridge-select="8"]').uncheck();
    await page.locator('[data-bridge-order="0"]').check();
    await page.locator('#bridgePickerAdd').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),15);
    await page.locator('#bridgeAdd').click();
    assert.equal(await page.locator('[data-bridge-select]:checked').count(),16);
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:`/tmp/perfect-cut-${name}-mobile-picker.png`,fullPage:true});
    const dialogBox=await page.locator('#bridgePicker').boundingBox();
    assert(dialogBox.x>=0 && dialogBox.x+dialogBox.width<=391);
    await page.locator('#bridgePickerCancel').click();
    await page.screenshot({path:`/tmp/perfect-cut-${name}-mobile.png`,fullPage:true});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
    await page.setViewportSize({width:1366,height:900});
    const screenshotOrder=require('./fixtures/perfect_cut_processing_order.json');
    await page.evaluate(order=>{
      clearProcessing(); addOrderToProcessing(order);
      appState.processing.rounding.manualApplied=true; rebuildProcessingRows();
    },screenshotOrder);
    await page.locator('#bridgeAdd').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),5);
    await page.evaluate(()=>{appState.processing.grouped=true; rebuildProcessingRows();});
    await page.locator('#bridgeAdd').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),4);
    const expectedScreenshot='quantity,width,height\r\n1,738,1835\r\n2,815,1903\r\n1,1268,168\r\n1,433,848\r\n';
    const groupedDownloadPromise=page.waitForEvent('download');
    await page.locator('#bridgeDownload').click();
    const groupedDownload=await groupedDownloadPromise;
    assert.equal(fs.readFileSync(await groupedDownload.path(),'utf8'),expectedScreenshot);
    await page.screenshot({path:`/tmp/perfect-cut-${name}-screenshot-order.png`,fullPage:true});
    // Manual Orders is an independent GET-only source; saved millimetres are never rounded.
    const modulesBefore=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
    await page.locator('#bridgeManualAdd').click();
    await page.locator('[data-bridge-manual-order="0"]').waitFor();
    assert(await page.locator('[data-bridge-manual-order="1"]').isDisabled()); // Draft
    await page.locator('[data-bridge-manual-order="0"]').check();
    await page.locator('#bridgeManualNext').click();
    await page.locator('[data-bridge-manual-order="0"]').waitFor();
    await page.locator('[data-bridge-manual-order="0"]').check();
    assert.match(await page.locator('#bridgeManualStatus').innerText(),/2 selected orders/);
    await page.locator('#bridgeManualReview').click();
    await page.locator('#bridgePicker').waitFor();
    assert.equal(await page.locator('[data-bridge-select]:checked').count(),6);
    await page.locator('#bridgePickerAdd').click();
    await page.locator('#bridgePicker').waitFor({state:'hidden'});
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),6);
    assert.equal(await page.locator('#bridgeRows tbody tr').first().locator('td').nth(4).innerText(),'791');
    assert.match(await page.locator('#bridgeRefresh').innerText(),/Manual Orders/);
    await page.locator('#bridgeRefresh').click();
    await page.locator('[data-bridge-order="1"]').uncheck();
    await page.locator('[data-bridge-select="1"]').uncheck();
    await page.locator('#bridgePickerAdd').click();
    await page.locator('#bridgePicker').waitFor({state:'hidden'});
    const manualDownloadPromise=page.waitForEvent('download');
    await page.locator('#bridgeDownload').click();
    const manualDownload=await manualDownloadPromise;
    assert.equal(fs.readFileSync(await manualDownload.path(),'utf8'),'quantity,width,height\r\n2,791,314\r\n1,791,314\r\n');
    assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),modulesBefore);
    await page.screenshot({path:`/tmp/perfect-cut-${name}-manual.png`,fullPage:true});
    // A source edit blocks download without silently editing the captured dimensions.
    manualOrders[31].rows[0].width_mm=800;
    await page.locator('#bridgeDownload').click();
    await page.waitForFunction(()=>document.getElementById('bridgeErrors').textContent.includes('changed'));
    assert(await page.locator('#bridgeDownload').isDisabled());
    assert.equal(await page.locator('#bridgeRows tbody tr').first().locator('td').nth(4).innerText(),'791');
    await page.locator('#bridgeRefresh').click();
    await page.locator('#bridgePickerAdd').click();
    await page.locator('#bridgePicker').waitFor({state:'hidden'});
    assert.equal(await page.locator('#bridgeRows tbody tr').first().locator('td').nth(4).innerText(),'800');
    // Reopening as a draft revokes eligibility even when dimensions have not changed.
    manualOrders[31].status='draft';
    await page.locator('#bridgeDownload').click();
    await page.waitForFunction(()=>document.getElementById('bridgeErrors').textContent.includes('only approved'));
    assert(await page.locator('#bridgeDownload').isDisabled());
    manualOrders[31].status='approved';
    // Request failures and empty searches do not destroy the existing snapshot.
    await page.locator('#bridgeManualAdd').click();
    await page.locator('#bridgeManualSearch').fill('missing');
    await page.locator('#bridgeManualSearchForm button').click();
    await page.waitForFunction(()=>document.getElementById('bridgeManualList').textContent.includes('No manual orders'));
    await page.locator('#bridgeManualSearch').fill('failed');
    await page.locator('#bridgeManualSearchForm button').click();
    await page.waitForFunction(()=>document.getElementById('bridgeManualStatus').textContent.includes('Fixture request failed'));
    await page.locator('#bridgeManualSearch').fill('');
    await page.locator('#bridgeManualSearchForm button').click();
    await page.locator('[data-bridge-manual-order="0"]').waitFor();
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:`/tmp/perfect-cut-${name}-manual-picker-mobile.png`,fullPage:true});
    const manualBox=await page.locator('#bridgeManualPicker').boundingBox();
    assert(manualBox.x>=0 && manualBox.x+manualBox.width<=391);
    await page.locator('#bridgeManualCancel').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),3);
    assert(manualRequests.length>0 && manualRequests.every(request=>request==='GET'));
    // Importing Processing again returns to Processing version checks and exact grouping.
    await page.locator('#bridgeAdd').click();
    assert.equal(await page.locator('#bridgeRows tbody tr').count(),4);

    assert.deepEqual(errors,[]);
    console.log(`${name}: empty, selection, rapid clicks, exact downloaded CSV, duplicates, source changes, refresh, busy guard, removal/clear isolation, geometry, whole-sheet grouped/ungrouped imports, Manual Orders search/paging/review/GET-only isolation/stale checks and mobile checks passed`);
  }finally{await browser.close();}
}
const server=http.createServer((req,res)=>{
  const relative=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const file=path.join(docs,relative==='/'?'index.html':relative);
  if(!file.startsWith(docs+path.sep)) {res.writeHead(403);res.end();return;}
  fs.readFile(file,(error,data)=>{
    if(error){res.writeHead(404);res.end();return;}
    const types={'.html':'text/html','.js':'application/javascript','.css':'text/css','.png':'image/png'};
    res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream'}); res.end(data);
  });
});
server.listen(0,'127.0.0.1',async()=>{
  try{
    const base=`http://127.0.0.1:${server.address().port}/`;
    await run(chromium,'chromium',base); await run(webkit,'webkit',base);
  }catch(error){console.error(error);process.exitCode=1;}finally{server.close();}
});
