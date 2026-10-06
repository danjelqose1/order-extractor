// Local Chromium/WebKit QA with the real PDF renderer and a fixture AI response.
const {chromium,webkit} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {spawn} = require('node:child_process');
const fixture = require('./fixtures/perfect_cut_order.json');
const voiceQA = require('./production_sheet_voice_browser.cjs');
const root = path.resolve(__dirname,'..');
const output = process.env.PRODUCTION_SHEET_QA_OUTPUT || '/tmp/production-sheet-browser-qa';
fs.mkdirSync(output,{recursive:true});
const python = process.env.PRODUCTION_TEST_PYTHON || 'python3';
const pdfjs = process.env.PRODUCTION_PDFJS_DIR || path.dirname(require.resolve('pdfjs-dist/package.json'));
const serverCode = `
import sys,os,tempfile,json
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
memory_dir=tempfile.TemporaryDirectory()
os.environ['DB_DIR']=memory_dir.name
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from production_sheets import SheetRequest,SheetAIRequest,SheetFeedbackRequest,render_sheet,_validate_images,suggest_sheet,TYPOGRAPHY_FIELDS
from production_sheet_memory import remember_sheet,similar_sheets
from db import engine,ProductionSheetExample
ProductionSheetExample.__table__.create(engine)
import uvicorn
app=FastAPI()
@app.post('/qa/reset')
def reset():
    with engine.begin() as conn:conn.execute(ProductionSheetExample.__table__.delete())
    return {'ok':True}
@app.post('/api/production-sheets/preview')
def preview(request:SheetRequest):
    try:return render_sheet(request)
    except ValueError as exc:raise HTTPException(400,str(exc))
@app.post('/api/production-sheets/ai')
def ai(request:SheetAIRequest):
    _validate_images(request.images)
    if request.mode=='automatic':
        memory=similar_sheets(request.source,request.settings)
        typography={key:getattr(request.settings,key) for key in TYPOGRAPHY_FIELDS}
        if request.source.row_count<12:typography.update(font_size=16,line_spacing=1.25,section_gap_pt=10,glass_after_pt=6)
        if memory:typography=memory[0]['typography']
        client=SimpleNamespace()
        client.with_options=lambda **kw:client
        client.responses=SimpleNamespace(create=lambda **kw:SimpleNamespace(status='completed',output_text=json.dumps({'typography':typography,'explanation':'Used space and saved preferences.','warnings':[]})))
        return suggest_sheet(client,request,memory)
    settings=request.settings.model_copy(update={'layout':'full','columns':'3','orientation':'landscape'})
    return {'proposal':{'settings':settings.model_dump(),'explanation':'Three readable columns in each complete copy.','warnings':[]},'preview':render_sheet(SheetRequest(source=request.source,settings=settings))}
@app.post('/api/production-sheets/feedback')
def feedback(request:SheetFeedbackRequest):
    try:return remember_sheet(request)
    except ValueError as exc:raise HTTPException(400,str(exc))
app.mount('/',StaticFiles(directory=sys.argv[2],html=True),name='frontend')
uvicorn.run(app,host='127.0.0.1',port=int(sys.argv[3]),log_level='warning')
`;
async function availablePort(){
  const server=http.createServer();
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const port=server.address().port;
  await new Promise(resolve=>server.close(resolve));
  return port;
}
async function run(engine,name,base){
  await fetch(base+'/qa/reset',{method:'POST'});
  const browser=await engine.launch({headless:true,
    ...(name==='chromium' && process.env.PRODUCTION_CHROMIUM_PATH ? {executablePath:process.env.PRODUCTION_CHROMIUM_PATH} : {})});
  try{
    const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
    await context.addInitScript(base=>{
      localStorage.setItem('loe.apiBase',base);
      window.print=()=>{document.documentElement.dataset.printCalled='yes';};
    },base);
    const page=await context.newPage();
    const errors=[], aiBodies=[], feedbackBodies=[];
    page.on('pageerror',error=>errors.push(error.message));
    let aiFailure=false, aiTimeout=false, aiGate=null, feedbackFailure=false;
    const downloads=[];
    page.on('download',download=>downloads.push(download));
    await page.route('**/*',async route=>{
      const url=new URL(route.request().url());
      if(route.request().headers().accept?.includes('text/event-stream')) return route.fulfill({status:204,body:''});
      if(url.href.includes('pdfjs-dist@')){
        const filename=url.pathname.endsWith('pdf.worker.mjs')?'pdf.worker.mjs':'pdf.mjs';
        return route.fulfill({contentType:'text/javascript',headers:{'Access-Control-Allow-Origin':'*'},
          body:fs.readFileSync(path.join(pdfjs,'build',filename))});
      }
      if(url.origin===base){
        if(url.pathname==='/api/production-sheets/ai'){
          aiBodies.push(route.request().postDataJSON());
          if(aiGate){
            const gate=aiGate;
            await gate.wait;
            if(gate.cancelled) return route.abort('aborted').catch(()=>{});
          }
          if(aiFailure) return route.fulfill({status:502,json:{detail:'AI unavailable. Your current sheet is still ready to print.'}});
          if(aiTimeout) return route.fulfill({status:504,json:{detail:'AI took too long to respond. Try again. Your current sheet is still ready to print.'}});
          return route.continue();
        }
        if(url.pathname==='/api/production-sheets/preview') return route.continue();
        if(url.pathname==='/api/production-sheets/feedback'){
          feedbackBodies.push(route.request().postDataJSON());
          if(feedbackFailure) return route.fulfill({status:503,json:{detail:'Memory unavailable'}});
          return route.continue();
        }
        if(url.pathname.startsWith('/api/') || url.pathname.startsWith('/orders') || url.pathname.startsWith('/manual-orders') || url.pathname.startsWith('/events/') || url.pathname.startsWith('/analysis')){
          return route.fulfill({status:200,json:{}});
        }
        return route.continue();
      }
      return route.fulfill({status:200,json:{}}); // Never contact deployed services.
    });
    await page.goto(base,{waitUntil:'networkidle'});
    await page.locator('[data-tab="processing"]').click();
    assert(await page.locator('#productionSheetOpen').isDisabled());
    assert(await page.locator('#productionSheetAuto').isDisabled());
    await page.evaluate(order=>addOrderToProcessing(order),fixture);
    const original=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
    await page.locator('#productionSheetOpen').click();
    await page.locator('#productionSheetPrint').waitFor({state:'visible'});
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.match(await page.locator('#productionSheetSummary').innerText(),/2 copies · 1 A4 sheet/);
    assert.match(await page.locator('#productionSheetCounts').innerText(),/8 rows · 22 pieces/);
    assert(await page.locator('#productionSheetPages canvas').count()===1);
    assert.equal(aiBodies.length,1,'new sheets run the formatting agent automatically');
    assert.equal(aiBodies[0].mode,'automatic');
    assert.equal(feedbackBodies.length,0,'opening an AI sheet must not teach preferences');
    assert.equal(downloads.length,0,'normal preparation must not download without a save action');
    await page.locator('#productionSheetZoom').click();
    assert.equal(await page.locator('#productionSheetZoom').innerText(),'Fit page');
    await page.locator('#productionSheetZoom').click();
    await page.screenshot({path:path.join(output,`${name}-short.png`),fullPage:true});
    await page.locator('.production-sheet-options summary').click();
    await page.locator('[data-sheet-setting="glass_after_pt"]').fill('8');
    await page.locator('[data-sheet-setting="glass_after_pt"]').press('Tab');
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.equal(await page.locator('[data-sheet-setting="glass_after_pt"]').inputValue(),'8');
    await page.locator('.production-sheet-options summary').click();
    await page.locator('[data-sheet-setting="note"]').fill('Keep this order together.');
    await page.locator('[data-sheet-setting="note"]').press('Tab');
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    await page.locator('#productionSheetReset').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.equal(await page.locator('[data-sheet-setting="note"]').inputValue(),'');
    assert.equal(await page.locator('[data-sheet-setting="glass_after_pt"]').inputValue(),'0');
    const downloadPromise=page.waitForEvent('download');
    await page.locator('#productionSheetSave').click();
    const download=await downloadPromise;
    await download.saveAs(path.join(output,`${name}-short.pdf`));
    const popupPromise=page.waitForEvent('popup');
    await page.locator('#productionSheetPrint').click();
    const popup=await popupPromise;
    await popup.waitForFunction(()=>document.documentElement.dataset.printCalled==='yes');
    assert.equal(await popup.locator('img').count(),1);
    if(name==='chromium') await popup.pdf({path:path.join(output,'chromium-print.pdf'),preferCSSPageSize:true,printBackground:true});
    await popup.close();
    await page.locator('#productionSheetRequest').fill('Please use three columns.');
    await page.locator('#productionSheetAsk').click();
    await page.locator('#productionSheetProposal').waitFor({state:'visible'});
    await page.waitForFunction(()=>!document.getElementById('productionSheetApply').disabled);
    assert(await page.locator('#productionSheetPrint').isDisabled());
    assert.match(await page.locator('#productionSheetSummary').innerText(),/3 columns per copy · 2 copies · 2/);
    assert(aiBodies.at(-1).images[0].startsWith('data:image/jpeg;base64,'));
    assert(aiBodies.at(-1).images[0].length>10000);
    assert.deepEqual(aiBodies.at(-1).rendered.sampled_pages,[1]);
    assert.equal(aiBodies.at(-1).mode,'review');
    await page.screenshot({path:path.join(output,`${name}-ai-proposal.png`),fullPage:true});
    await page.locator('#productionSheetDiscard').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.match(await page.locator('#productionSheetSummary').innerText(),/1 A4 sheet/);
    await page.locator('#productionSheetAsk').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetApply').disabled);
    await page.locator('#productionSheetApply').click();
    assert(!await page.locator('#productionSheetPrint').isDisabled());
    assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),original);
    aiFailure=true;
    await page.locator('#productionSheetAsk').click();
    await page.waitForFunction(()=>document.getElementById('productionSheetStatus').textContent.includes('AI unavailable'));
    assert(!await page.locator('#productionSheetPrint').isDisabled());
    aiFailure=false;
    await page.evaluate(()=>{appState.processing.rows[0].width+=5; recalcProcessingPreview(); updateProcessingUI();});
    assert(await page.locator('#productionSheetStale').isVisible());
    assert(await page.locator('#productionSheetPrint').isDisabled());
    await page.locator('#productionSheetRefresh').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    await page.locator('#productionSheetClose').click();
    const large=structuredClone(fixture);
    large.id=999;large.order_number='R-26-0999';
    large.rows=Array.from({length:160},(_,index)=>({...fixture.rows[0],position:String(index+1),dimension:`${400+index}x1200`,quantity:1}));
    await page.evaluate(order=>{clearProcessing();addOrderToProcessing(order);},large);
    await page.locator('#productionSheetOpen').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.match(await page.locator('#productionSheetSummary').innerText(),/[23] columns per copy/);
    assert.match(await page.locator('#productionSheetCounts').innerText(),/160 rows · 160 pieces/);
    await page.locator('#productionSheetNext').click();
    await page.waitForFunction(()=>document.getElementById('productionSheetPageLabel').textContent==='Page 2 of 4');
    await page.screenshot({path:path.join(output,`${name}-large.png`),fullPage:true});
    await page.locator('#productionSheetAsk').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetApply').disabled);
    assert.deepEqual(aiBodies.at(-1).rendered.sampled_pages,[1,2]);
    await page.locator('#productionSheetDiscard').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    await page.locator('#productionSheetReset').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.equal(await page.locator('#productionSheetLayout').inputValue(),'auto');
    await page.emulateMedia({colorScheme:'dark'});
    await page.screenshot({path:path.join(output,`${name}-dark.png`),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,`${name}-mobile.png`),fullPage:true});
    assert(await page.locator('#productionSheetPrint').isVisible());
    const box=await page.locator('#productionSheetDialog').boundingBox();
    assert(box.x>=0 && box.x+box.width<=391);
    assert(await page.evaluate(()=>document.getElementById('productionSheetDialog').scrollWidth<=document.getElementById('productionSheetDialog').clientWidth+1));
    await page.locator('#productionSheetClose').click();
    await page.setViewportSize({width:1440,height:1000});
    const largeBefore=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
    const largeSource=await page.evaluate(()=>ProductionSheets.capture(appState.processing).source);
    const autoLargePromise=page.waitForEvent('download');
    await page.locator('#productionSheetAuto').click();
    const autoLarge=await autoLargePromise;
    await autoLarge.saveAs(path.join(output,`${name}-automatic-large.pdf`));
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.equal(await page.locator('#productionSheetResultTitle').innerText(),'AI chosen layout');
    assert(await page.locator('#productionSheetReview').isHidden());
    assert.match(await page.locator('#productionSheetStatus').innerText(),/AI prepared your PDF and started the download/);
    assert.deepEqual(aiBodies.at(-1).source,largeSource);
    assert.deepEqual(aiBodies.at(-1).rendered.sampled_pages,[1,2]);
    assert(['2','3'].includes(await page.locator('#productionSheetColumns').inputValue()));
    assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),largeBefore);
    await page.screenshot({path:path.join(output,`${name}-automatic-large.png`),fullPage:true});
    await page.locator('#productionSheetClose').click();
    await page.evaluate(order=>{clearProcessing();addOrderToProcessing(order);},fixture);
    const smallBefore=await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels}));
    const smallSource=await page.evaluate(()=>ProductionSheets.capture(appState.processing).source);
    const autoSmallPromise=page.waitForEvent('download');
    await page.locator('#productionSheetAuto').click();
    const autoSmall=await autoSmallPromise;
    await autoSmall.saveAs(path.join(output,`${name}-automatic-short.pdf`));
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.deepEqual(aiBodies.at(-1).source,smallSource);
    assert.deepEqual(aiBodies.at(-1).rendered.sampled_pages,[1]);
    assert.match(await page.locator('#productionSheetSummary').innerText(),/2 copies · 1 A4 sheet/);
    assert.equal(await page.evaluate(()=>JSON.stringify({processing:appState.processing,labels:appState.labels})),smallBefore);
    await page.screenshot({path:path.join(output,`${name}-automatic-short.png`),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,`${name}-automatic-mobile.png`),fullPage:true});
    assert(await page.locator('#productionSheetPrint').isVisible());
    assert(await page.evaluate(()=>document.getElementById('productionSheetDialog').scrollWidth<=document.getElementById('productionSheetDialog').clientWidth+1));
    await page.locator('#productionSheetClose').click();
    await page.setViewportSize({width:1440,height:1000});
    const downloaded=downloads.length;
    aiFailure=true;
    await page.locator('#productionSheetAuto').click();
    await page.waitForFunction(()=>document.getElementById('productionSheetStatus').textContent.includes('AI unavailable'));
    assert.equal(downloads.length,downloaded);
    assert(!await page.locator('#productionSheetPrint').isDisabled());
    assert(await page.locator('#productionSheetProposal').isHidden());
    aiFailure=false;
    await page.locator('#productionSheetClose').click();
    aiTimeout=true;
    await page.locator('#productionSheetAuto').click();
    await page.waitForFunction(()=>document.getElementById('productionSheetStatus').textContent.includes('AI took too long'));
    assert.equal(downloads.length,downloaded);
    assert(!await page.locator('#productionSheetPrint').isDisabled());
    aiTimeout=false;
    await page.locator('#productionSheetClose').click();
    await page.evaluate(()=>{
      const interval=window.setInterval;
      window.setInterval=(callback,delay,...args)=>interval(callback,delay===15000?50:delay,...args);
    });
    for(const cancel of ['close','source']){
      let release;
      aiGate={wait:new Promise(resolve=>{release=resolve;}),cancelled:false};
      const requested=page.waitForRequest(request=>request.url().endsWith('/api/production-sheets/ai'));
      await page.locator('#productionSheetAuto').click();
      await requested;
      await page.waitForFunction(()=>document.getElementById('productionSheetStatus').textContent.includes('AI is still reviewing'));
      assert(await page.locator('#productionSheetAuto').isDisabled());
      assert(await page.locator('#productionSheetOpen').isDisabled());
      if(cancel==='close'){
        await page.locator('#productionSheetClose').click();
        await page.locator('#productionSheetOpen').click();
        await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
      }else{
        await page.evaluate(()=>{appState.processing.rows[0].width+=5;recalcProcessingPreview();updateProcessingUI();});
        assert(await page.locator('#productionSheetStale').isVisible());
        assert(await page.locator('#productionSheetPrint').isDisabled());
      }
      aiGate.cancelled=true;release();aiGate=null;
      await page.waitForTimeout(100);
      assert.equal(downloads.length,downloaded);
      assert(await page.locator('#productionSheetProposal').isHidden());
      await page.locator('#productionSheetClose').click();
    }
    await page.evaluate(order=>{clearProcessing();addOrderToProcessing(order);},fixture);
    await page.locator('#productionSheetOpen').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    // Only the finished manual corrections are learned, including after reopening.
    await page.locator('.production-sheet-options summary').click();
    await page.locator('[data-sheet-setting="glass_after_pt"]').fill('9');
    await page.locator('[data-sheet-setting="glass_after_pt"]').press('Tab');
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    const beforeSave=feedbackBodies.length;
    const aiBeforeReopen=aiBodies.length;
    await page.locator('#productionSheetClose').click();
    await page.locator('#productionSheetOpen').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.equal(aiBodies.length,aiBeforeReopen,'reopening a draft preserves edits without rerunning AI');
    assert.equal(feedbackBodies.length,beforeSave,'editing and closing must not teach');
    feedbackFailure=true;
    const failedMemoryDownload=page.waitForEvent('download');
    await page.locator('#productionSheetSave').click();await failedMemoryDownload;
    await page.waitForFunction(()=>document.getElementById('productionSheetMemory').textContent.includes('could not be remembered'));
    assert(!await page.locator('#productionSheetPrint').isDisabled(),'memory failures must not block printing');
    feedbackFailure=false;
    const learnedDownload=page.waitForEvent('download');
    await page.locator('#productionSheetSave').click();await learnedDownload;
    await page.waitForFunction(()=>document.getElementById('productionSheetMemory').textContent.includes('remembered for similar'));
    assert.equal(feedbackBodies.at(-1).settings.glass_after_pt,9);
    assert.notEqual(feedbackBodies.at(-1).baseline_settings.glass_after_pt,9);
    // A full browser reload clears in-memory drafts; database examples remain.
    await page.reload({waitUntil:'networkidle'});
    await page.locator('[data-tab="processing"]').click();
    await page.evaluate(order=>{clearProcessing();addOrderToProcessing(order);},fixture);
    const beforeNewSheet=feedbackBodies.length;
    await page.locator('#productionSheetOpen').click();
    await page.waitForFunction(()=>!document.getElementById('productionSheetPrint').disabled);
    assert.equal(await page.locator('[data-sheet-setting="glass_after_pt"]').inputValue(),'9');
    assert.equal(feedbackBodies.length,beforeNewSheet,'AI must not teach itself from its own output');
    assert(await page.locator('#productionSheetReview').isHidden());
    await voiceQA(page,name,output,aiBodies);
    assert.equal(errors.length,0,errors.join('\n'));
    console.log(`${name}: preview, PDF, Print, visual AI proposal, Apply/Discard, one-click AI downloads, cancellation, failure recovery, stale source, large jobs, reset and responsive themes passed`);
  }finally{await browser.close();}
}
(async()=>{
  const port=await availablePort(),base=`http://127.0.0.1:${port}`;
  const server=spawn(python,['-c',serverCode,path.join(root,'backend'),path.join(root,'docs'),String(port)],{stdio:['ignore','ignore','pipe']});
  let stderr='';server.stderr.on('data',data=>{stderr+=data;});
  try{
    for(let attempt=0;attempt<100;attempt++){
      try{if((await fetch(base)).ok)break;}catch{}
      if(server.exitCode!=null)throw new Error(stderr);
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    if(!process.env.PRODUCTION_SHEET_QA_ENGINE || process.env.PRODUCTION_SHEET_QA_ENGINE==='chromium') await run(chromium,'chromium',base);
    if(!process.env.PRODUCTION_SHEET_QA_ENGINE || process.env.PRODUCTION_SHEET_QA_ENGINE==='webkit') await run(webkit,'webkit',base);
  }finally{server.kill();}
})().catch(error=>{console.error(error);process.exitCode=1;});
