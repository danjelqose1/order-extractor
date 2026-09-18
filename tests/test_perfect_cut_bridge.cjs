const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname,'..');
const bridge = require('../docs/js/perfect-cut-bridge.js');
const fixture = require('./fixtures/perfect_cut_order.json');
const clone = value => JSON.parse(JSON.stringify(value));
function context(){
  const ctx = { DIMENSION_TOLERANCE_MM:1, UNKNOWN_CLIENT_LABEL:'(Unknown Client)' };
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(root,'docs/js/platform-workflows.js'),'utf8'),ctx);
  return ctx;
}
function prepare(orders = [clone(fixture)], options = {}){
  const ctx = context();
  const rows = orders.flatMap(order => ctx.convertOrderToProcessingEntry(order).rows);
  ctx.appState = {processing:{rows,options:{autoDanko:!!options.rounded},rounding:{manualApplied:false}}};
  ctx.applyProcessingRoundingToRows();
  const processing = {preview:ctx.generateMotherSheet(rows,{
    normalizeLPtoG:true, decimalSeparator:'comma', groupDimensions:false, mergeAcrossOrders:false,...options,
  })};
  return {ctx,processing,sections:bridge.collect(processing)};
}
const expected = 'quantity,width,height\r\n2,315,790\r\n2,365,1260\r\n2,375,390\r\n2,535,1030\r\n4,535,1240\r\n2,535,1250\r\n2,535,1280\r\n6,575,530\r\n';
test('R-26-0826 exact contract, 22 pieces, cutting area and independent declared area',()=>{
  const {sections,processing}=prepare();
  const before=JSON.stringify(processing);
  const job=bridge.add(bridge.emptyJob(),sections[0].rows);
  assert.equal(job.rows.length,8);
  assert.equal(job.rows.reduce((n,r)=>n+r.quantity,0),22);
  assert.equal(job.rows.reduce((n,r)=>n+r.quantity*r.width*r.height,0)/1e6,10.0013);
  assert.equal(job.rows[0].sources[0].bridgeSource.declaredArea,'10.080');
  assert.equal(bridge.csv(job.rows),expected);
  const bytes=Buffer.from(bridge.csv(job.rows),'utf8');
  assert(bytes.every(byte=>byte<128)); assert.notEqual(bytes[0],0xef);
  assert.equal(bytes.toString().replaceAll('\r\n','').includes('\n'),false);
  assert(bytes.toString().trim().split('\r\n').every(line=>line.split(',').length===3));
  assert.equal(JSON.stringify(processing),before);
});
test('empty Processing and all busy states block capture',()=>{
  assert.deepEqual(bridge.collect({preview:{groups:[]}}),[]);
  assert.throws(()=>bridge.add(bridge.emptyJob(),[]),/Select/);
  for(const p of [{loading:true},{recalculating:true}]) assert.throws(()=>bridge.collect(p),/busy/);
  assert.throws(()=>bridge.collect(prepare().processing,true),/busy/);
});
test('subsets preserve canonical order and snapshot owns every nested value',()=>{
  const {sections}=prepare(); const rows=sections[0].rows;
  const job=bridge.add(bridge.emptyJob(),[rows[1],rows[7]]);
  rows[1].width=999; rows[1].sources[0].bridgeSource.quantity=999;
  assert.equal(job.rows[0].width,365); assert.equal(job.rows[0].sources[0].bridgeSource.quantity,2);
  assert.equal(job.rows[1].width,575);
});
test('duplicate and partially overlapping regrouped sources are blocked atomically',()=>{
  const {sections}=prepare(); const rows=sections[0].rows;
  const job=bridge.add(bridge.emptyJob(),rows.slice(0,2)); const before=JSON.stringify(job);
  assert.throws(()=>bridge.add(job,[rows[0]]),/more than once/);
  assert.throws(()=>bridge.add(job,[rows[2],rows[0]]),/more than once/);
  assert.equal(JSON.stringify(job),before);
});
test('identical dimensions on distinct source rows remain distinct unless Processing grouped them',()=>{
  const order=clone(fixture); order.rows=[order.rows[0],{...order.rows[0],id:99,position:'99'}];
  const ungrouped=prepare([order]); assert.equal(bridge.add(bridge.emptyJob(),ungrouped.sections[0].rows).rows.length,2);
  const grouped=prepare([order],{groupDimensions:true});
  assert.equal(grouped.sections[0].rows.length,1); assert.equal(grouped.sections[0].rows[0].quantity,4);
  assert.equal(grouped.sections[0].rows[0].sources.length,2);
  const job=bridge.add(bridge.emptyJob(),ungrouped.sections[0].rows.slice(0,1));
  assert.throws(()=>bridge.add(job,grouped.sections[0].rows),/more than once/);
});
test('multiple orders in a section retain clients, including merged origins and colliding row ids',()=>{
  const second=clone(fixture); second.id=827; second.order_number='R-26-0827'; second.client_name='Second';
  const {sections}=prepare([fixture,second]);
  const job=bridge.add(bridge.emptyJob(),sections[0].rows);
  assert.equal(job.rows.length,16);
  const merged=prepare([fixture,second],{groupDimensions:true,mergeAcrossOrders:true});
  assert.equal(merged.sections[0].rows.length,8); assert.equal(merged.sections[0].rows[0].quantity,4);
  assert.equal(new Set(merged.sections[0].rows[0].sources.map(s=>s.client)).size,2);
});
test('different source sections import in Processing order, even with identical display headers',()=>{
  const second=clone(fixture); second.id=99; second.rows.forEach(r=>r.type='6F');
  const {sections}=prepare([fixture,second],{headerOverrides:{'6F':'Same','2 vetri 4F + 16 + 4 LowE':'Same'}});
  const job=bridge.importPrepared(bridge.emptyJob(),sections);
  assert.equal(job.rows.length,16);
  assert.deepEqual(job.rows.map(r=>r.id),clone(sections.flatMap(s=>s.rows.map(r=>r.id))));
});
test('changed values and missing/regrouped rows require explicit snapshot replacement',()=>{
  const {sections}=prepare(); const job=bridge.add(bridge.emptyJob(),sections[0].rows);
  const order=clone(fixture); order.rows[0].dimension='320x790';
  const fresh=prepare([order]); assert.equal(bridge.changes(job,fresh.sections).length,1);
  assert.equal(job.rows[0].width,315);
  const replaced=bridge.importPrepared(job,fresh.sections);
  assert.equal(replaced.rows[0].width,320);
  assert.equal(bridge.changes(replaced,fresh.sections).length,0);
  assert.equal(bridge.changes(job,[]).length,8);
});
test('canonical rounding and orientation used once; label quantity expansion remains separate',()=>{
  const order=clone(fixture); order.rows=[{...order.rows[0],dimension:'791x314'}];
  const {sections,ctx,processing}=prepare([order],{rounded:true,groupDimensions:true});
  const row=sections[0].rows[0]; assert.equal(row.width,790); assert.equal(row.height,315);
  assert.equal(row.quantity,2);
  const labels=ctx.buildProcessingLabelRows(processing.preview.groups[0].lines[0]);
  assert.equal(labels.length,2); assert(labels.every(label=>label.quantity===1));
  assert.equal(bridge.add(bridge.emptyJob(),[row]).rows.length,1);
});
test('all malformed numeric values are reported, including source coercion hidden by grouping',()=>{
  for(const value of [null,undefined,'',' ',0,-1,1.2,'2x','1e2','+2','2.0',true,Infinity,NaN]){
    const {sections}=prepare(); const row=sections[0].rows[0]; row.quantity=value;
    assert(bridge.validate([row]).errors.length, String(value));
  }
  for(const [field,max] of [['quantity',999],['width',10000],['height',10000]]){
    const row=prepare().sections[0].rows[0]; row[field]=max; assert.equal(bridge.validate([row]).errors.length,0);
    row[field]=max+1; assert(bridge.validate([row]).errors.some(e=>e.includes(field)));
    row[field]=1.5; assert(bridge.validate([row]).errors.some(e=>e.includes(field)));
  }
  for(const value of ['1e2','2.0',true,'bad',0,1.5]){
    const order=clone(fixture); order.rows[0].quantity=value;
    assert(bridge.validate(prepare([order],{groupDimensions:true}).sections[0].rows).errors.some(e=>e.includes('source quantity')));
  }
});
test('unsupported shapes and special requirements list every affected source position',()=>{
  const order=clone(fixture);
  order.rows[0].shape='triangle'; order.rows[1].is_rectangular=false; order.rows[2].notes='holes'; order.rows[3].special_requirements=['edge cutouts'];
  const rows=prepare([order],{groupDimensions:true}).sections[0].rows;
  const result=bridge.validate(rows); assert.equal(result.rows.length,0);
  for(let pos=1;pos<=4;pos++) assert(result.errors.some(e=>e.includes(`position ${pos}`)));
  assert.throws(()=>bridge.csv(rows),/unsupported/);
  assert.equal(bridge.validate(rows.slice(4)).errors.length,0);
});
test('remove and clear are isolated; source area and Labels are unchanged',()=>{
  const {processing,sections,ctx}=prepare();
  const labels=processing.preview.groups.flatMap(g=>g.lines.flatMap(l=>ctx.buildProcessingLabelRows(l)));
  const before=JSON.stringify({processing,labels});
  let job=bridge.add(bridge.emptyJob(),sections[0].rows); job.rows.splice(0,1); job=bridge.emptyJob();
  assert.equal(job.rows.length,0); assert.equal(JSON.stringify({processing,labels}),before);
});
test('existing Processing and Labels match pre-feature transformations',()=>{
  // Golden fixture values exercise grouped and ungrouped, original positions, source areas and labels.
  for(const grouped of [true,false]) for(const merge of [true,false]){
    const {processing,ctx}=prepare([clone(fixture)],{groupDimensions:grouped,mergeAcrossOrders:merge,rounded:true});
    assert.equal(processing.preview.meta.rows,8);
    const lines=processing.preview.groups[0].lines;
    assert.equal(lines.reduce((n,l)=>n+l.qty,0),22);
    assert.equal(processing.preview.groups[0].areaValue,10.08);
    const labels=lines.flatMap(l=>ctx.buildProcessingLabelRows(l));
    assert.equal(labels.length,22); assert.equal(labels[0].position,'1'); assert.equal(labels.at(-1).position,'8');
    assert.equal(labels[0].dimension,'315 × 790'); assert.equal(labels.at(-1).dimension,'575 × 530');
  }
});
test('actual asynchronous Processing import keeps busy guard through fetch and releases it on failure',async()=>{
  const app=fs.readFileSync(path.join(root,'docs/js/app.js'),'utf8');
  const start=app.indexOf('async function sendOrderToProcessing(');
  const end=app.indexOf('function syncProcessingHeaderEditor(',start);
  let resolveFetch;
  const ctx={processingBridgeBusy:0,fetchOrder:()=>new Promise(resolve=>resolveFetch=resolve)};
  vm.createContext(ctx); vm.runInContext(app.slice(start,end),ctx);
  const pending=ctx.sendOrderToProcessing(826);
  assert.equal(ctx.processingBridgeBusy,1);
  assert.throws(()=>bridge.collect({},ctx.processingBridgeBusy>0),/busy/);
  resolveFetch(null);
  await assert.rejects(pending,/no rows/);
  assert.equal(ctx.processingBridgeBusy,0);
});

test('R-26-0830 matches both screenshots exactly across glass types and repeated imports',()=>{
  const order=require('./fixtures/perfect_cut_processing_order.json');
  const plain=prepare([order],{rounded:true});
  const grouped=prepare([order],{rounded:true,groupDimensions:true});
  let job=bridge.importPrepared(bridge.emptyJob(),plain.sections);
  assert.equal(bridge.csv(job.rows),'quantity,width,height\r\n1,738,1835\r\n1,815,1903\r\n1,815,1903\r\n1,1268,168\r\n1,433,848\r\n');
  const before=JSON.stringify(grouped.processing);
  job=bridge.importPrepared(job,grouped.sections);
  const expected='quantity,width,height\r\n1,738,1835\r\n2,815,1903\r\n1,1268,168\r\n1,433,848\r\n';
  assert.equal(bridge.csv(job.rows),expected);
  assert.equal(job.rows.reduce((sum,row)=>sum+row.quantity,0),5);
  assert.equal(job.rows.at(-1).width,433); // No global sorting or glass filtering.
  job=bridge.importPrepared(job,grouped.sections);
  assert.equal(bridge.csv(job.rows),expected);
  assert.equal(JSON.stringify(grouped.processing),before);
});
test('manual import preserves saved mm, interleaved types, positions, quantities and independent area',()=>{
  const order=clone(require('./fixtures/perfect_cut_manual_order.json'));
  const before=JSON.stringify(order);
  const sections=bridge.collectManual([order]);
  const job=bridge.importPrepared(bridge.emptyJob(),sections);
  assert.equal(bridge.csv(job.rows),'quantity,width,height\r\n2,791,314\r\n3,500,1200\r\n1,791,314\r\n');
  assert.equal(job.rows[2].sources[0].position,'A-2');
  assert.equal(job.rows[0].sources[0].bridgeSource.declaredArea,'4.100');
  assert.equal(JSON.stringify(order),before);
  order.rows[0].width_mm=900;
  assert.equal(job.rows[0].width,791);
  assert.equal(bridge.changes(job,bridge.collectManual([order]),'Manual Orders').length,1);
  assert.equal(bridge.importPrepared(job,sections).rows.length,3);
});
test('manual status guards and malformed saved values fail closed without coercion',()=>{
  const fixture=require('./fixtures/perfect_cut_manual_order.json');
  for(const status of ['draft','cancelled','finished','unexpected']){
    assert.throws(()=>bridge.collectManual([{...fixture,status}]),/only approved or processing/);
  }
  assert.equal(bridge.collectManual([{...fixture,status:'processing'}]).length,1);
  assert.throws(()=>bridge.collectManual([{...fixture,rows:[]}]),/no saved rows/);
  for(const field of ['quantity','width_mm','height_mm']) for(const value of [null,true,'1e2','5.0',1.5,0,-1]){
    const order=clone(fixture); order.rows[0][field]=value;
    assert(bridge.validate(bridge.collectManual([order])[0].rows).errors.length);
  }
  const order=clone(fixture); order.rows[0].shape='triangle';
  assert(bridge.validate(bridge.collectManual([order])[0].rows).errors.some(e=>e.includes('triangle')));
});
test('manual source identity is separate from PDFs and preserves equal-sized rows across orders',()=>{
  const fixture=require('./fixtures/perfect_cut_manual_order.json');
  const sections=bridge.collectManual([fixture,{...fixture,id:32,order_number:'M-032'}]);
  const job=bridge.importPrepared(bridge.emptyJob(),sections);
  assert.equal(job.rows.length,6);
  assert.equal(new Set(job.rows.flatMap(r=>r.sourceIds)).size,6);
  assert.equal(bridge.validate(job.rows).errors.length,0);
});
