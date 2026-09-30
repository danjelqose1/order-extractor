const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const sheets = require('../docs/js/production-sheet.js');
const fixture = require('./fixtures/perfect_cut_order.json');
function prepared(){
  const ctx={DIMENSION_TOLERANCE_MM:1,UNKNOWN_CLIENT_LABEL:'(Unknown Client)'};
  ctx.window=ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../docs/js/platform-workflows.js'),'utf8'),ctx);
  const rows=ctx.convertOrderToProcessingEntry(structuredClone(fixture)).rows;
  const options={normalizeLPtoG:true,decimalSeparator:'comma',groupDimensions:false,mergeAcrossOrders:false};
  return {rows,options,preview:ctx.generateMotherSheet(rows,options)};
}
test('captures exact prepared rows, headers and quantities once without mutation',()=>{
  const processing=prepared(), before=JSON.stringify(processing);
  const captured=sheets.capture(processing);
  assert.equal(captured.source.text,processing.preview.text);
  assert.equal(captured.source.row_count,8);
  assert.equal(captured.source.piece_count,22);
  assert.deepEqual(Array.from(captured.source.glass_headers),Array.from(processing.preview.groups,g=>g.headerText));
  assert.equal(JSON.stringify(processing),before);
});
test('source signature changes for provenance and prepared display edits',()=>{
  const processing=prepared(), original=sheets.capture(processing).signature;
  processing.preview.groups[0].lines[0].originRows[0].position='changed';
  assert.notEqual(sheets.capture(processing).signature,original);
  const updated=sheets.capture(processing).signature;
  processing.preview.text+='\nChanged note';
  assert.notEqual(sheets.capture(processing).signature,updated);
});
test('empty, busy and invalid source rows cannot be printed as plausible data',()=>{
  assert.throws(()=>sheets.capture({preview:{groups:[],text:''}}),/Add orders/);
  const processing=prepared();
  assert.throws(()=>sheets.capture(processing,true),/busy/);
  processing.preview.groups[0].lines[0].qty=0;
  assert.throws(()=>sheets.capture(processing),/invalid dimensions or quantities/);
});
