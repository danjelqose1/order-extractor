const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const fixture=require('./fixtures/perfect_cut_order.json');
function context(){
  const ctx={DIMENSION_TOLERANCE_MM:1,UNKNOWN_CLIENT_LABEL:'(Unknown Client)'};
  ctx.window=ctx;vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../docs/js/platform-workflows.js'),'utf8'),ctx);
  return ctx;
}
test('centimetre display converts mm and uses comma without rounding fractional measurements',()=>{
  const ctx=context();
  assert.equal(ctx.formatProcessingDimension(1250,'1250','dot','cm'),'125,0');
  assert.equal(ctx.formatProcessingDimension(805,'805','comma','cm'),'80,5');
  assert.equal(ctx.formatProcessingDimension(1250.25,'1250.25','dot','cm'),'125,025');
  assert.equal(ctx.formatProcessingDimension(null,'unreadable','comma','cm'),'?');
  assert.equal(ctx.formatProcessingDimension(1250,'1250','comma','mm'),'1250');
});
test('unit toggle changes sheet presentation only and is reversible for grouped and ungrouped sheets',()=>{
  const ctx=context();
  for(const grouped of [false,true]){
    const rows=ctx.convertOrderToProcessingEntry(structuredClone(fixture)).rows;
    const original=JSON.stringify(rows);
    const options={normalizeLPtoG:true,decimalSeparator:'comma',groupDimensions:grouped,mergeAcrossOrders:false};
    const mm=ctx.generateMotherSheet(rows,options);
    const cm=ctx.generateMotherSheet(rows,{...options,dimensionUnit:'cm'});
    assert.equal(JSON.stringify(rows),original);
    assert.equal(JSON.stringify(cm.groups),JSON.stringify(mm.groups));
    assert.equal(cm.meta.dimensionUnit,'cm');
    assert.match(cm.text,/Dimensions: cm/);
    const line=mm.groups[0].lines[0];
    assert(cm.text.includes(`${ctx.formatProcessingDimension(line.width,'','comma','cm')} × ${ctx.formatProcessingDimension(line.height,'','comma','cm')}`));
    assert.equal(ctx.generateMotherSheet(rows,{...options,dimensionUnit:'mm'}).text,mm.text);
    assert.equal(ctx.formatProcessingDimensionLabel(cm.groups[0].lines[0]),ctx.formatProcessingDimensionLabel(line));
  }
});
