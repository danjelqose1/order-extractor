const {test} = require('node:test');
const assert = require('node:assert/strict');
const grouping = require('../docs/js/manual-dimension-groups.js');
const bridge = require('../docs/js/perfect-cut-bridge.js');
const fixture = require('./fixtures/perfect_cut_manual_order.json');

test('grouping preserves first appearance, orientation and all source rows across glass types',()=>{
  const rows=structuredClone(fixture.rows);
  rows[2].glass_type='Different glass';
  rows.push({...rows[0],id:304,width_mm:314,height_mm:791});
  const before=JSON.stringify(rows);
  assert.deepEqual(grouping.buckets(rows),[
    {indexes:[0,2],quantity:3},{indexes:[1],quantity:3},{indexes:[3],quantity:2},
  ]);
  assert.equal(JSON.stringify(rows),before);
});

test('grouping never merges malformed quantities or dimensions into plausible values',()=>{
  for (const field of ['quantity','width_mm','height_mm']) for(const value of [null,true,'1e2','',0,-1,NaN]){
    const row={...fixture.rows[0],[field]:value};
    assert.deepEqual(grouping.buckets([row,row]).map(g=>g.indexes),[[0],[1]]);
  }
  assert.deepEqual(grouping.buckets([{width_mm:'10.5',height_mm:'20',quantity:'2'},
    {width_mm:10.5,height_mm:20,quantity:3}]),[{indexes:[0,1],quantity:5}]);
});

test('manual grouping flows to Bridge and is reversible with source change detection',()=>{
  const before=JSON.stringify(fixture);
  grouping.setGrouped(fixture.id,true);
  try{
    const grouped=bridge.collectManual([fixture]);
    const job=bridge.importPrepared(bridge.emptyJob(),grouped);
    assert.equal(bridge.csv(job.rows),'quantity,width,height\r\n3,791,314\r\n3,500,1200\r\n');
    assert.deepEqual(job.rows[0].sources.map(s=>s.position),['A-1','A-2']);
    assert.equal(job.rows[0].manualGroup,1);
    assert.equal(job.rows[1].manualGroup,2);
    const edited=structuredClone(fixture); edited.rows[2].quantity=5;
    assert(bridge.changes(job,bridge.collectManual([edited])).length);
    grouping.setGrouped(fixture.id,false);
    const original=bridge.collectManual([fixture]);
    assert.equal(original[0].rows.length,3);
    assert(bridge.changes(job,original).length);
    assert.equal(JSON.stringify(fixture),before);
  }finally{grouping.setGrouped(fixture.id,false);}
});

test('grouped Bridge quantity limits and requirements retain every original source',()=>{
  const order=structuredClone(fixture);
  order.rows[0].quantity=999;
  order.rows[2].notes='holes';
  grouping.setGrouped(order.id,true);
  try{
    const result=bridge.validate(bridge.collectManual([order])[0].rows);
    assert(result.errors.some(e=>e.includes('quantity must be a whole number')));
    assert(result.errors.some(e=>e.startsWith('M-031 / position A-2:')&&e.includes('holes')));
    assert.equal(result.rows.length,0);
  }finally{grouping.setGrouped(order.id,false);}
});
