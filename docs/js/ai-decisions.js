/* Advisory UI: all content from orders/models is escaped; no production mutations. */
(() => {
  let records = [];
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}/api/ai-decisions${path}`, { ...options, headers: {'Content-Type':'application/json', ...($('aiAccessKey').value ? {'X-App-Key': $('aiAccessKey').value} : {}), ...options.headers} });
    if (!response.ok) throw new Error(response.status === 401 ? 'Backend authorization required.' : `Request failed (${response.status}). Manual workflows remain available.`);
    return response.json();
  }
  function indicator(id, kind = 'order') {
    const record = records.find(r => String(r.order_id) === String(id) && r.kind === kind);
    return record ? `<span class="ai-indicator">${esc(record.readiness)} · ${esc(record.classification.replaceAll('_', ' '))}</span>` : '';
  }
  async function refresh() {
    try {
      const data = await request(''); records = data.items;
      $('aiSummary').innerHTML = Object.entries(data.summary).map(([label, value]) => `<div class="ai-stat"><strong>${esc(value)}</strong><span>${esc(label.replaceAll('_',' '))}</span></div>`).join('');
      $('aiCards').innerHTML = records.map(r => `<article class="card ai-card"><div class="ai-card-heading"><h3>${esc(r.client || 'Unknown client')} · ${esc(r.order_number || 'Unknown order')}</h3><span class="ai-indicator">${esc(r.readiness)}</span></div>
        <p><strong>${esc(r.classification.replaceAll('_', ' '))}</strong> · ${r.confidence == null ? 'Deterministic evidence / confidence unavailable' : `${Math.round(r.confidence*100)}% model confidence`} · ${esc(r.category)}</p>
        <ul>${[...r.blocking_issues,...r.issues].map(i => `<li>${esc(i)}</li>`).join('') || '<li>No detected blocking issues</li>'}</ul>
        ${r.candidates.map(c => `<details><summary>Matching ${esc(c.kind)} #${esc(c.id)} · ${esc(c.classification || 'Candidate')}</summary><p>Matched: ${esc(c.matched.join(', '))}</p><pre>${esc(JSON.stringify(c.differences, null, 2))}</pre></details>`).join('')}
        <p>Recommended action: <strong>${esc(r.route)}</strong></p>
        <p class="muted">Latest review: ${esc(r.audit.at(-1)?.action || 'Pending')}. Actions record a review decision; order values and statuses stay unchanged.</p>
        <div class="ai-controls">${[['confirm_duplicate','Confirm Duplicate'],['legitimate','Mark as Legitimate'],['open','Open Original'],['review','Send to Review'],['accept','Accept Recommendation'],['ignore','Ignore Recommendation'],['reset','Undo Review Action']].map(([a,l]) => `<button class="btn small" data-ai-action="${a}" data-ai-id="${r.id}">${l}</button>`).join('')}</div>
        <details><summary>Audit trail</summary><pre>${esc(JSON.stringify(r.audit, null, 2))}</pre></details></article>`).join('') || '<div class="card">No orders evaluated yet. New extractions will appear here when enabled.</div>';
      $('aiStatus').textContent = '';
    } catch (error) { $('aiStatus').textContent = error.message; }
  }
  $('aiRefresh').addEventListener('click', refresh);
  $('aiEvaluate').addEventListener('click', async () => {
    const id = Number($('aiOrderId').value);
    if (!Number.isInteger(id) || id < 1) { $('aiStatus').textContent = 'Enter a valid order ID.'; return; }
    $('aiEvaluate').disabled = true; $('aiStatus').textContent = 'Evaluating evidence…';
    try { await request(`/orders/${id}/evaluate?kind=${$('aiOrderKind').value}`, {method:'POST'}); await refresh(); }
    catch (e) { $('aiStatus').textContent = e.message; }
    finally { $('aiEvaluate').disabled = false; }
  });
  $('aiCards').addEventListener('click', async event => {
    const button = event.target.closest('[data-ai-action]'); if (!button) return;
    const record = records.find(r => String(r.id) === button.dataset.aiId); if (!record) return;
    const action = button.dataset.aiAction;
    if (action === 'open') {
      if (record.kind === 'order') { activateTab('history'); await openOrderFromList(record.order_id); }
      else { activateTab('manual'); await openManualOrder(record.order_id, true); }
      return;
    }
    button.disabled = true;
    try { await request(`/${record.id}/override`, {method:'POST', body:JSON.stringify({action})}); await refresh(); }
    catch(e) { $('aiStatus').textContent = e.message; button.disabled = false; }
  });
  window.AIDecisions = {refresh, indicator};
  fetch(`${API_BASE}/api/features`).then(r => r.json()).then(f => {
    if (f.ai_decisions) { $('aiDecisionsNav').hidden = false; refresh(); }
  }).catch(() => {});
})();
