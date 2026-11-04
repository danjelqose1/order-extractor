/* Client ledgers use only saved standalone delivery documents and payments. */
(() => {
  'use strict';
  const root = document.getElementById('tabFleteDalje');
  const panel = document.getElementById('fdClients');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const money = cents => `${new Intl.NumberFormat('sq-AL', { maximumFractionDigits: 2 }).format(cents / 100)} lekë`;
  const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; };
  const methods = {cash:'Para në dorë',bank:'Transfertë bankare',other:'Tjetër'};
  let cards = [], selected = null, pendingId = null, busy = false;
  panel.innerHTML = `<div class="fd-client-layout"><aside class="fd-client-list"><h3>Kartelat e klientëve</h3><label>Kërko klientin<input id="fdClientSearch" type="search" placeholder="Emri i klientit"></label><p class="fd-client-help">Fletët grupohen sipas emrit të marrësit. Shkronjat e mëdha dhe hapësirat nuk krijojnë kartela të veçanta.</p><button class="btn small" id="fdClientsRefresh">Rifresko</button><div id="fdClientList"></div></aside><section class="fd-client-detail"><div id="fdClientDetail"></div><p id="fdPaymentStatus" role="status" aria-live="polite"></p><form id="fdPaymentForm" hidden><h3>Regjistro pagesë</h3><div class="fd-payment-fields"><label>Shuma (lekë)<input name="amount" type="text" inputmode="decimal" required placeholder="p.sh. 5000 ose 5000,50"></label><label>Data e pagesës<input name="date" type="date" required></label><label>Mënyra e pagesës<select name="method"><option value="cash">Para në dorë</option><option value="bank">Transfertë bankare</option><option value="other">Tjetër</option></select></label><label>Shënim / referencë<input name="notes" maxlength="2000" placeholder="Nr. transferte ose shënim"></label></div><button class="btn primary" type="submit">Ruaj pagesën</button><p class="fd-client-help">Pagesa zbritet nga gjendja e klientit. Fletë-daljet mbeten me vlerën e tyre origjinale.</p></form></section></div>`;
  const $ = id => panel.querySelector(`#${id}`);
  const form = $('fdPaymentForm');
  form.elements.date.value = today();
  const active = () => cards.find(card => card.key === selected);
  const draft = () => [...form.elements].some(el => ['amount','notes'].includes(el.name) && el.value.trim());
  function canChange() { return !busy && (!draft() || confirm('Ka një pagesë të paruajtur. Të hiqen të dhënat e saj?')); }
  function clearDraft() { form.reset(); form.elements.date.value=today(); pendingId=null; }
  function showView(view) {
    if(view==='editor' && !canChange()) return;
    if(view==='editor') clearDraft();
    panel.hidden=view!=='clients';
    root.querySelector('.fd-layout').hidden=view==='clients';
    root.querySelector('.fd-header .fd-toolbar').hidden=view==='clients';
    root.querySelector('#fdStatus').hidden=view==='clients';
    root.querySelectorAll('[data-fd-view]').forEach(button=>{
      const on=button.dataset.fdView===view; button.classList.toggle('primary',on);button.setAttribute('aria-pressed',String(on));
    });
  }
  function totals(card) {
    return `<div class="fd-client-totals"><div><span>Totali i fletë-daljeve</span><strong>${money(card.documents_cents)}</strong></div><div><span>Pagesa gjithsej</span><strong>${money(card.paid_cents)}</strong></div><div><span>${card.balance_cents<0?'Parapagim / tepricë':'Për t’u paguar'}</span><strong>${money(Math.abs(card.balance_cents))}</strong></div></div>`;
  }
  function rows(card, print=false) {
    return card.entries.map(row=>`<tr class="${row.voided?'fd-payment-voided':''}"><td>${esc(row.date)}</td><td>${row.kind==='document'?(print?`Fletë nr. ${esc(row.number)}`:`<button class="btn small" data-ledger-open="${esc(row.id)}">Fletë nr. ${esc(row.number)}</button>`):`Pagesë${row.voided?' e anuluar':''}`}<small>${esc(row.kind==='document'?row.address:methods[row.method])}${row.notes?' · '+esc(row.notes):''}${row.voided?' · '+money(row.amount_cents):''}</small></td><td>${row.debit_cents?money(row.debit_cents):'—'}</td><td>${row.credit_cents?money(row.credit_cents):'—'}</td><td>${money(row.balance_cents)}</td>${print?'':`<td>${row.kind==='payment'&&!row.voided?`<button class="btn small danger" data-void-payment="${esc(row.id)}" data-version="${row.version}">Anulo pagesën</button>`:''}</td>`}</tr>`).join('');
  }
  function ledger(card, print=false) {
    return `<div class="fd-table-wrap"><table class="fd-ledger"><thead><tr>${['Data','Përshkrimi','Fletë-dalje','Pagesë','Gjendja',...(print?[]:['Veprime'])].map(label=>`<th>${label}</th>`).join('')}</tr></thead><tbody>${rows(card,print)}</tbody></table></div>`;
  }
  function renderList() {
    const query=$('fdClientSearch').value.trim().toLocaleLowerCase('sq');
    const matches=cards.filter(card=>card.name.toLocaleLowerCase('sq').includes(query));
    $('fdClientList').innerHTML=matches.length?matches.map(card=>`<button class="fd-client-choice ${card.key===selected?'active':''}" data-client="${esc(card.key)}"><strong>${esc(card.name)}</strong><span>${card.balance_cents<0?'Tepricë':'Gjendja'}: ${money(Math.abs(card.balance_cents))}</span></button>`).join(''):'<p>Nuk ka klientë. Ruani një fletë-dalje për të hapur kartelën.</p>';
  }
  function renderDetail() {
    const card=active();
    form.hidden=!card || !card.key;
    $('fdClientDetail').innerHTML=card?`<div class="fd-client-heading"><div><h2>${esc(card.name)}</h2><p>Kartela e klientit · ALL</p></div><button class="btn" id="fdPrintClient">Printo kartelën / PDF</button></div><p class="fd-client-help">Nëse ndryshoni emrin e marrësit të një flete, ajo kalon në kartelën me atë emër. Pagesat mbeten në kartelën ku janë regjistruar.</p>${ledger(card)}${totals(card)}`:'<h2>Kartelat e klientëve</h2><p>Zgjidhni një klient për të parë fletët, pagesat dhe gjendjen.</p>';
  }
  function receive(data) {
    cards=data.clients || [];
    if(!cards.some(card=>card.key===selected)) {selected=cards[0]?.key ?? null;clearDraft();}
    renderList();renderDetail();
  }
  root.querySelectorAll('[data-fd-view]').forEach(button=>button.addEventListener('click',()=>showView(button.dataset.fdView)));
  $('fdClientSearch').addEventListener('input',renderList);
  // Retain the ID after a timeout: changed details must never create a duplicate payment.
  async function transact(action) {
    if(busy) return;
    busy=true;panel.inert=true;
    try {await action();} catch(error) {$('fdPaymentStatus').textContent=`Gabim: ${error.message}`;}
    finally {busy=false;panel.inert=false;}
  }
  form.addEventListener('submit',event=>{
    event.preventDefault();
    transact(async()=>{
      const card=active(); if(!card) return;
      const text=form.elements.amount.value.trim().replace(',','.');
      if(!/^\d+(\.\d{1,2})?$/.test(text) || Number(text)<=0 || Number(text)>1e9) throw Error('Vendosni një shumë pozitive me deri në dy shifra pas presjes.');
      const [whole,fraction='']=text.split('.');
      const amount=Number(whole)*100+Number(fraction.padEnd(2,'0'));
      if(amount>card.balance_cents && !confirm('Pagesa tejkalon gjendjen. Diferenca do të ruhet si parapagim. Vazhdo?')) return;
      pendingId ||= crypto.randomUUID();
      await window.FleteDalje.request('/payments',{method:'POST',body:JSON.stringify({id:pendingId,recipient:card.name,date:form.elements.date.value,amount:text,method:form.elements.method.value,notes:form.elements.notes.value.trim()})});
      clearDraft();$('fdPaymentStatus').textContent='Pagesa u ruajt.';
      try {await window.FleteDalje.refresh();} catch { $('fdPaymentStatus').textContent='Pagesa u ruajt. Rifreskoni kartelën për gjendjen e fundit.'; }
    });
  });
  panel.addEventListener('click',event=>{
    const button=event.target.closest('button');if(!button || busy) return;
    if(button.dataset.client!==undefined) {
      if(button.dataset.client!==selected && canChange()){clearDraft();selected=button.dataset.client;renderList();renderDetail();$('fdPaymentStatus').textContent='';}return;
    }
    if(button.dataset.ledgerOpen) {
      if(!canChange() || !window.FleteDalje.canLeave()) return;
      showView('editor');
      const search=root.querySelector('#fdSearch');search.value='';search.dispatchEvent(new Event('input',{bubbles:true}));
      root.querySelector(`[data-open="${CSS.escape(button.dataset.ledgerOpen)}"]`)?.click();
      return;
    }
    if(button.dataset.voidPayment) {
      if(!confirm('Të anulohet kjo pagesë? Ajo do të mbetet në historik dhe shuma do t’i shtohet përsëri gjendjes.')) return;
      transact(async()=>{await window.FleteDalje.request(`/payments/${encodeURIComponent(button.dataset.voidPayment)}/void?version=${button.dataset.version}`,{method:'POST'});$('fdPaymentStatus').textContent='Pagesa u anulua.';await window.FleteDalje.refresh();});return;
    }
    if(button.id==='fdClientsRefresh') {transact(()=>window.FleteDalje.refresh());return;}
    if(button.id==='fdPrintClient') {
      const card=active();if(!card) return;
      document.getElementById('fdPrint')?.remove();const sheet=document.createElement('div');sheet.id='fdPrint';sheet.lang='sq';sheet.className='fd-client-print';
      sheet.innerHTML=`<h1>KARTELA E KLIENTIT</h1><p>${esc(card.name)} · ${today()} · ALL</p>${ledger(card,true)}${totals(card)}`;
      document.body.append(sheet);document.body.classList.add('fd-printing');window.print();
    }
  });
  window.addEventListener('beforeunload',event=>{if(draft()){event.preventDefault();event.returnValue='';}});
  window.FleteDaljeClients={receive,showEditor:()=>showView('editor'),canLeave:()=>canChange()};
  const initial=window.FleteDalje.getArchive();if(initial) receive(initial);
})();
