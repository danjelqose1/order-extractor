/* Standalone delivery documents. All mutations go only to /api/flete-dalje. */
(() => {
  'use strict';
  const root = document.getElementById('tabFleteDalje');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const blankLine = () => ({ description: '', unit: 'm²', quantity: '', price: '' });
  let numberManual = false, lastSaved = null, archiveData = null, payments = [], records = [], current, baseline, loaded = false, busy = false;
  const snapshot = () => JSON.stringify(current);
  const dirty = () => current && snapshot() !== baseline;
  const canLeave = () => (!window.FleteDaljeClients || window.FleteDaljeClients.canLeave()) && (!dirty() || confirm('Ka ndryshime të paruajtura. Dëshironi t’i hidhni poshtë?'));
  // Keep the reference's Number parsing and Math.round per line, before summing.
  const number = value => {
    let text = String(value ?? '').trim().replace(/\s/g, '');
    if (text.includes(',') && !text.includes('.')) text = text.replace(',', '.');
    else if (text.includes(',') && text.includes('.')) text = text.replace(/\./g, '').replace(',', '.');
    return Number(text) || 0;
  };
  const lineValue = line => Math.round(number(line.quantity) * number(line.price));
  const money = value => new Intl.NumberFormat('sq-AL', { maximumFractionDigits: 0 }).format(value || 0);
  const total = doc => doc.lines.reduce((sum, line) => sum + lineValue(line), 0);
  const hasData = line => line.description.trim() || number(line.quantity) || number(line.price);
  const pad = doc => { while (doc.lines.length < 1) doc.lines.push(blankLine()); return doc; };
  const clientKey = name => String(name || '').normalize('NFKC').toLocaleLowerCase('sq').trim().replace(/\s+/g,' ');
  const nextNumber = (recipient='') => Math.max(0, ...records.filter(doc=>clientKey(doc.recipient)===clientKey(recipient)).map(doc=>doc.number)) + 1;
  const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; };
  function fresh() {
    numberManual=false;
    current = { number: nextNumber(), recipient: '', date: today(), address: '', notes: '', lines: [blankLine()] };
    baseline = snapshot(); renderEditor(); renderArchive();
  }
  root.innerHTML = `<div class="fd-header"><div><h2>FLETË–DALJE <span>DIGJITALE</span></h2><p>Blloku elektronik për fabrikën • pa letra • pa llogaritje me dorë</p></div><div class="fd-toolbar"><button class="btn" data-fd="new">+ Fletë e re</button><button class="btn primary" data-fd="save">Ruaj fletën</button><button class="btn" data-fd="print">Printo / PDF</button><button class="btn" data-fd="history">Historiku</button></div></div>
    <div class="fd-toolbar fd-view-tabs"><button class="btn primary" data-fd-view="editor" aria-pressed="true">Fletë-Dalje</button><button class="btn" data-fd-view="clients" aria-pressed="false">Kartelat e klientëve</button></div><section id="fdClients" hidden aria-label="Kartelat e klientëve"></section><p id="fdStatus" role="status" aria-live="polite"></p><div class="fd-layout"><div id="fdEditor" class="fd-paper"></div><aside class="fd-archive" id="fdArchive"><h3>Historiku i fletëve</h3><label class="fd-search-label">Kërko<input id="fdSearch" type="search" placeholder="Kërko nr., klientin, datën ose adresën..."></label><div class="fd-archive-heading"><span id="fdCount"></span><button class="btn small" data-fd="refresh">Rifresko</button></div><div id="fdRecords"></div><div class="fd-rule"></div><h3>Backup & eksport</h3><p>Fletët ruhen në server, veçmas nga porositë. Backup përfshin fletët dhe pagesat e ruajtura.</p><div class="fd-toolbar"><button class="btn small" data-fd="backup">Eksporto backup</button><button class="btn small" data-fd="import">Importo backup</button><button class="btn small" data-fd="csv">Eksporto Excel / CSV</button></div><input id="fdImport" type="file" accept="application/json,.json" hidden><div class="fd-alert">Për ruajtje afatgjatë përdor <b>Eksporto backup</b>. Mund të importoni edhe backup nga blloku origjinal.</div></aside></div>`;
  const $ = id => root.querySelector(`#${id}`);
  function status(message) { $('fdStatus').textContent = message; }
  function updateTotals() {
    try {
      current.lines.forEach((line, i) => { const el = root.querySelector(`[data-amount="${i}"]`); if(el) el.textContent = lineValue(line) ? money(lineValue(line)) : ''; });
      $('fdTotal').textContent = money(total(current));
      $('fdSaveState').textContent = dirty() ? 'PA RUAJTUR' : current.id ? 'E RUAJTUR' : 'FLETË E RE';
      status('');
    } catch (error) { status(error.message); }
  }
  function renderEditor() {
    $('fdEditor').innerHTML = `<div class="fd-topbrand"><h2>FLETË - DALJE</h2><span id="fdSaveState" class="fd-pill" aria-live="polite"></span></div><div class="fd-fields">${[['number','Nr. flete','number'],['recipient','Marrësi / Klienti','text'],['date','Data','date'],['address','Adresa ku shkon malli','text']].map(([key,label,type])=>`<label class="fd-field-${key}">${label}<input data-field="${key}" type="${type}" ${key==='number'?'min="1" step="1"':''} ${key==='recipient'?'placeholder="p.sh. Bledi Husi" autocomplete="off" list="fdRecipientSuggestions"':key==='address'?'placeholder="Adresa ose vendndodhja e dorëzimit"':''} value="${esc(current[key])}"></label>`).join('')}</div>
      <div class="fd-table-wrap"><table><thead><tr>${['Nr.','Emërtimi i mallit','Njësia','Sasia','Çmimi','Vlefta'].map(v=>`<th>${v}</th>`).join('')}</tr></thead><tbody>${current.lines.map((line,i)=>`<tr><td>${i+1}</td>${['description','unit','quantity','price'].map(key=>`<td><input aria-label="${{description:'Emërtimi',unit:'Njësia',quantity:'Sasia',price:'Çmimi'}[key]} ${i+1}" data-line="${i}" data-key="${key}" value="${esc(line[key])}" type="text" ${['quantity','price'].includes(key)?'inputmode="decimal"':key==='description'?'placeholder="p.sh. TR + 12 + TR"':''}></td>`).join('')}<td data-amount="${i}"></td></tr>`).join('')}</tbody></table></div><div class="fd-toolbar"><button class="btn small" data-fd="add">+ Shto rresht</button><button class="btn small" data-fd="reset">Kthe në 1 rresht</button></div><div class="fd-footgrid"><label>Shënime (opsionale)<textarea data-field="notes" rows="3" placeholder="Shënime, porosi speciale, transport...">${esc(current.notes)}</textarea></label><div class="fd-totals"><span><strong>SHUMA</strong><br><small>Totali në lekë</small></span><b id="fdTotal"></b></div></div><div class="fd-signature">MARRËSI: ___________________________________________</div>`;
    const suggestions=document.createElement('datalist');suggestions.id='fdRecipientSuggestions';
    suggestions.innerHTML=(archiveData?.clients || []).filter(card=>card.key).map(card=>`<option value="${esc(card.name)}"></option>`).join('');
    $('fdEditor').append(suggestions);
    updateTotals();
  }
  function renderArchive() {
    const query = $('fdSearch').value.trim().toLocaleLowerCase('sq');
    const matches = records.filter(doc => [doc.number, doc.recipient, doc.date, doc.address].some(v=>String(v).toLocaleLowerCase('sq').includes(query)));
    $('fdCount').textContent = `${records.length} fletë të ruajtura`;
    $('fdRecords').innerHTML = matches.length ? matches.map(doc=>`<article class="fd-entry ${current?.id===doc.id?'active':''}"><button class="fd-entry-open" data-open="${esc(doc.id)}"><span class="fd-entry-main"><strong>#${esc(doc.number)} · ${esc(doc.recipient)}</strong><span class="fd-amount">${money(total(doc))} lekë</span></span><span class="fd-meta">${esc(doc.date)} ${doc.address?'· '+esc(doc.address):''}</span></button><button class="btn small" data-print-document="${esc(doc.id)}">Printo / PDF</button> <button class="btn small danger" data-delete="${esc(doc.id)}" aria-label="Fshi fletën nr. ${esc(doc.number)}">Fshi</button></article>`).join('') : '<p class="fd-empty">Nuk ka fletë.</p>';
  }
  async function request(path = '', options = {}) {
    const response = await fetch(`${API_BASE}/api/flete-dalje${path}`, { ...options, headers: { 'Content-Type':'application/json', ...options.headers }, signal: AbortSignal.timeout(15000) });
    const body = await response.json();
    if (!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : 'Të dhëna të pavlefshme. Kontrolloni fushat.');
    return body;
  }
  function receive(data) {
    archiveData = data; records = data.documents; payments = data.payments || []; loaded = true; renderArchive();
    window.FleteDaljeClients?.receive(data);
  }
  async function refresh() { receive(await request()); }
  async function load() {
    if (loaded || busy) return;
    busy = true;
    try { await refresh(); if (!current) fresh(); }
    catch(error) { status(`Arkivi nuk u ngarkua: ${error.message} Përdorni Rifresko për të provuar përsëri.`); }
    finally { busy=false; }
  }
  function download(name, data, type) {
    const url = URL.createObjectURL(new Blob([data], {type}));
    const a = document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function print(doc=current) {
    if (doc===current && dirty() && !confirm('Ka ndryshime të paruajtura. Printo pa i ruajtur?')) return;
    document.getElementById('fdPrint')?.remove();
    const sheet = document.createElement('div'); sheet.id='fdPrint'; sheet.lang='sq';
    sheet.innerHTML = `<h1>FLETË - DALJE</h1><div class="fd-print-fields">${[['number','Nr. flete'],['recipient','Marrësi / Klienti'],['date','Data'],['address','Adresa ku shkon malli']].map(([key,label])=>`<div class="fd-print-${key}"><span>${label}</span><p>${esc(doc[key])}</p></div>`).join('')}</div><table><thead><tr>${['Nr.','Emërtimi i mallit','Njësia','Sasia','Çmimi','Vlefta'].map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${doc.lines.map((line,i)=>`<tr><td>${i+1}</td><td>${esc(line.description)}</td><td>${esc(line.unit)}</td><td>${esc(line.quantity)}</td><td>${esc(line.price)}</td><td>${lineValue(line)?money(lineValue(line)):''}</td></tr>`).join('')}</tbody></table><div class="fd-print-total"><span><strong>SHUMA</strong><br>Totali në lekë</span><b>${money(total(doc))}</b></div><div class="fd-print-sign"><span>MARRËSI: ____________________________</span><span>Nënshkrimi: ______________________</span></div>${doc.notes?`<p class="fd-print-notes">Shënime: ${esc(doc.notes)}</p>`:''}`;
    document.body.append(sheet); document.body.classList.add('fd-printing'); window.print();
  }
  window.addEventListener('afterprint',()=>{ document.body.classList.remove('fd-printing'); document.getElementById('fdPrint')?.remove(); });
  root.addEventListener('input',event=>{
    const el=event.target;
    if (el.closest('#fdClients')) return;
    if(el.id==='fdSearch') return renderArchive();
    if(el.dataset.field==='number') numberManual=true;
    if(el.dataset.field) current[el.dataset.field]=el.dataset.field==='number'?Number(el.value):el.value;
    if(el.dataset.field==='recipient' && !current.id && !numberManual) {
      current.number=nextNumber(current.recipient);root.querySelector('[data-field=number]').value=current.number;
    }
    if(el.dataset.line!==undefined) current.lines[Number(el.dataset.line)][el.dataset.key]=el.value;
    updateTotals();
  });
  root.addEventListener('click', async event=>{
    const button=event.target.closest('button'); if(!button || busy || button.closest('#fdClients') || button.dataset.fdView) return;
    busy=true;
    $('fdEditor').inert=true;
    try {
      if(button.dataset.printDocument) {
        await refresh();const doc=records.find(row=>row.id===button.dataset.printDocument);
        if(!doc) throw Error('Fleta nuk u gjet.');print(doc);return;
      }
      if(button.dataset.open) {
        if(!canLeave()) return;
        await refresh(); const doc=records.find(d=>d.id===button.dataset.open);
        if(!doc) throw Error('Fleta është fshirë.');
        window.FleteDaljeClients?.showEditor();
        current=pad(structuredClone(doc)); baseline=snapshot(); renderEditor(); renderArchive(); return;
      }
      if(button.dataset.delete) {
        const doc=records.find(d=>d.id===button.dataset.delete);
        if(!confirm(`Të fshihet fleta nr. ${doc.number}?`)) return;
        if(current?.id===doc.id && !canLeave()) return;
        await request(`/${encodeURIComponent(doc.id)}?version=${doc.version}`,{method:'DELETE'});
        await refresh(); if(current?.id===doc.id) fresh(); status('Fleta u fshi.'); return;
      }
      switch(button.dataset.fd) {
        case 'refresh': await refresh(); if(!current) fresh(); status('Historiku u rifreskua.'); break;
        case 'new': if(canLeave()) { await refresh(); fresh(); } break;
        case 'history': $('fdArchive').scrollIntoView({behavior:'smooth'}); $('fdSearch').focus(); break;
        case 'add': if(current.lines.length>=500) throw Error('Lejohen deri në 500 rreshta.'); current.lines.push(blankLine()); renderEditor(); break;
        case 'reset': if(!current.lines.slice(1).some(hasData) || confirm('Rreshtat shtesë kanë të dhëna. T’i heq?')) { current.lines=current.lines.slice(0,1); renderEditor(); } break;
        case 'save': {
          total(current);
          if(!Number.isInteger(current.number) || current.number<1 || !current.date) throw Error('Plotësoni numrin dhe datën.');
          const payload=structuredClone(current);
          payload.recipient=payload.recipient.trim(); payload.address=payload.address.trim(); payload.notes=payload.notes.trim();
          if(!payload.recipient) throw Error('Shkruaj emrin e marrësit / klientit.');
          payload.lines=payload.lines.filter(hasData);
          if(!payload.lines.length) throw Error('Shto të paktën një artikull.');
          if(payload.lines.some(line=>line.description.trim() && (!line.quantity || !line.price)) && !confirm('Disa rreshta nuk kanë sasi ose çmim. Ta ruaj gjithsesi?')) return;
          const saved=await request(current.id?`/${encodeURIComponent(current.id)}`:'',{method:current.id?'PUT':'POST',body:JSON.stringify(payload)});
          lastSaved=structuredClone(saved);
          records=[saved,...records.filter(doc=>doc.id!==saved.id)];
          let refreshed=true;try {await refresh();} catch {refreshed=false;}
          fresh();
          status(`Fleta nr. ${saved.number} për ${saved.recipient} u ruajt. U hap një fletë e re.${refreshed?'':' Historiku nuk u rifreskua; provoni përsëri.'}`);break;
        }
        case 'print': if(!current.id && !dirty() && !current.recipient.trim() && !current.lines.some(hasData) && lastSaved) print(lastSaved);else print(); break;
        case 'backup': await refresh(); download('flete-dalje-backup.json', JSON.stringify({format:'flete-dalje-v1',documents:records,payments,schema:1,exported:new Date().toISOString(),records:records.map(doc=>({num:doc.number,client:doc.recipient,date:doc.date,address:doc.address,notes:doc.notes,lines:doc.lines.filter(hasData).map(line=>({name:line.description,unit:line.unit,qty:line.quantity,price:line.price}))}))},null,2),'application/json'); break;
        case 'import': $('fdImport').click(); break;
        case 'csv': {
          await refresh();
          const safe = v => { let s=String(v??''); if(/^[\s]*[=+@-]/.test(s)) s="'"+s; return '"'+s.replace(/"/g,'""')+'"'; };
          const rows=[['Nr. flete','Data','Klienti','Adresa','Pozicioni','Emertimi','Njesia','Sasia','Cmimi','Vlefta','Shenime']];
          [...records].sort((a,b)=>a.number-b.number).forEach(doc=>doc.lines.filter(hasData).forEach((line,i)=>rows.push([doc.number,doc.date,doc.recipient,doc.address,i+1,line.description,line.unit,number(line.quantity),number(line.price),lineValue(line),doc.notes])));
          download('flete-dalje.csv','\uFEFF'+rows.map(row=>row.map(safe).join(';')).join('\r\n'),'text/csv;charset=utf-8'); break;
        }
      }
    } catch(error) { status(`Gabim: ${error.message}`); }
    finally { busy=false; $('fdEditor').inert=false; }
  });
  $('fdImport').addEventListener('change',async event=>{
    const file=event.target.files[0]; if(!file || busy) return;
    busy=true; $('fdEditor').inert=true;
    try {
      if(file.size>20000000) throw Error('Backup është tepër i madh (maksimumi 20 MB).');
      const backup=JSON.parse(await file.text());
      if(!confirm('Të importohen fletët dhe pagesat? Regjistrimet ekzistuese nuk mbishkruhen; konfliktet e ndalojnë të gjithë importin.')) return;
      const data=await request('/import',{method:'POST',body:JSON.stringify(backup)}); receive(data); status('Backup u importua.');
    } catch(error) { status(`Importi dështoi: ${error.message}`); } finally { event.target.value=''; busy=false; $('fdEditor').inert=false; }
  });
  window.addEventListener('beforeunload',event=>{if(dirty()){event.preventDefault();event.returnValue='';}});
  window.FleteDalje={load,canLeave,refresh,request,getArchive:()=>archiveData};
  if (!root.hidden) load();
})();
