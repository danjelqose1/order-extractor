/* Invoice register and document editing; persistence stays with /api/invoices. */
(() => {
  const $ = id => document.getElementById(id);
  const esc = escapeHtml;
  const clone = value => JSON.parse(JSON.stringify(value));
  const round = value => Math.round((value + Number.EPSILON) * 100) / 100;
  const today = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  const fields = [...document.querySelectorAll('[data-invoice-field]')];
  const state = { sources: [], loading: false, creating: false, returnFocus: null, request: 0 };
  const amount = value => `${formatMoney(value || 0)} ALL`;
  const documentData = job => job.document || {};
  const clientName = job => documentData(job).clientName || formatInvoiceClientLabel(job);
  const issueDate = job => documentData(job).issueDate || String(job.createdAt || '').slice(0, 10);
  const dateLabel = value => value ? new Date(`${value}T12:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : '—';

  async function ready() {
    ensureInvoicesLoaded();
    await ensureInvoiceJobsLoaded();
    if (invoiceRuntime.configPromise) await invoiceRuntime.configPromise;
  }

  async function load() {
    $('invoiceNew').disabled = true;
    try {
      await ready();
      setInvoiceStatus('');
      updateInvoicesUI();
    } catch {
      setInvoiceStatus('Could not load invoices. Check your connection, then select Refresh.');
      $('invoiceJobsWrap').innerHTML = '<div class="invoice-empty-state"><h4>Invoices are unavailable</h4><p>Your saved invoices could not be loaded. Refresh to try again.</p><button class="btn" data-register-action="retry">Try again</button></div>';
    } finally {
      $('invoiceNew').disabled = !invoiceRuntime.jobsLoaded;
    }
  }

  function renderSummary() {
    const jobs = getVisibleInvoiceJobs();
    const drafts = jobs.filter(job => job.status !== 'finalized');
    const finalized = jobs.filter(job => job.status === 'finalized');
    const total = list => list.reduce((sum, job) => sum + Number(job.calculated?.total ?? job.totalAmount ?? 0), 0);
    const cards = [
      ['Total invoices', jobs.length, 'All saved documents'],
      ['Drafts', drafts.length, 'Ready for your review'],
      ['Draft value', amount(total(drafts)), 'Not yet finalized'],
      ['Finalized value', amount(total(finalized)), `${finalized.length} finalized document${finalized.length === 1 ? '' : 's'}`],
    ];
    $('invoiceSummary').innerHTML = cards.map(([label, value, detail], i) => `<div class="invoice-metric ${i === 3 ? 'invoice-metric-accent' : ''}"><span>${label}</span><strong>${esc(value)}</strong><small>${detail}</small></div>`).join('');
  }

  function renderRegister() {
    const jobs = applyInvoiceFilters(appState.invoices.jobs || []);
    const pagination = ensureInvoicePaginationState();
    pagination.pageSize = 10;
    pagination.totalPages = Math.max(1, Math.ceil(jobs.length / pagination.pageSize));
    pagination.currentPage = Math.min(pagination.currentPage, pagination.totalPages);
    if (!jobs.length) {
      const filtered = Object.values(appState.invoices.filters).some(Boolean);
      $('invoiceJobsWrap').innerHTML = `<div class="invoice-empty-state"><div class="invoice-choice-icon" aria-hidden="true">≡</div><h4>${filtered ? 'No matching invoices' : 'Your next invoice starts here'}</h4><p>${filtered ? 'Try a different client, number, or status.' : 'Create a polished invoice from an order, or start with a blank document and add your own items.'}</p><button class="btn ${filtered ? '' : 'primary'}" data-register-action="${filtered ? 'clear' : 'new'}">${filtered ? 'Clear filters' : '+ Create your first invoice'}</button></div>`;
      updateInvoicePaginationControls(0, 1, 1);
      return;
    }
    const start = (pagination.currentPage - 1) * pagination.pageSize;
    $('invoiceJobsWrap').innerHTML = `<table><thead><tr><th>Invoice / client</th><th>Reference</th><th>Issue date</th><th>Status</th><th class="invoice-amount">Amount</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>${jobs.slice(start, start + pagination.pageSize).map(job => {
      const locked = job.status === 'finalized';
      const reference = job.source === 'invoice' ? 'Standalone invoice' : (job.orderNumbers || []).join(', ');
      return `<tr><td><strong>${esc(job.invoiceNumber || 'Unnumbered draft')}</strong><small>${esc(clientName(job))}</small></td><td>${esc(reference || '—')}<small>${job.source === 'manual' ? 'Manual order' : job.source === 'invoice' ? 'Custom items' : 'Order invoice'}</small></td><td>${esc(dateLabel(issueDate(job)))}</td><td><span class="pill invoice-status-badge ${locked ? 'finalized' : 'draft'}">${locked ? 'Finalized' : 'Draft'}</span></td><td class="invoice-amount"><strong>${esc(amount(job.calculated?.total ?? job.totalAmount))}</strong>${job.calculated?.missing ? '<small>Pricing incomplete</small>' : ''}</td><td><div class="invoice-row-actions"><button class="btn small" data-register-action="open" data-id="${esc(job.id)}">Open</button><button class="btn small tertiary" data-register-action="pdf" data-id="${esc(job.id)}" aria-label="Download PDF for ${esc(job.invoiceNumber || clientName(job))}">PDF</button><button class="btn small tertiary" data-register-action="duplicate" data-id="${esc(job.id)}" aria-label="Duplicate ${esc(job.invoiceNumber || clientName(job))}">Copy</button>${locked ? '' : `<button class="btn small tertiary" data-register-action="delete" data-id="${esc(job.id)}" aria-label="Delete draft ${esc(job.invoiceNumber || clientName(job))}">×</button>`}</div></td></tr>`;
    }).join('')}</tbody></table>`;
    $('invoicePageInfo').textContent = `${start + 1}–${Math.min(start + pagination.pageSize, jobs.length)} of ${jobs.length} invoices`;
    updateInvoicePaginationControls(jobs.length, pagination.totalPages, pagination.currentPage);
  }

  function calculate(job) {
    const lines = (job.items || []).map((item, index) => {
      const quantity = Number(item.quantity);
      const rate = Number(item.rate);
      const valid = item.rate !== null && item.quantity !== null && String(item.description || '').trim() && Number.isFinite(quantity) && quantity > 0 && Number.isFinite(rate) && rate >= 0 && Number.isSafeInteger(Math.round(quantity * rate * 100));
      return { index: index + 1, type: String(item.description || ''), quantity: Number.isFinite(quantity) ? quantity : 0,
        area: 0, unitPricePerM2: Number.isFinite(rate) ? rate : 0, lineTotal: valid ? round(quantity * rate) : 0,
        missing: valid ? null : { reason: 'item' } };
    });
    finalizeInvoiceTotals(job, lines);
    if (!lines.length) job.calculated.missing = { reason: 'item' };
    for (const key of ['subtotal', 'discountValue', 'vatValue', 'total']) job.calculated[key] = round(job.calculated[key]);
    applyJobTotals(job, job.calculated);
    return job.calculated;
  }

  function defaults(job) {
    if (!job.document) {
      const recent = appState.invoices.jobs.find(other => other.id !== job.id && other.document?.businessName)?.document || {};
      job.document = { businessName: recent.businessName || '', businessDetails: recent.businessDetails || '', businessTaxId: recent.businessTaxId || '', clientName: formatInvoiceClientLabel(job) === '—' ? '' : formatInvoiceClientLabel(job), clientDetails: '', clientTaxId: '', issueDate: job.invoiceNumber ? String(job.createdAt || today()).slice(0, 10) : today(), dueDate: '', notes: '' };
    }
    return job.document;
  }

  function renderEditor({ renderItems = true } = {}) {
    const job = getActiveInvoiceJob();
    if (!job) return;
    const doc = defaults(job);
    const locked = job.status === 'finalized';
    $('manualInvoiceTitle').textContent = job.invoiceNumber || 'New invoice';
    fields.forEach(input => {
      input.value = input.dataset.invoiceField === 'invoiceNumber' ? job.invoiceNumber || '' : doc[input.dataset.invoiceField] || '';
      input.disabled = locked;
    });
    $('invoiceDueDate').min = doc.issueDate || '';
    $('invoiceStatusSelect').value = job.status;
    $('invoiceDiscountMode').disabled = locked;
    $('invoiceDiscountValue').disabled = locked;
    $('invoiceDiscountValue').max = job.discountMode === 'absolute' ? '' : '100';
    $('invoiceVatValue').disabled = locked;
    $('invoiceVatValue').max = '100';
    $('invoiceRecalc').disabled = locked;
    $('invoiceSaveDocument').disabled = locked;
    $('invoiceAddLine').hidden = job.source !== 'invoice' || locked;
    $('invoicePricesCard').hidden = job.source === 'invoice' || locked;
    if (job.source === 'invoice' && renderItems) {
      $('invoiceLinesWrap').innerHTML = `<table class="invoice-custom-lines"><thead><tr><th>Description</th><th>Quantity</th><th>Unit price (ALL)</th><th>Total (ALL)</th><th></th></tr></thead><tbody>${(job.items || []).map((item, i) => `<tr><td><input aria-label="Item ${i + 1} description" data-item-index="${i}" data-item-field="description" type="text" maxlength="500" placeholder="Product or service" value="${esc(item.description)}" ${locked ? 'disabled' : ''} /></td><td><input aria-label="Item ${i + 1} quantity" data-item-index="${i}" data-item-field="quantity" type="number" min="0.001" step="0.001" value="${esc(item.quantity)}" ${locked ? 'disabled' : ''} /></td><td><input aria-label="Item ${i + 1} unit price" data-item-index="${i}" data-item-field="rate" type="number" min="0" step="0.01" value="${esc(item.rate)}" ${locked ? 'disabled' : ''} /></td><td class="invoice-item-total">${formatMoney(job.calculated?.lines[i]?.lineTotal || 0)}</td><td>${locked ? '' : `<button type="button" class="btn small tertiary" data-remove-item="${i}" aria-label="Remove item ${i + 1}">×</button>`}</td></tr>`).join('')}</tbody></table>`;
      $('invoiceMissingNotice').hidden = !job.calculated?.missing;
      $('invoiceMissingNotice').textContent = 'Add a description, a quantity greater than zero, and a non-negative price for each item.';
    } else {
      $('invoiceLinesWrap').querySelectorAll('input,select').forEach(input => { input.disabled = locked; });
    }
    renderTotals(job);
    if (locked) $('invoiceDetailMeta').insertAdjacentHTML('beforeend', '<div class="invoice-document-locked">Finalized · Amounts are locked. Change status to Draft to edit.</div>');
  }

  function renderTotals(job) {
    const calc = job.calculated;
    if (calc) $('invoiceTotals').innerHTML = `<dl><div><dt>Subtotal</dt><dd>${amount(calc.subtotal)}</dd></div><div><dt>Discount${job.discountMode === 'percent' ? ` (${Number(job.discountValue || 0)}%)` : ''}</dt><dd>− ${amount(calc.discountValue)}</dd></div><div><dt>VAT (${Number(job.vatRate || 0)}%)</dt><dd>${amount(calc.vatValue)}</dd></div><div class="invoice-grand-total"><dt>Total</dt><dd>${amount(calc.total)}</dd></div></dl>`;
    $('invoiceGeneratePdf').disabled = !calc || !!calc.missing;
    if (calc?.missing?.reason === 'legacy_snapshot') {
      $('invoiceMissingNotice').hidden = false;
      $('invoiceMissingNotice').textContent = 'This older finalized invoice has no saved price breakdown. Its recorded total is preserved. Reopen it as a draft to review current prices and export a new PDF.';
      $('invoiceTotals').innerHTML = `<dl><div class="invoice-grand-total"><dt>Recorded total</dt><dd>${amount(calc.total)}</dd></div></dl>`;
    } else if (job.source === 'invoice') {
      $('invoiceMissingNotice').hidden = !calc?.missing;
      $('invoiceMissingNotice').textContent = 'Add a description, a quantity greater than zero, and a non-negative price for each item.';
    }
  }

  function collectDocument(job) {
    if (!job || job.status === 'finalized') return;
    const doc = defaults(job);
    fields.forEach(input => {
      const key = input.dataset.invoiceField;
      if (key === 'invoiceNumber') job.invoiceNumber = input.value.trim() || null;
      else doc[key] = input.value.trim();
    });
    if (doc.clientName) { job.client = doc.clientName; job.clients = [doc.clientName]; }
  }

  function validate(job, complete = false) {
    const doc = defaults(job);
    const errors = [];
    if (job.invoiceNumber && appState.invoices.jobs.some(other => other.id !== job.id && other.invoiceNumber?.toLowerCase() === job.invoiceNumber.toLowerCase())) errors.push('This invoice number is already in use. Choose a different number.');
    if (doc.dueDate && doc.issueDate && doc.dueDate < doc.issueDate) errors.push('The due date cannot be before the issue date.');
    if (!Number.isFinite(Number(job.discountValue)) || Number(job.discountValue) < 0 || (job.discountMode !== 'absolute' && Number(job.discountValue) > 100)) errors.push('Enter a valid discount (0–100 for a percentage).');
    if (!Number.isFinite(Number(job.vatRate)) || Number(job.vatRate) < 0 || Number(job.vatRate) > 100) errors.push('VAT must be between 0 and 100%.');
    if (complete) {
      if (!doc.businessName) errors.push('Add your business name.');
      if (!doc.clientName) errors.push('Add a client name.');
      if (!doc.issueDate) errors.push('Choose an issue date.');
      const calc = job.calculated;
      if (!calc?.lines?.length || calc.missing || calc.lines.some(line => line.pricingUnderstanding && !line.pricingUnderstanding.safeToPrice)) errors.push('Complete all items and resolve missing or uncertain prices.');
    }
    if (errors.length) throw new Error(errors.join(' '));
  }

  async function persist(job, { complete = false } = {}) {
    if (!invoiceRuntime.jobsLoaded) throw new Error('Load your existing invoices before saving.');
    validate(job, complete);
    if (!job.invoiceNumber) job.invoiceNumber = nextInvoiceNumber();
    job.updatedAt = new Date().toISOString();
    if (!await saveInvoiceJobs()) throw new Error('Could not save this invoice. Your changes remain in this session; use Save invoice to retry.');
  }

  async function saveDocument() {
    const job = getActiveInvoiceJob();
    if (!job || job.status === 'finalized') return;
    collectDocument(job);
    await recalcInvoiceJob(job, { allowPrompt: false, allowAi: false });
    await persist(job);
    updateInvoicesUI();
    setInvoiceStatus('Invoice saved.');
  }

  async function changeStatus(value) {
    const job = getActiveInvoiceJob();
    if (!job) return;
    const previous = clone(job);
    try {
      if (value === 'finalized') {
        collectDocument(job);
        await recalcInvoiceJob(job, { allowPrompt: false, allowAi: false });
        validate(job, true);
        job.finalizedSnapshot = clone(job.calculated);
        job.finalizedAt = new Date().toISOString();
        job.status = 'finalized';
      } else {
        if (!confirm('Reopen this finalized invoice as a draft? Recalculating may apply current prices.')) { renderEditor(); return; }
        job.status = 'draft';
        delete job.finalizedSnapshot;
        delete job.finalizedAt;
      }
      await persist(job);
      updateInvoicesUI();
      setInvoiceStatus(value === 'finalized' ? 'Invoice finalized. Its amounts are now locked.' : 'Invoice reopened as a draft.');
    } catch (error) {
      Object.keys(job).forEach(key => delete job[key]);
      Object.assign(job, previous);
      updateInvoicesUI();
      setInvoiceStatus(error.message);
    }
  }

  function open(job) {
    appState.invoices.activeJobId = job.id;
    defaults(job);
    updateInvoicesUI();
    openManualInvoiceModal();
    setInvoiceStatus('');
  }

  async function createBlank() {
    await ready();
    const job = normalizeInvoiceJob({ id: `inv-${crypto.randomUUID()}`, source: 'invoice', orderNumbers: [], orders: [], client: '', clients: [], createdAt: new Date().toISOString(), items: [{ description: '', quantity: 1, rate: 0 }], status: 'draft', rows: [], rawRows: [] });
    defaults(job);
    calculate(job);
    appState.invoices.jobs.unshift(job);
    appState.invoices.filters = { client: '', order: '', status: '' };
    let status = 'Draft created. Add your details and items, then save your invoice.';
    try { await persist(job); } catch (error) { status = error.message; }
    $('invoiceCreateModal').hidden = true;
    open(job);
    setInvoiceStatus(status);
    $('invoiceBusinessName').focus();
  }

  async function fetchSources(source) {
    const results = [];
    for (let offset = 0; ; offset += 200) {
      const data = source === 'manual' ? await manualApi(`/manual-orders?limit=200&offset=${offset}`) : await fetchOrders({ limit: 200, offset, year: 'all' });
      const items = data?.items || [];
      results.push(...items.filter(item => source === 'manual' ? ['approved', 'processing', 'finished'].includes(item.status) : canInvoiceStatus(item.status)).map(item => ({ ...item, invoiceSource: source })));
      if (!data?.has_more || !items.length) break;
    }
    return results;
  }

  function existingFor(order) {
    const source = order.invoiceSource;
    const numbers = source === 'manual' ? [order.order_number] : parseInvoiceOrderNumbers(order);
    return appState.invoices.jobs.find(job => !job.hidden && job.source === source && (job.orderNumbers || []).some(num => numbers.some(value => String(value).toLowerCase() === String(num).toLowerCase())));
  }

  function renderSources() {
    const query = $('invoiceSourceSearch').value.trim().toLowerCase();
    const list = state.sources.filter(order => `${order.client_name || order.client || order.client_hint || ''} ${order.order_number || ''} ${(order.order_numbers || []).join(', ')}`.toLowerCase().includes(query));
    $('invoiceSourceList').innerHTML = list.length ? list.map(order => {
      const index = state.sources.indexOf(order);
      const name = order.client_name || order.client || order.client_hint || 'Unnamed client';
      const label = order.invoiceSource === 'manual' ? order.order_number : parseInvoiceOrderNumbers(order).join(', ');
      return `<button type="button" class="invoice-source-choice" data-source-index="${index}" ${state.creating ? 'disabled' : ''}><span><strong>${esc(name)}</strong><small>${esc(label || 'Order')} · ${order.invoiceSource === 'manual' ? 'Manual order' : 'Order'}</small></span><span>${existingFor(order) ? 'Open invoice' : 'Create draft'} →</span></button>`;
    }).join('') : `<div class="invoice-empty">${state.loading ? 'Loading eligible orders…' : query ? 'No orders match your search.' : 'No eligible orders yet. Approve an order, or start a blank invoice above.'}</div>`;
  }

  async function showCreate() {
    state.returnFocus = document.activeElement;
    $('invoiceCreateModal').hidden = false;
    $('invoiceSourceSearch').value = '';
    $('invoiceCreateBlank').disabled = true;
    $('invoiceCreateStatus').textContent = 'Loading saved invoices and orders…';
    state.loading = true;
    state.sources = [];
    renderSources();
    $('invoiceCreateClose').focus();
    const request = ++state.request;
    try {
      await ready();
      $('invoiceCreateBlank').disabled = false;
      const results = await Promise.allSettled([fetchSources('pdf'), fetchSources('manual')]);
      if (request !== state.request) return;
      state.sources = results.flatMap(result => result.status === 'fulfilled' ? result.value : []);
      $('invoiceCreateStatus').textContent = results.some(result => result.status === 'rejected') ? 'Some orders could not be loaded. You can start a blank invoice or close and retry.' : '';
    } catch {
      $('invoiceCreateStatus').textContent = 'Could not load saved invoices. Close this window and select Refresh to retry.';
    } finally { state.loading = false; renderSources(); }
  }

  async function createFromOrder(index) {
    const source = state.sources[index];
    if (!source || state.creating) return;
    const existing = existingFor(source);
    if (existing) { $('invoiceCreateModal').hidden = true; open(existing); return; }
    state.creating = true;
    renderSources();
    try {
      $('invoiceCreateStatus').textContent = 'Preparing your invoice…';
      const order = source.invoiceSource === 'manual' ? manualOrderToShared(await manualApi(`/manual-orders/${source.id}`), { perUnitArea: true }) : await fetchOrder(source.id);
      const job = await createInvoiceJobFromOrder(order, { allowAi: false, allowPrompt: false });
      if (!job) throw new Error('This order has no billable items.');
      defaults(job);
      appState.invoices.filters = { client: '', order: '', status: '' };
      await persist(job);
      $('invoiceCreateModal').hidden = true;
      open(job);
      setInvoiceStatus('Draft created from order. Review your details and prices before finalizing.');
    } catch (error) { $('invoiceCreateStatus').textContent = error.message; setInvoiceStatus(error.message); }
    finally { state.creating = false; renderSources(); }
  }

  async function duplicate(job) {
    // A copy is independent of the order, so it cannot hide or replace its original invoice.
    const copy = normalizeInvoiceJob({ ...clone(job), id: `inv-${crypto.randomUUID()}`, source: 'invoice', status: 'draft', invoiceNumber: null, createdAt: new Date().toISOString(), finalizedSnapshot: null, finalizedAt: null, orderNumbers: [], orders: [], rows: [], rawRows: [], items: (job.calculated?.lines || []).map(line => ({ description: `${line.type || ''}${job.source === 'invoice' ? '' : ' (m²)'}`, quantity: job.source === 'invoice' ? line.quantity : line.area, rate: line.unitPricePerM2, })) });
    copy.document = { ...defaults(job), issueDate: today(), dueDate: '' };
    calculate(copy);
    appState.invoices.jobs.unshift(copy);
    appState.invoices.filters = { client: '', order: '', status: '' };
    let status = 'Copied to a new draft. Review the details before use.';
    try { await persist(copy); } catch (error) { status = error.message; }
    open(copy);
    setInvoiceStatus(status);
  }

  async function prepareExport(job) {
    if (job.id === appState.invoices.activeJobId && !$('manualInvoiceModal').hidden) collectDocument(job);
    await recalcInvoiceJob(job, { allowPrompt: false, allowAi: false });
    await persist(job, { complete: true });
  }

  function guard(fn) {
    return async event => {
      const button = event?.target?.closest?.('button') || null;
      if (button) button.disabled = true;
      try { await fn(event); } catch (error) { setInvoiceStatus(error.message || 'Something went wrong. Please try again.'); }
      finally { if (button) button.disabled = false; }
    };
  }

  $('invoiceNew').addEventListener('click', guard(showCreate));
  $('invoiceRefresh').addEventListener('click', guard(async () => {
    if (invoiceRuntime.unsaved && !confirm('Some invoice changes have not been saved. Refreshing will discard them. Continue?')) return;
    invoiceRuntime.jobsLoaded = false;
    await load();
  }));
  const clear = () => setInvoiceFilters({ client: '', order: '', status: '' });
  $('invoiceClearFilters').addEventListener('click', clear);
  $('invoiceJobsWrap').addEventListener('click', guard(async event => {
    const button = event.target.closest('[data-register-action]');
    if (!button) return;
    const action = button.dataset.registerAction;
    if (action === 'new') return showCreate();
    if (action === 'retry') return load();
    if (action === 'clear') return clear();
    const job = appState.invoices.jobs.find(item => item.id === button.dataset.id);
    if (!job) return;
    if (action === 'open') open(job);
    if (action === 'pdf') { await generateInvoicePdf(job); setInvoiceStatus('Invoice PDF downloaded.'); }
    if (action === 'duplicate') await duplicate(job);
    if (action === 'delete' && job.status !== 'finalized' && confirm(`Delete draft ${job.invoiceNumber || clientName(job)}?`)) {
      const previous = appState.invoices.jobs.slice();
      appState.invoices.jobs = previous.filter(item => item.id !== job.id);
      if (!await saveInvoiceJobs()) { appState.invoices.jobs = previous; throw new Error('Could not delete the draft. Please retry.'); }
      updateInvoicesUI(); setInvoiceStatus('Draft deleted.');
    }
  }));
  $('invoiceCreateBlank').addEventListener('click', guard(createBlank));
  const closeCreate = () => { $('invoiceCreateModal').hidden = true; state.returnFocus?.focus(); };
  $('invoiceCreateClose').addEventListener('click', closeCreate);
  $('invoiceCreateModal').addEventListener('click', event => { if (event.target === $('invoiceCreateModal')) closeCreate(); });
  $('invoiceSourceSearch').addEventListener('input', renderSources);
  $('invoiceSourceList').addEventListener('click', guard(event => {
    const button = event.target.closest('[data-source-index]');
    if (button) return createFromOrder(Number(button.dataset.sourceIndex));
  }));
  $('invoiceSaveDocument').addEventListener('click', guard(saveDocument));
  $('invoiceDocumentFields').addEventListener('submit', event => event.preventDefault());
  fields.forEach(input => input.addEventListener('input', () => {
    const job = getActiveInvoiceJob();
    if (!job || job.status === 'finalized') return;
    collectDocument(job);
    invoiceRuntime.unsaved = true;
    setInvoiceStatus('Unsaved changes · Select Save invoice when ready.');
  }));
  for (const id of ['invoiceDiscountMode', 'invoiceDiscountValue', 'invoiceVatValue']) {
    $(id).addEventListener(id === 'invoiceDiscountMode' ? 'change' : 'input', () => {
      const job = getActiveInvoiceJob();
      if (!job || job.status === 'finalized') return;
      job.discountMode = $('invoiceDiscountMode').value;
      job.discountValue = Number($('invoiceDiscountValue').value);
      job.vatRate = Number($('invoiceVatValue').value);
      if (job.source === 'invoice') calculate(job);
      else if (job.calculated) finalizeInvoiceTotals(job, job.calculated.lines);
      invoiceRuntime.unsaved = true;
      renderTotals(job); renderSummary();
      setInvoiceStatus('Unsaved changes · Select Save invoice when ready.');
    });
  }
  $('invoiceAddLine').addEventListener('click', () => {
    const job = getActiveInvoiceJob();
    if (!job || job.source !== 'invoice' || job.status === 'finalized') return;
    job.items.push({ description: '', quantity: 1, rate: 0 }); calculate(job); updateInvoicesUI();
    invoiceRuntime.unsaved = true;
    $('invoiceLinesWrap').querySelector(`[data-item-index="${job.items.length - 1}"]`)?.focus();
  });
  $('invoiceLinesWrap').addEventListener('input', event => {
    const input = event.target.closest('[data-item-field]');
    const job = getActiveInvoiceJob();
    if (!input || !job || job.source !== 'invoice' || job.status === 'finalized') return;
    const value = input.dataset.itemField === 'description' ? input.value : input.value === '' ? null : Number(input.value);
    job.items[Number(input.dataset.itemIndex)][input.dataset.itemField] = value;
    calculate(job); invoiceRuntime.unsaved = true;
    // Update amounts without replacing the input that the user is tabbing from.
    input.closest('tr').querySelector('.invoice-item-total').textContent = formatMoney(job.calculated.lines[Number(input.dataset.itemIndex)].lineTotal);
    renderSummary();
    renderTotals(job);
    setInvoiceStatus('Unsaved changes · Select Save invoice when ready.');
  });
  $('invoiceLinesWrap').addEventListener('click', event => {
    const button = event.target.closest('[data-remove-item]');
    const job = getActiveInvoiceJob();
    if (!button || !job || job.status === 'finalized') return;
    job.items.splice(Number(button.dataset.removeItem), 1); calculate(job); updateInvoicesUI(); invoiceRuntime.unsaved = true;
  });
  document.addEventListener('keydown', event => {
    const modal = !$('invoiceCreateModal').hidden ? $('invoiceCreateModal') : !$('manualInvoiceModal').hidden && $('invoicePrompt').hidden ? $('manualInvoiceModal') : null;
    if (!modal) return;
    if (event.key === 'Escape' && modal === $('invoiceCreateModal')) { closeCreate(); event.preventDefault(); }
    if (event.key !== 'Tab') return;
    const focusable = [...modal.querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),summary')].filter(node => node.getClientRects().length);
    const first = focusable[0], last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) { last?.focus(); event.preventDefault(); }
    else if (!event.shiftKey && document.activeElement === last) { first?.focus(); event.preventDefault(); }
  });
  window.addEventListener('beforeunload', event => { if (invoiceRuntime.unsaved) { event.preventDefault(); event.returnValue = ''; } });
  window.InvoiceWorkspace = { load, calculate, renderSummary, renderRegister, renderEditor, changeStatus, prepareExport,
    beforeOpen() { state.returnFocus = document.activeElement; renderEditor(); },
    afterClose() { state.returnFocus?.focus(); }, validate, documentData };
})();
