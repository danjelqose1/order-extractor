/* The Bridge consumes prepared Mother Sheet lines. It never rounds, groups or expands them. */
(function(root){
  "use strict";
  const copy = value => JSON.parse(JSON.stringify(value));
  const present = value => value != null && value !== false && value !== "" &&
    !(Array.isArray(value) && !value.length);
  const integer = (value, max) =>
    (typeof value === "number" || (typeof value === "string" && /^[0-9]+$/.test(value))) &&
    Number.isInteger(Number(value)) && Number(value) >= 1 && Number(value) <= max;
  const sourceKey = origin => {
    const source = origin.bridgeSource;
    return source ? JSON.stringify([source.source, source.orderId, source.rowId]) : null;
  };
  const describe = row => row.sources.map(s => `${s.orderId || "Unknown order"} / position ${s.position || "—"}`).join("; ") || "Unknown order / position";
  // References and filenames are metadata, not machining instructions. Recognize
  // explicit unsupported features in free text; structured requirements still block.
  const unsupportedNote = value => typeof value === "string" &&
    /\b(?:holes?|drill(?:ed|ing)?|cut[ -]?outs?|notch(?:es|ed)?|triang(?:le|ular)|trapez(?:oid|ium|oidal)|oval|rhomboid|polygon|arched|circular)\b/i.test(value);
  function collect(processing, busy = false){
    if (busy || processing?.loading || processing?.recalculating) throw new Error("Processing is busy. Wait until preparation finishes, then reopen this review.");
    return (processing?.preview?.groups || []).map(group => ({
      key: group.raw,
      label: group.display || group.raw,
      sourceArea: group.areaValue ?? null,
      rows: (group.lines || []).map(line => {
        const sources = copy(line.originRows || []);
        const sourceIds = sources.map(sourceKey).filter(Boolean).sort();
        const row = {
          id: JSON.stringify(sourceIds), section: group.raw, sectionLabel: group.display || group.raw,
          quantity: line.qty, width: line.width, height: line.height, invalid: !!line.invalid,
          sources, sourceIds, sourceArea: line.area ?? null,
        };
        row.version = JSON.stringify(row);
        return row;
      }),
    }));
  }
  const manualEligible = order => ["approved", "processing"].includes(order?.status);
  function collectManual(orders){
    return orders.map(order => {
      if (!manualEligible(order)) throw new Error(`${order.order_number || order.id}: only approved or processing manual orders can be imported.`);
      if (order.id == null) throw new Error("Manual order identity is missing.");
      if (!order.rows?.length) throw new Error(`${order.order_number || order.id}: manual order has no saved rows.`);
      return { key: `manual-${order.id}`, label: order.order_number, rows: (order.rows || []).map((item,index) => {
        const source = {
          source: "manual", orderId: `manual-${order.id}`, rowId: String(item.id ?? `index:${index}`),
          version: order.version ?? null, updatedAt: order.updated_at ?? null, status: order.status,
          quantity: item.quantity, shape: item.shape ?? item.geometry ?? item.shape_type ?? null,
          rectangular: item.is_rectangular ?? null, requirements: item.special_requirements ?? item.requirements ?? null,
          notes: item.notes ?? null, orderNotes: order.notes ?? null, declaredArea: order.total_area_m2 ?? null,
        };
        const origin = { bridgeSource: source, orderId: order.order_number, client: order.client_name,
          position: item.position || item.client_position || String(item.index_number ?? index + 1),
          section: item.section || "", red_index: item.index_number ?? null };
        const sourceIds = [sourceKey(origin)];
        const row = {
          id: JSON.stringify(sourceIds), section: item.glass_type || "", sectionLabel: item.glass_type || "",
          quantity: item.quantity, width: item.width_mm, height: item.height_mm, invalid: false,
          sources: [origin], sourceIds, sourceArea: item.final_area_m2 ?? null,
        };
        row.version = JSON.stringify(row);
        return copy(row);
      }) };
    });
  }
  function reasons(row){
    const errors = [];
    const report = (message, origin) => errors.push(`${origin ? describe({sources:[origin]}) : describe(row)}: ${message}`);
    for (const [key, max] of [["quantity",999],["width",10000],["height",10000]]){
      if (!integer(row[key], max)) report(`${key} must be a whole number from 1 through ${max}${key === "quantity" ? "" : " mm"}`);
    }
    if (row.invalid) report("Source marks these dimensions invalid");
    if (!row.sources.length || row.sourceIds.length !== row.sources.length) report("source identity is missing; reload this order from its source");
    for (const origin of row.sources){
      const source = origin.bridgeSource;
      if (!source) continue;
      if (!integer(source.quantity, 999)) report("source quantity is malformed or outside 1–999", origin);
      if ((present(source.shape) && !/^(rectangle|rectangular|rect)$/i.test(String(source.shape))) || source.rectangular === false){
        report(`unsupported geometry (${String(source.shape || "non-rectangular")})`, origin);
      }
      if (present(source.requirements)){
        report("special requirements cannot be represented by this CSV; review in the source module", origin);
      }
      for (const [label, note] of [["Row note", source.notes], ["Order note", source.orderNotes]]){
        if (unsupportedNote(note)) report(`${label} mentions unsupported geometry or machining: ${note}`, origin);
      }
    }
    if (/triang|trapez|shaped|sagomat|\b(?:arch|circle|oval|rhomboid|polygon)\b/i.test(row.section)) report("glass section indicates unsupported geometry");
    return errors;
  }
  function validate(rows){
    const errors = rows.flatMap(reasons);
    if (!rows.length) errors.push("Select at least one prepared row.");
    const seen = new Set();
    rows.forEach(row => row.sourceIds.forEach(id => {
      if (seen.has(id)) errors.push(`${describe(row)}: source row occurs more than once`);
      seen.add(id);
    }));
    // Keep provenance on the validated array so preview and CSV share exactly these rows.
    return { errors, rows: errors.length ? [] : rows.map(row => ({ ...row,
      quantity: Number(row.quantity), width: Number(row.width), height: Number(row.height),
    })) };
  }
  const emptyJob = () => ({ rows: [], capturedAt: null });
  function add(job, rows, replace = false){
    const candidate = copy(rows);
    if (!candidate.length) throw new Error("Select at least one prepared row.");
    const previous = replace ? [] : job.rows;
    const combined = [...previous, ...candidate];
    const result = validate(combined);
    if (result.errors.length) throw new Error(result.errors.join("\n"));
    return { ...job, rows: copy(result.rows), capturedAt: new Date().toISOString() };
  }
  // An explicit import replaces the draft with the entire current prepared sheet.
  // Flatten in preview order; never sort or merge across its glass sections.
  function importPrepared(job, sections){
    return add(job, sections.flatMap(section => section.rows), true);
  }
  function changes(job, sections, sourceLabel = "Processing"){
    const current = new Map(sections.flatMap(section => section.rows).map(row => [row.id, row]));
    return job.rows.flatMap(row => {
      const match = current.get(row.id);
      return !match ? [`${describe(row)}: no longer available with the same grouping in ${sourceLabel}.`]
        : match.version !== row.version ? [`${describe(row)}: prepared values, section or source information changed.`] : [];
    });
  }
  function csv(validatedRows){
    const result = validate(validatedRows);
    if (result.errors.length) throw new Error(result.errors.join("\n"));
    return "quantity,width,height\r\n" + validatedRows.map(row => `${row.quantity},${row.width},${row.height}\r\n`).join("");
  }
  const api = { collect, collectManual, manualEligible, reasons, validate, emptyJob, add, importPrepared, changes, csv, integer, describe };
  root.PerfectCutBridge = api;
  if (typeof module !== "undefined") module.exports = api;
  if (typeof document === "undefined" || !document.getElementById("bridgePicker")) return;

  // Deliberately session-only, like the current Processing cart and Labels jobs.
  let job = emptyJob();
  let picker = null;
  let previewRows = [];
  let busy = false;
  let manualSections = [];
  let manualError = "";
  const manualPicker = { selected: new Map(), offset: 0, items: [], hasMore: false, request: 0, loading: false };
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? "—").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
  const current = () => collect(appState.processing, processingBridgeBusy > 0);
  const errorList = errors => errors.length ? `<ul>${errors.map(error => `<li>${esc(error)}</li>`).join("")}</ul>` : "";
  const sourceCells = row => `<td>${esc(row.sources.map(s => s.position || "—").join(", "))}</td><td>${esc([...new Set(row.sources.map(s => s.orderId))].join(", "))}<br><small>${esc([...new Set(row.sources.map(s => s.client))].join(", "))}</small></td>`;
  const numericCells = row => `<td>${esc(row.quantity)}</td><td>${esc(row.width)}</td><td>${esc(row.height)}</td>`;
  const tableHead = first => `<thead><tr><th>${first}</th><th>Source position</th><th>Order / client</th><th>Qty</th><th>Width (mm)</th><th>Height (mm)</th></tr></thead>`;
  function sourceNotes(rows){
    const notes = new Set();
    rows.forEach(row => row.sources.forEach(origin => {
      const source = origin.bridgeSource;
      if (source?.orderNotes?.trim()) notes.add(`${origin.orderId}: ${source.orderNotes}`);
      if (source?.notes?.trim()) notes.add(`${describe({sources:[origin]})}: ${source.notes}`);
    }));
    return notes.size ? `<details class="muted small"><summary>Source notes (not included in CSV)</summary>${errorList([...notes])}</details>` : "";
  }
  function render(){
    const validation = validate(job.rows);
    previewRows = validation.rows;
    let stale = [];
    try{ stale = job.sourceKind === "manual"
      ? (manualError ? [manualError] : changes(job, manualSections, "Manual Orders"))
      : changes(job, current()); }catch(error){ stale = [error.message]; }
    const errors = [...(job.rows.length ? validation.errors : []), ...stale];
    el("bridgeErrors").innerHTML = errorList(errors);
    el("bridgeOpenProcessing").textContent = job.sourceKind === "manual" ? "Open Manual Orders" : "Open Processing";
    el("bridgeAdd").disabled = busy;
    el("bridgeManualAdd").disabled = busy;
    el("bridgeRefresh").textContent = job.sourceKind === "manual" ? "Review / refresh Manual Orders" : "Review / refresh from Processing";
    el("bridgeRefresh").disabled = busy || !job.rows.length;
    el("bridgeClear").disabled = busy || !job.rows.length;
    el("bridgeDownload").disabled = busy || !previewRows.length || errors.length > 0;
    const rows = previewRows.length ? previewRows : job.rows;
    const pieces = previewRows.reduce((sum,row) => sum + row.quantity, 0);
    const area = previewRows.reduce((sum,row) => sum + row.quantity * row.width * row.height, 0) / 1000000;
    el("bridgeSummary").textContent = previewRows.length ? `${previewRows.length} dimension rows · ${pieces} pieces · Calculated cutting area: ${area.toFixed(4)} m²` : "No exportable rows yet.";
    const orderAreas = new Map();
    rows.forEach(row => row.sources.forEach(s => {
      if (s.bridgeSource?.declaredArea != null) orderAreas.set(s.orderId, s.bridgeSource.declaredArea);
    }));
    el("bridgeSourceSummary").innerHTML = rows.length ? `<p class="muted small">Source order areas (whole orders, not selected cutting area): ${orderAreas.size ? [...orderAreas].map(([order,area]) => `${esc(order)}: ${esc(area)} m²`).join("; ") : "not declared"}. Values are preserved separately.</p>` : "<p>Nothing added. Prepare a sheet in Processing, then add it here.</p>";
    el("bridgeSourceSummary").innerHTML += sourceNotes(rows);
    el("bridgeRows").innerHTML = rows.length ? `<table>${tableHead("Action")}<tbody>${rows.map((row,i) => `<tr><td><button class="btn small muted" data-bridge-remove="${i}" ${busy ? "disabled" : ""} aria-label="Remove ${esc(describe(row))}">Remove</button></td>${sourceCells(row)}${numericCells(row)}</tr>`).join("")}</tbody></table>` : "";
  }
  function openPicker(replace = false, suppliedSections = null, manualIds = null){
    let sections;
    try{ sections = suppliedSections || current(); }catch(error){ el("bridgeStatus").textContent = error.message; render(); return; }
    const rows = sections.flatMap(section => section.rows);
    picker = { sections, rows, selected: new Set(rows.map(row => row.id)), replace, manualIds };
    el("bridgePickerDescription").textContent = manualIds
      ? "Saved Manual Orders rows, in their original order and millimetres. No rounding or grouping is applied. Deselect rows for a partial export."
      : "All prepared rows appear in Processing order, across glass types. Grouped rows stay together. Deselect rows only if you want a partial export.";
    el("bridgePickerProcessing").textContent = manualIds ? "Open Manual Orders" : "Open Processing";
    el("bridgePickerTitle").textContent = manualIds ? "Review Manual Orders" : replace ? "Review replacement snapshot" : "Add from Processing";
    el("bridgePickerAdd").textContent = replace ? "Replace job with selected" : "Add selected";
    el("bridgePickerErrors").innerHTML = "";
    renderPicker();
    el("bridgePicker").showModal();
  }
  function renderPicker(){
    let html = "<p>The selected rows will replace the Bridge snapshot in the order shown below.</p>";
    if (!picker.rows.length){
      html += picker.manualIds ? '<p>No saved rows in these manual orders.</p>' : '<p class="processing-empty">Processing is empty. Open Processing to prepare a sheet.</p>';
    }else{
      const buckets = new Map();
      picker.rows.forEach(row => {
        const label = [...new Set(row.sources.map(s => `${s.orderId} — ${s.client}`))].join(" / ");
        if (!buckets.has(label)) buckets.set(label, []);
        buckets.get(label).push(row.id);
      });
      picker.buckets = [...buckets.values()];
      html += '<div class="bridge-actions">' + [...buckets].map(([label,ids],i) =>
        `<label><input type="checkbox" data-bridge-order="${i}" ${ids.every(id => picker.selected.has(id)) ? "checked" : ""}> ${esc(label)}</label>`
      ).join("") + '</div>';
      html += sourceNotes(picker.rows);
      html += '<div class="table-responsive"><table>' + tableHead("Select") + '<tbody>';
      picker.rows.forEach((row,i) => {
        const issues = reasons(row);
        html += `<tr><td><input type="checkbox" data-bridge-select="${i}" aria-label="Select ${esc(describe(row))}" ${picker.selected.has(row.id) ? "checked" : ""}></td>${sourceCells(row)}${numericCells(row)}</tr>`;
        if (issues.length) html += `<tr><td colspan="6" class="bridge-errors">${esc(issues.join("; "))}</td></tr>`;
      });
      html += "</tbody></table></div>";
    }
    el("bridgePickerBody").innerHTML = html;
    updateSelection();
  }
  function selectedRows(){ return picker.rows.filter(row => picker.selected.has(row.id)); }
  function updateSelection(){
    const rows = selectedRows();
    const validQty = rows.every(row => integer(row.quantity,999));
    el("bridgePickerCount").textContent = `${rows.length} selected rows · ${validQty ? rows.reduce((sum,row) => sum + Number(row.quantity),0) : "Invalid"} total quantity`;
    const errors = rows.length ? validate(rows).errors : [];
    el("bridgePickerErrors").innerHTML = errorList(errors);
    el("bridgePickerAdd").disabled = busy || !rows.length || errors.length > 0;
  }
  el("bridgePickerBody").addEventListener("change", event => {
    if (!picker || busy) return;
    const target = event.target;
    if (target.hasAttribute("data-bridge-select")){
      const row = picker.rows[Number(target.dataset.bridgeSelect)];
      if (target.checked) picker.selected.add(row.id); else picker.selected.delete(row.id);
    }else if (target.hasAttribute("data-bridge-order")){
      picker.buckets[Number(target.dataset.bridgeOrder)].forEach(id => {
        if (target.checked) picker.selected.add(id); else picker.selected.delete(id);
      });
    }else return;
    // Re-render order checkboxes without leaving stale selection summaries.
    const focusAttribute = target.hasAttribute("data-bridge-select") ? "data-bridge-select" : target.hasAttribute("data-bridge-order") ? "data-bridge-order" : null;
    const focusValue = focusAttribute ? target.getAttribute(focusAttribute) : null;
    renderPicker();
    if (focusAttribute) el("bridgePickerBody").querySelector(`[${focusAttribute}="${focusValue}"]`)?.focus();
  });
  el("bridgePickerAdd").addEventListener("click", async () => {
    if (!picker || busy) return; // Synchronous guard also covers rapid duplicate clicks.
    const activePicker = picker;
    busy = true; updateSelection(); render();
    el("bridgePickerBody").querySelectorAll("input").forEach(input => { input.disabled = true; });
    try{
      const rows = selectedRows();
      const selectedManualIds = activePicker.manualIds?.filter(id => rows.some(row => row.sources.some(source => source.bridgeSource.orderId === `manual-${id}`))) || null;
      const latest = selectedManualIds ? await fetchManualSections(selectedManualIds) : current();
      if (picker !== activePicker) return;
      const stale = changes({rows},latest, activePicker.manualIds ? "Manual Orders" : "Processing");
      if (stale.length) throw new Error(`${activePicker.manualIds ? "Manual Orders" : "Processing"} changed while this review was open. Cancel and reopen to review current values.\n` + stale.join("\n"));
      job = { ...add(job, rows, true), sourceKind: activePicker.manualIds ? "manual" : "processing", manualIds: selectedManualIds };
      if (activePicker.manualIds){ manualSections = latest; manualError = ""; }
      picker = null;
      el("bridgePicker").close();
      el("bridgeStatus").textContent = activePicker.manualIds ? "Manual Orders rows copied in their saved order. Ready to download." : "Prepared rows copied in Processing order. Ready to download.";
    }catch(error){ if (picker === activePicker) el("bridgePickerErrors").innerHTML = errorList(error.message.split("\n")); }
    finally{
      busy = false; render();
      if (picker === activePicker){
        el("bridgePickerBody").querySelectorAll("input").forEach(input => { input.disabled = false; });
        el("bridgePickerAdd").disabled = !selectedRows().length || validate(selectedRows()).errors.length > 0;
      }
    }
  });
  el("bridgePicker").addEventListener("close", () => { picker = null; });
  el("bridgePickerCancel").addEventListener("click", () => el("bridgePicker").close());
  const openProcessing = () => { if (el("bridgePicker").open) el("bridgePicker").close(); activateTab(job.sourceKind === "manual" ? "manual" : "processing"); };
  el("bridgeOpenProcessing").addEventListener("click",openProcessing);
  el("bridgePickerProcessing").addEventListener("click", () => {
    const manual = !!picker?.manualIds;
    el("bridgePicker").close(); activateTab(manual ? "manual" : "processing");
  });
  el("bridgeAdd").addEventListener("click", () => {
    let sections;
    try{ sections = current(); }catch(error){ el("bridgeStatus").textContent = error.message; render(); return; }
    if (!sections.some(section => section.rows.length)){ openPicker(true); return; }
    try{
      job = { ...importPrepared(job, sections), sourceKind: "processing", manualIds: null };
      el("bridgeStatus").textContent = "Prepared rows copied in Processing order. Ready to download.";
      render();
    }catch(error){
      // Keep every invalid row visible and selected for explicit review/deselection.
      openPicker(true);
    }
  });
  el("bridgeRefresh").addEventListener("click", async () => {
    if (busy) return;
    if (job.sourceKind !== "manual"){ openPicker(true); return; }
    busy = true; render();
    try{ openPicker(true, await fetchManualSections(job.manualIds), job.manualIds); }
    catch(error){ manualError = error.message; el("bridgeStatus").textContent = error.message; }
    finally{ busy = false; render(); if (picker) updateSelection(); }
  });
  el("bridgeClear").addEventListener("click", () => {
    if (!window.confirm("Clear this Bridge job? Processing, Labels and source orders will remain unchanged.")) return;
    job = emptyJob(); el("bridgeStatus").textContent = "Bridge job cleared."; render();
  });
  el("bridgeRows").addEventListener("click", event => {
    const button = event.target.closest("[data-bridge-remove]");
    if (!button) return;
    job.rows.splice(Number(button.dataset.bridgeRemove),1);
    if (job.manualIds) job.manualIds = job.manualIds.filter(id => job.rows.some(row => row.sources.some(source => source.bridgeSource.orderId === `manual-${id}`)));
    render();
  });
  el("bridgeDownload").addEventListener("click", async () => {
    if (busy) return;
    if (job.sourceKind === "manual"){
      busy = true; render();
      try{ manualSections = await fetchManualSections(job.manualIds); manualError = ""; }
      catch(error){ manualError = error.message; }
      finally{ busy = false; }
    }
    render(); // Revalidate source versions and preview immediately before download.
    if (el("bridgeDownload").disabled) return;
    const blob = new Blob([csv(previewRows)], {type:"text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "job.csv";
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url),1000);
    el("bridgeStatus").textContent = "CSV download requested. Check the saved filename and the material selected in Perfect Cut before pressing F8.";
  });
  async function fetchManualSections(ids){
    const orders = await Promise.all(ids.map(async id => {
      const order = await manualApi(`/manual-orders/${encodeURIComponent(id)}`);
      if (!order || String(order.id) !== String(id)) throw new Error(`Manual order ${id}: unexpected or missing response.`);
      return order;
    }));
    return collectManual(orders);
  }
  function renderManualList(){
    const state = manualPicker;
    el("bridgeManualPrevious").disabled = state.loading || state.offset === 0;
    el("bridgeManualNext").disabled = state.loading || !state.hasMore;
    el("bridgeManualReview").disabled = state.loading || !state.selected.size;
    el("bridgeManualStatus").textContent = state.loading ? "Loading Manual Orders…" : `${state.selected.size} selected orders · Page ${state.offset / 50 + 1}`;
    el("bridgeManualList").innerHTML = state.loading ? "" : state.items.length ? `<table><thead><tr><th>Select</th><th>Order / client</th><th>Status</th><th>Rows</th><th>Pieces</th></tr></thead><tbody>${state.items.map((order,index) => `<tr><td><input type="checkbox" data-bridge-manual-order="${index}" aria-label="Select manual order ${esc(order.order_number)}" ${state.selected.has(String(order.id)) ? "checked" : ""} ${manualEligible(order) ? "" : "disabled"}></td><td>${esc(order.order_number)}<br><small>${esc(order.client_name)}</small></td><td>${esc(order.status)}</td><td>${esc(order.row_count)}</td><td>${esc(order.total_quantity)}</td></tr>`).join("")}</tbody></table>` : '<p>No manual orders found. Try a different search, or save an order in Manual Orders.</p>';
  }
  async function loadManualList(){
    const request = ++manualPicker.request;
    manualPicker.loading = true; renderManualList();
    const params = new URLSearchParams({ limit: "50", offset: String(manualPicker.offset), query: el("bridgeManualSearch").value.trim() });
    try{
      const result = await manualApi(`/manual-orders?${params}`);
      if (request !== manualPicker.request) return;
      manualPicker.items = Array.isArray(result?.items) ? result.items : [];
      manualPicker.hasMore = !!result?.has_more;
      // A returned status change invalidates an earlier selection on this page.
      manualPicker.items.forEach(order => { if (!manualEligible(order)) manualPicker.selected.delete(String(order.id)); });
      manualPicker.loading = false; renderManualList();
    }catch(error){
      if (request !== manualPicker.request) return;
      manualPicker.items = []; manualPicker.hasMore = false; manualPicker.loading = false;
      renderManualList(); el("bridgeManualStatus").textContent = error.message;
    }
  }
  el("bridgeManualAdd").addEventListener("click", () => {
    if (busy) return;
    manualPicker.selected.clear(); manualPicker.offset = 0;
    el("bridgeManualPicker").showModal(); loadManualList();
  });
  el("bridgeManualSearchForm").addEventListener("submit", event => {
    event.preventDefault(); if (busy) return;
    manualPicker.offset = 0; loadManualList();
  });
  el("bridgeManualPrevious").addEventListener("click", () => { manualPicker.offset -= 50; loadManualList(); });
  el("bridgeManualNext").addEventListener("click", () => { manualPicker.offset += 50; loadManualList(); });
  el("bridgeManualList").addEventListener("change", event => {
    if (manualPicker.loading || !event.target.hasAttribute("data-bridge-manual-order")) return;
    const order = manualPicker.items[Number(event.target.dataset.bridgeManualOrder)];
    if (!manualEligible(order)) return;
    if (event.target.checked) manualPicker.selected.set(String(order.id),order.order_number);
    else manualPicker.selected.delete(String(order.id));
    el("bridgeManualReview").disabled = !manualPicker.selected.size;
    el("bridgeManualStatus").textContent = `${manualPicker.selected.size} selected orders · Page ${manualPicker.offset / 50 + 1}`;
  });
  el("bridgeManualCancel").addEventListener("click", () => el("bridgeManualPicker").close());
  el("bridgeManualPicker").addEventListener("close", () => { manualPicker.request++; });
  el("bridgeManualReview").addEventListener("click", async () => {
    if (busy || manualPicker.loading || !manualPicker.selected.size) return;
    const request = ++manualPicker.request;
    const ids = [...manualPicker.selected.keys()];
    busy = true; manualPicker.loading = true; render(); renderManualList();
    try{
      const sections = await fetchManualSections(ids);
      if (request !== manualPicker.request || !el("bridgeManualPicker").open) return;
      el("bridgeManualPicker").close(); openPicker(true, sections, ids);
    }catch(error){
      if (request === manualPicker.request){
        manualPicker.loading = false; renderManualList(); el("bridgeManualStatus").textContent = error.message;
      }
    }finally{
      busy = false; manualPicker.loading = false; render(); if (picker) updateSelection();
    }
  });

  root.PerfectCutBridgeUI = { render };
  render();
})(typeof window !== "undefined" ? window : globalThis);
