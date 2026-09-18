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
  function reasons(row){
    const errors = [];
    for (const [key, max] of [["quantity",999],["width",10000],["height",10000]]){
      if (!integer(row[key], max)) errors.push(`${key} must be a whole number from 1 through ${max}${key === "quantity" ? "" : " mm"}`);
    }
    if (row.invalid) errors.push("Processing marks these dimensions invalid");
    if (!row.sources.length || row.sourceIds.length !== row.sources.length) errors.push("source identity is missing; prepare this order again in Processing");
    for (const origin of row.sources){
      const source = origin.bridgeSource;
      if (!source) continue;
      const at = `${origin.orderId} / position ${origin.position}`;
      if (!integer(source.quantity, 999)) errors.push(`${at}: source quantity is malformed or outside 1–999`);
      if ((present(source.shape) && !/^(rectangle|rectangular|rect)$/i.test(String(source.shape))) || source.rectangular === false){
        errors.push(`${at}: unsupported geometry (${String(source.shape || "non-rectangular")})`);
      }
      // Free-form instructions have no equivalent in the three-column CSV. Fail closed.
      if (present(source.requirements) || present(source.notes) || present(source.orderNotes)){
        errors.push(`${at}: notes or special requirements cannot be represented by this CSV; review in Processing`);
      }
    }
    if (/triang|trapez|shaped|sagomat|\b(?:arch|circle|oval|rhomboid|polygon)\b/i.test(row.section)) errors.push("glass section indicates unsupported geometry");
    return errors;
  }
  function validate(rows){
    const errors = rows.flatMap(row => reasons(row).map(reason => `${describe(row)}: ${reason}`));
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
  function changes(job, sections){
    const current = new Map(sections.flatMap(section => section.rows).map(row => [row.id, row]));
    return job.rows.flatMap(row => {
      const match = current.get(row.id);
      return !match ? [`${describe(row)}: no longer available with the same grouping in Processing.`]
        : match.version !== row.version ? [`${describe(row)}: prepared values, section or source information changed.`] : [];
    });
  }
  function csv(validatedRows){
    const result = validate(validatedRows);
    if (result.errors.length) throw new Error(result.errors.join("\n"));
    return "quantity,width,height\r\n" + validatedRows.map(row => `${row.quantity},${row.width},${row.height}\r\n`).join("");
  }
  const api = { collect, reasons, validate, emptyJob, add, importPrepared, changes, csv, integer, describe };
  root.PerfectCutBridge = api;
  if (typeof module !== "undefined") module.exports = api;
  if (typeof document === "undefined" || !document.getElementById("bridgePicker")) return;

  // Deliberately session-only, like the current Processing cart and Labels jobs.
  let job = emptyJob();
  let picker = null;
  let previewRows = [];
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? "—").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
  const current = () => collect(appState.processing, processingBridgeBusy > 0);
  const errorList = errors => errors.length ? `<ul>${errors.map(error => `<li>${esc(error)}</li>`).join("")}</ul>` : "";
  const sourceCells = row => `<td>${esc(row.sources.map(s => s.position || "—").join(", "))}</td><td>${esc([...new Set(row.sources.map(s => s.orderId))].join(", "))}<br><small>${esc([...new Set(row.sources.map(s => s.client))].join(", "))}</small></td>`;
  const numericCells = row => `<td>${esc(row.quantity)}</td><td>${esc(row.width)}</td><td>${esc(row.height)}</td>`;
  const tableHead = first => `<thead><tr><th>${first}</th><th>Source position</th><th>Order / client</th><th>Qty</th><th>Width (mm)</th><th>Height (mm)</th></tr></thead>`;
  function render(){
    const validation = validate(job.rows);
    previewRows = validation.rows;
    let stale = [];
    try{ stale = changes(job, current()); }catch(error){ stale = [error.message]; }
    const errors = [...(job.rows.length ? validation.errors : []), ...stale];
    el("bridgeErrors").innerHTML = errorList(errors);
    el("bridgeRefresh").disabled = !job.rows.length;
    el("bridgeClear").disabled = !job.rows.length;
    el("bridgeDownload").disabled = !previewRows.length || errors.length > 0;
    const rows = previewRows.length ? previewRows : job.rows;
    const pieces = previewRows.reduce((sum,row) => sum + row.quantity, 0);
    const area = previewRows.reduce((sum,row) => sum + row.quantity * row.width * row.height, 0) / 1000000;
    el("bridgeSummary").textContent = previewRows.length ? `${previewRows.length} dimension rows · ${pieces} pieces · Calculated cutting area: ${area.toFixed(4)} m²` : "No exportable rows yet.";
    const orderAreas = new Map();
    rows.forEach(row => row.sources.forEach(s => {
      if (s.bridgeSource?.declaredArea != null) orderAreas.set(s.orderId, s.bridgeSource.declaredArea);
    }));
    el("bridgeSourceSummary").innerHTML = rows.length ? `<p class="muted small">Source order areas (whole orders, not selected cutting area): ${orderAreas.size ? [...orderAreas].map(([order,area]) => `${esc(order)}: ${esc(area)} m²`).join("; ") : "not declared"}. Values are preserved separately.</p>` : "<p>Nothing added. Prepare a sheet in Processing, then add it here.</p>";
    el("bridgeRows").innerHTML = rows.length ? `<table>${tableHead("Action")}<tbody>${rows.map((row,i) => `<tr><td><button class="btn small muted" data-bridge-remove="${i}" aria-label="Remove ${esc(describe(row))}">Remove</button></td>${sourceCells(row)}${numericCells(row)}</tr>`).join("")}</tbody></table>` : "";
  }
  function openPicker(replace = false){
    let sections;
    try{ sections = current(); }catch(error){ el("bridgeStatus").textContent = error.message; render(); return; }
    const rows = sections.flatMap(section => section.rows);
    picker = { sections, rows, selected: new Set(rows.map(row => row.id)), replace };
    el("bridgePickerTitle").textContent = replace ? "Review replacement snapshot" : "Add from Processing";
    el("bridgePickerAdd").textContent = replace ? "Replace job with selected" : "Add selected";
    el("bridgePickerErrors").innerHTML = "";
    renderPicker();
    el("bridgePicker").showModal();
  }
  function renderPicker(){
    let html = "<p>The selected current rows will replace the Bridge snapshot, in Processing order.</p>";
    if (!picker.rows.length){
      html += '<p class="processing-empty">Processing is empty. Open Processing to prepare a sheet.</p>';
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
      html += '<div class="table-responsive"><table>' + tableHead("Select") + '<tbody>';
      picker.rows.forEach((row,i) => {
        const issues = reasons(row);
        html += `<tr><td><input type="checkbox" data-bridge-select="${i}" aria-label="Select ${esc(describe(row))}" ${picker.selected.has(row.id) ? "checked" : ""}></td>${sourceCells(row)}${numericCells(row)}</tr>`;
        if (issues.length) html += `<tr><td colspan="6" class="bridge-errors">${esc(describe(row))}: ${esc(issues.join("; "))}</td></tr>`;
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
    el("bridgePickerAdd").disabled = !rows.length || errors.length > 0;
  }
  el("bridgePickerBody").addEventListener("change", event => {
    if (!picker) return;
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
  el("bridgePickerAdd").addEventListener("click", () => {
    if (!picker) return; // Synchronous guard also covers rapid duplicate clicks.
    try{
      const rows = selectedRows();
      const stale = changes({rows},current());
      if (stale.length) throw new Error("Processing changed while this review was open. Cancel and reopen to review current values.\n" + stale.join("\n"));
      job = add(job, rows, true);
      picker = null;
      el("bridgePicker").close();
      el("bridgeStatus").textContent = "Prepared rows copied in Processing order. Ready to download.";
      render();
    }catch(error){ el("bridgePickerErrors").innerHTML = errorList(error.message.split("\n")); }
  });
  el("bridgePicker").addEventListener("close", () => { picker = null; });
  el("bridgePickerCancel").addEventListener("click", () => el("bridgePicker").close());
  const openProcessing = () => { if (el("bridgePicker").open) el("bridgePicker").close(); activateTab("processing"); };
  el("bridgeOpenProcessing").addEventListener("click",openProcessing);
  el("bridgePickerProcessing").addEventListener("click",openProcessing);
  el("bridgeAdd").addEventListener("click", () => {
    let sections;
    try{ sections = current(); }catch(error){ el("bridgeStatus").textContent = error.message; render(); return; }
    if (!sections.some(section => section.rows.length)){ openPicker(true); return; }
    try{
      job = importPrepared(job, sections);
      el("bridgeStatus").textContent = "Prepared rows copied in Processing order. Ready to download.";
      render();
    }catch(error){
      // Keep every invalid row visible and selected for explicit review/deselection.
      openPicker(true);
    }
  });
  el("bridgeRefresh").addEventListener("click", () => openPicker(true));
  el("bridgeClear").addEventListener("click", () => {
    if (!window.confirm("Clear this Bridge job? Processing, Labels and source orders will remain unchanged.")) return;
    job = emptyJob(); el("bridgeStatus").textContent = "Bridge job cleared."; render();
  });
  el("bridgeRows").addEventListener("click", event => {
    const button = event.target.closest("[data-bridge-remove]");
    if (!button) return;
    job.rows.splice(Number(button.dataset.bridgeRemove),1); render();
  });
  el("bridgeDownload").addEventListener("click", () => {
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
  root.PerfectCutBridgeUI = { render };
  render();
})(typeof window !== "undefined" ? window : globalThis);
