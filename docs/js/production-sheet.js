/* Production copies are presentation drafts; prepared Processing data stays authoritative. */
(function(root){
  "use strict";
  const defaults = Object.freeze({layout:"auto", columns:"auto", orientation:"auto", font_size:14,
    line_spacing:1.15, margin_mm:12.7, section_gap_pt:7, glass_after_pt:0, note:"", cut_guide:true});
  function capture(processing, busy=false){
    if (busy || processing?.loading || processing?.recalculating) throw new Error("Processing is busy. Wait until preparation finishes.");
    const preview = processing?.preview;
    const groups = preview?.groups || [];
    const lines = groups.flatMap(group => group.lines || []);
    if (!preview?.text?.trim() || !lines.length) throw new Error("Add orders to Processing first.");
    if (lines.some(line => line.invalid || !Number.isInteger(line.qty) || line.qty <= 0
      || !Number.isFinite(line.width) || line.width <= 0 || !Number.isFinite(line.height) || line.height <= 0)){
      throw new Error("Review invalid dimensions or quantities in Processing before printing.");
    }
    const source = {text:preview.text, glass_headers:groups.map(group => group.headerText || group.display || "(Header not set)"),
      order_headers:groups.flatMap(group => (group.sections || []).map(section => section.orderHeaderText)).filter(Boolean),
      row_count:lines.length, piece_count:lines.reduce((sum,line) => sum + line.qty,0)};
    const signature = JSON.stringify({source, groups, rows:processing.rows, options:processing.options});
    return {source, signature};
  }
  const api = {defaults, capture};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ProductionSheets = api;
  if (!root.document) return;

  const el = id => document.getElementById(id);
  const dialog = el("productionSheetDialog");
  if (!dialog) return;
  const fields = [...dialog.querySelectorAll("[data-sheet-setting]")];
  const state = {snapshot:null, settings:{...defaults}, current:null, proposal:null, shown:null,
    pdf:null, page:1, canvas:null, generation:0, controller:null, busy:"", dirty:false, stale:false, draft:null, zoom:false,
    sourceAvailable:false, aiResult:null};
  const status = message => { el("productionSheetStatus").textContent = message; };
  const take = () => capture(appState.processing, processingBridgeBusy > 0);
  function form(settings){
    fields.forEach(field => {
      const value = settings[field.dataset.sheetSetting];
      if (field.type === "checkbox") field.checked = !!value;
      else field.value = value;
    });
  }
  function readForm(){
    const settings = {...defaults};
    fields.forEach(field => {
      settings[field.dataset.sheetSetting] = field.type === "checkbox" ? field.checked
        : field.type === "number" ? Number(field.value) : field.value;
    });
    return settings;
  }
  function summary(preview){
    if (!preview) return "";
    return preview.layout === "cuttable" ? "2 copies · 1 A4 sheet · cut down the middle"
      : `${preview.columns} column${preview.columns === 1 ? "" : "s"} per copy · 2 copies · ${preview.sheet_count} A4 sheets`;
  }
  function controls(){
    const pending = !!state.proposal;
    const unavailable = state.stale || state.dirty || !state.current || pending;
    el("productionSheetOpen").disabled = !state.sourceAvailable || !!state.busy;
    el("productionSheetAuto").disabled = !state.sourceAvailable || !!state.busy;
    el("productionSheetPrint").disabled = unavailable || !!state.busy;
    el("productionSheetSave").disabled = unavailable || !!state.busy;
    el("productionSheetAsk").disabled = state.stale || state.dirty || !state.pdf || !!state.busy || pending;
    el("productionSheetApply").disabled = state.stale || !!state.busy || !pending;
    el("productionSheetDiscard").disabled = !!state.busy;
    el("productionSheetReset").disabled = !!state.busy;
    fields.forEach(field => { field.disabled = !!state.busy || pending || state.stale; });
    if (el("productionSheetLayout").value === "cuttable"){
      el("productionSheetColumns").disabled = true;
      el("productionSheetOrientation").disabled = true;
    }
    el("productionSheetProposal").hidden = !pending && !state.aiResult;
    el("productionSheetResultTitle").textContent = pending ? "AI proposal" : "AI chosen layout";
    el("productionSheetReview").hidden = !pending;
    el("productionSheetStale").hidden = !state.stale;
    el("productionSheetPrev").disabled = !!state.busy || state.page <= 1;
    el("productionSheetNext").disabled = !!state.busy || !state.pdf || state.page >= state.pdf.numPages;
    el("productionSheetZoom").disabled = !!state.busy || !state.pdf;
    root.ProductionSheetVoice?.update();
  }
  function stop(){
    state.generation++;
    state.controller?.abort();
    state.controller = null;
  }
  function fresh(){
    if (take().signature !== state.snapshot?.signature) throw new Error("Processing changed. Refresh this sheet before continuing.");
  }
  async function post(path, body, timeout){
    state.controller = new AbortController();
    const controller = state.controller;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); },timeout);
    try{
      const response = await fetch(`${API_BASE}/api/production-sheets/${path}`,{
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body), signal:controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok){
        if (response.status === 404) throw new Error("Production sheets are not available on this server yet.");
        throw new Error(typeof data.detail === "string" ? data.detail : "Check the sheet settings and try again.");
      }
      return data;
    }catch(error){
      if (error.name === "AbortError"){
        if (timedOut && path === "ai") throw new Error("AI took too long to respond. Try again. Your current sheet is still ready to print.");
        throw new Error(timedOut ? "Preparing the sheet timed out. You can try again." : "The request was cancelled.");
      }
      throw error;
    }finally{
      clearTimeout(timer);
      if (state.controller === controller) state.controller = null;
    }
  }
  function bytes(preview){
    return Uint8Array.from(atob(preview.pdf_base64),char => char.charCodeAt(0));
  }
  async function pageCanvas(pdf, number, scale=1.8){
    const page = await pdf.getPage(number);
    const viewport = page.getViewport({scale});
    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(viewport.width); canvas.height = Math.ceil(viewport.height);
    await page.render({canvasContext:canvas.getContext("2d"), viewport, background:"white"}).promise;
    return canvas;
  }
  async function showPage(number){
    const generation = state.generation, pdf = state.pdf;
    if (!pdf) return;
    const canvas = await pageCanvas(pdf,number);
    if (generation !== state.generation || pdf !== state.pdf || !dialog.open) return;
    state.page = number; state.canvas = canvas;
    el("productionSheetPages").replaceChildren(canvas);
    canvas.setAttribute("aria-label",`Production sheet page ${number} of ${pdf.numPages}`);
    el("productionSheetPageLabel").textContent = `Page ${number} of ${pdf.numPages}`;
    controls();
  }
  async function show(preview, generation){
    const lib = await ensurePdfJs();
    // Draw embedded glyph outlines rather than depending on browser FontFace
    // loading/caching between PDFs with different font subsets.
    const pdf = await lib.getDocument({data:bytes(preview), useWorkerFetch:false,
      disableFontFace:true, useSystemFonts:false}).promise;
    if (generation !== state.generation || !dialog.open){ await pdf.destroy(); return; }
    const previous = state.pdf;
    state.pdf = pdf; state.shown = preview; state.canvas = null;
    if (previous) await previous.destroy();
    el("productionSheetSummary").textContent = summary(preview);
    el("productionSheetCounts").textContent = `${preview.row_count} rows · ${preview.piece_count} pieces in the job · ${preview.orientation} A4`;
    await showPage(1);
  }
  async function render(settings){
    stop(); const generation = state.generation;
    state.busy = "render"; state.dirty = true; state.aiResult = null; controls(); status("Preparing your two production copies…");
    try{
      fresh();
      const preview = await post("preview",{source:state.snapshot.source,settings},45000);
      if (generation !== state.generation || !dialog.open) return;
      fresh();
      await show(preview,generation);
      if (generation !== state.generation) return;
      state.current = preview; state.settings = settings; state.dirty = false;
      state.draft = {signature:state.snapshot.signature, settings:{...settings}};
      status("Ready to print. The document includes both copies; set the printer’s Copies to 1.");
      return true;
    }catch(error){
      if (generation === state.generation) status(error.message);
    }finally{
      if (generation === state.generation){ state.busy = ""; controls(); }
    }
  }
  async function open(refresh=false, automatic=false){
    if (state.busy) return;
    try{
      const snapshot = take();
      stop();
      state.snapshot = snapshot; state.stale = false; state.proposal = null; state.aiResult = null; state.current = null;
      const saved = !refresh && !automatic && state.draft?.signature === snapshot.signature ? state.draft.settings : defaults;
      state.settings = {...saved}; form(state.settings);
      state.zoom=false; el("productionSheetPages").classList.remove("is-zoomed"); el("productionSheetZoom").textContent="Zoom in";
      if (!dialog.open) dialog.showModal();
      el("productionSheetRequest").value = "";
      if (await render(state.settings) && automatic) await askAI(true);
    }catch(error){
      if (dialog.open) status(error.message);
      else setStatusMessage(error.message);
    }
  }
  async function askAI(automatic=false, voiceInstruction=null){
    if (state.busy || state.stale || state.dirty || !state.pdf || state.proposal) return;
    stop(); const generation = state.generation;
    state.busy = "ai"; state.aiResult = null; controls(); status("AI is looking at the sheet and checking the layout…");
    const started = Date.now();
    const progress = setInterval(() => {
      if (generation === state.generation && state.busy === "ai" && dialog.open){
        status(`AI is still reviewing the sheet · ${Math.floor((Date.now()-started)/1000)}s elapsed. Close this window to cancel.`);
      }
    },15000);
    try{
      fresh();
      const count = state.current.pages_per_copy;
      const sampled = [...new Set([1, Math.ceil(count / 2), count])];
      const images = [];
      for (const number of sampled){
        const canvas = await pageCanvas(state.pdf,number,1.65);
        const small = document.createElement("canvas");
        const ratio = Math.min(1,1400 / Math.max(canvas.width,canvas.height));
        small.width = Math.round(canvas.width * ratio); small.height = Math.round(canvas.height * ratio);
        small.getContext("2d").drawImage(canvas,0,0,small.width,small.height);
        images.push(small.toDataURL("image/jpeg",.85));
        canvas.width = canvas.height = 0;
      }
      if (generation !== state.generation) return;
      const instruction = voiceInstruction || (!automatic && el("productionSheetRequest").value.trim())
        || "Look at this production sheet and choose the clearest readable layout, using two production copies and saving paper where practical.";
      const current = state.current;
      const result = await post("ai",{source:state.snapshot.source,settings:state.settings,instruction,images,
        rendered:{layout:current.layout,columns:current.columns,orientation:current.orientation,
          pages_per_copy:current.pages_per_copy,sheet_count:current.sheet_count,sampled_pages:sampled}},1020000);
      if (generation !== state.generation || !dialog.open) return;
      fresh();
      if (result.preview?.source_digest !== current.source_digest) throw new Error("The AI proposal belongs to a different sheet. Refresh and try again.");
      if (!automatic) state.proposal = result;
      form(result.proposal.settings);
      el("productionSheetExplanation").textContent = result.proposal.explanation;
      el("productionSheetWarnings").textContent = result.proposal.warnings.join(" · ");
      await show(result.preview,generation);
      if (generation !== state.generation || !dialog.open) return;
      fresh();
      if (automatic){
        applyResult(result); state.aiResult = result;
        downloadPdf(true);
        status(`AI prepared your PDF and started the download · reviewed ${sampled.length} page${sampled.length === 1 ? "" : "s"} from one copy. Both copies are included; printer Copies should be 1.`);
      }else{
        status(`AI proposal · reviewed ${sampled.length} page${sampled.length === 1 ? "" : "s"} from one copy. Apply or discard before printing.`);
      }
      return result;
    }catch(error){
      if (generation === state.generation){
        if (state.shown !== state.current && !state.proposal) state.dirty = true;
        status(error.message);
      }
    }finally{
      clearInterval(progress);
      if (generation === state.generation){ state.busy = ""; controls(); }
    }
  }
  function filename(){
    const orders = appState.processing.preview?.meta?.orders || [];
    return `Mother Sheet ${orders.join(" ") || "production"}`.replace(/[<>:"/\\|?*]/g,"-") + ".pdf";
  }
  function applyResult(result){
    state.current = result.preview; state.settings = result.proposal.settings;
    state.draft = {signature:state.snapshot.signature, settings:{...state.settings}};
    state.proposal = null;
  }
  function downloadPdf(automatic=false){
    fresh();
    if (!dialog.open || (state.busy && !automatic) || state.stale || state.dirty || state.proposal || !state.current) throw new Error("Apply or discard the proposal before saving the PDF.");
    const url=URL.createObjectURL(new Blob([bytes(state.current)],{type:"application/pdf"}));
    const link=document.createElement("a"); link.href=url; link.download=filename(); link.click();
    setTimeout(() => URL.revokeObjectURL(url),60000);
  }
  el("productionSheetOpen").addEventListener("click",() => open());
  el("productionSheetAuto").addEventListener("click",() => open(false,true));
  el("productionSheetClose").addEventListener("click",() => dialog.close());
  dialog.addEventListener("close",() => { root.ProductionSheetVoice?.end(); stop(); state.busy = ""; state.proposal = null; controls(); });
  el("productionSheetRefresh").addEventListener("click",() => open(true));
  el("productionSheetReset").addEventListener("click",() => {state.proposal=null; form(defaults); render({...defaults});});
  fields.forEach(field => field.addEventListener("change",() => {
    if (state.busy || state.stale || state.proposal) return;
    const settings = readForm();
    if (settings.layout === "cuttable"){settings.columns="1"; settings.orientation="landscape"; form(settings);}
    render(settings);
  }));
  el("productionSheetAsk").addEventListener("click",() => askAI());
  el("productionSheetZoom").addEventListener("click",() => {
    state.zoom=!state.zoom;
    el("productionSheetPages").classList.toggle("is-zoomed",state.zoom);
    el("productionSheetZoom").textContent=state.zoom ? "Fit page" : "Zoom in";
  });
  function applyProposal(){
    fresh();
    if (state.busy || state.stale || !state.proposal) throw new Error("There is no ready proposal to apply.");
    applyResult(state.proposal); state.aiResult = null;
    controls(); status("AI layout applied. Ready to print two copies.");
  }
  async function discardProposal(){
    fresh();
    if (state.busy || state.stale || !state.proposal) throw new Error("There is no ready proposal to discard.");
    state.proposal = null; form(state.settings);
    state.busy="render"; controls();
    try{ await show(state.current,state.generation); status("Kept your previous layout."); }
    catch(error){ state.dirty=true; status(error.message); throw error; }
    finally{state.busy=""; controls();}
  }
  el("productionSheetApply").addEventListener("click",() => {try{applyProposal();}catch(error){status(error.message);}});
  el("productionSheetDiscard").addEventListener("click",() => discardProposal().catch(error=>status(error.message)));
  for (const [id,delta] of [["productionSheetPrev",-1],["productionSheetNext",1]]){
    el(id).addEventListener("click",async () => {
      state.busy="page"; controls();
      try{ await showPage(state.page+delta); }catch(error){status(error.message);}
      finally{state.busy=""; controls();}
    });
  }
  el("productionSheetSave").addEventListener("click",() => {
    try{
      downloadPdf();
    }catch(error){status(error.message);}
  });
  el("productionSheetPrint").addEventListener("click",async () => {
    let printWindow;
    try{
      fresh();
      printWindow=window.open("","_blank");
      if (!printWindow) throw new Error("Allow the print window in your browser, then try again.");
      const preview=state.current, pdf=state.pdf, generation=state.generation;
      const width=preview.orientation === "landscape" ? 297 : 210;
      const height=preview.orientation === "landscape" ? 210 : 297;
      printWindow.document.title="Production sheet · 2 copies";
      printWindow.document.body.textContent="Preparing the print dialog…";
      const style=printWindow.document.createElement("style");
      style.textContent=`@page{size:A4 ${preview.orientation};margin:0}html,body{margin:0;padding:0;background:white}img{display:block;width:${width}mm;height:${height}mm;break-after:page;page-break-after:always}img:last-child{break-after:auto;page-break-after:auto}`;
      printWindow.document.head.appendChild(style);
      const images=[];
      state.busy="print"; controls(); status("Preparing every page for printing…");
      for(let number=1;number<=pdf.numPages;number++){
        const canvas=await pageCanvas(pdf,number,2.5);
        const image=printWindow.document.createElement("img"); image.alt=`Production sheet page ${number}`;
        image.src=canvas.toDataURL("image/png"); await image.decode(); images.push(image);
        canvas.width=canvas.height=0;
        fresh();
        if(generation!==state.generation || printWindow.closed) throw new Error("Printing cancelled because the sheet or window changed.");
      }
      printWindow.document.body.replaceChildren(...images);
      printWindow.focus(); printWindow.print();
      status("Print dialog opened. Both copies are included; printer Copies should be 1.");
    }catch(error){printWindow?.close(); status(error.message);}
    finally{state.busy=""; controls();}
  });
  function sourceChanged(){
    let snapshot;
    // Busy state is checked again at every action. Async Processing imports may
    // release their busy guard after their final UI update.
    try{snapshot=capture(appState.processing); state.sourceAvailable=true;}
    catch{state.sourceAvailable=false;}
    if(dialog.open && (!snapshot || snapshot.signature!==state.snapshot?.signature)){
      root.ProductionSheetVoice?.end("Processing changed. Start voice again after refreshing the sheet.");
      stop(); state.busy=""; state.stale=true; state.proposal=null; state.aiResult=null;
      status("Processing changed. Refresh to prepare the current orders.");
    }
    controls();
  }
  function voiceContext(){
    fresh();
    if (!dialog.open || state.stale || state.dirty || state.busy || !state.current) throw new Error("Wait until the sheet is ready.");
    return {title:state.snapshot.source.text.split("\n")[0].slice(0,2000),source_digest:state.current.source_digest,
      row_count:state.current.row_count,piece_count:state.current.piece_count,settings:{...state.settings},
      proposal:state.proposal ? JSON.stringify({settings:state.proposal.proposal.settings,explanation:state.proposal.proposal.explanation}).slice(0,3000) : ""};
  }
  async function voiceAction(decision, expected){
    if (JSON.stringify(voiceContext())!==JSON.stringify(expected)) throw new Error("The sheet changed while AI was listening. Please repeat your request.");
    switch(decision.action){
      case "propose": {
        const result=await askAI(false,decision.instruction);
        if (!result) throw new Error(el("productionSheetStatus").textContent || "The proposal could not be prepared.");
        const settings=result.proposal.settings;
        return `Proposal NOT applied: two copies, ${result.preview.layout}, ${result.preview.columns} cols/copy, ${settings.font_size} pt, after-heading gap ${settings.glass_after_pt} pt. ${result.proposal.warnings.length} warnings shown. Ask review and explicit apply/discard. ${result.proposal.explanation}`;
      }
      case "apply": applyProposal(); return "The displayed layout proposal was applied. Both production copies are ready; no order data changed.";
      case "discard": await discardProposal(); return "The proposal was discarded; the previous layout is displayed.";
      case "save_pdf":
        if (state.proposal) throw new Error("Apply or discard the proposal before saving the PDF.");
        el("productionSheetSave").focus(); el("productionSheetSave").scrollIntoView({block:"nearest"});
        return "The PDF is ready, including both copies. Ask the user to tap Save PDF to download it; the browser needs a click. No download has started yet.";
      case "clarify": return decision.reply;
      default: throw new Error("That voice action is unavailable.");
    }
  }
  root.ProductionSheetUI={sourceChanged,voiceContext,voiceAction};
  sourceChanged();
})(typeof window !== "undefined" ? window : globalThis);
