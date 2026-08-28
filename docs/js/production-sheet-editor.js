(function(){
  "use strict";

  const STORAGE_KEY = "loe.productionSheetEditor.v1";
  const STORAGE_VERSION = 1;
  const MAX_SAVED_SHEETS = 12;
  const PX_PER_MM = 96 / 25.4;
  const PAPER_MM = Object.freeze({ A4: [210, 297], A3: [297, 420] });
  const DEFAULT_LAYOUT = Object.freeze({
    paper: "A4",
    orientation: "portrait",
    marginTop: 15,
    marginBottom: 15,
    marginLeft: 15,
    marginRight: 15,
    header: "",
    footer: "",
    pageNumbers: true,
    repeatHeader: true,
  });

  function clampNumber(value, min, max, fallback){
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.min(max, Math.max(min, numeric));
  }

  function escapeHtml(value){
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function cssString(value){
    return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/[\r\n]+/g, " ");
  }

  function safeStorageRecords(){
    try{
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      if (!parsed || parsed.version !== STORAGE_VERSION || typeof parsed.sheets !== "object"){
        return {};
      }
      return parsed.sheets || {};
    }catch{
      return {};
    }
  }

  function saveStorageRecords(records){
    try{
      const sorted = Object.entries(records || {})
        .sort((a, b) => Number(b[1]?.savedAt || 0) - Number(a[1]?.savedAt || 0))
        .slice(0, MAX_SAVED_SHEETS);
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, sheets: Object.fromEntries(sorted) }));
      return true;
    }catch{
      return false;
    }
  }

  function sanitizeSavedHtml(html){
    const template = document.createElement("template");
    template.innerHTML = String(html || "");
    template.content.querySelectorAll("script,style,iframe,object,embed,link,meta").forEach(node => node.remove());
    template.content.querySelectorAll("*").forEach(node => {
      Array.from(node.attributes).forEach(attribute => {
        const name = attribute.name.toLowerCase();
        const value = String(attribute.value || "").trim().toLowerCase();
        if (name.startsWith("on") || name === "srcdoc" || ((name === "href" || name === "src") && value.startsWith("javascript:"))){
          node.removeAttribute(attribute.name);
        }
      });
    });
    return template.innerHTML;
  }

  class ProductionSheetEditor{
    constructor(root, options = {}){
      this.root = root;
      this.options = options;
      this.documentEl = root?.querySelector("#productionSheetDocument") || null;
      this.paperEl = root?.querySelector("#productionSheetPaperPreview") || null;
      this.markersEl = root?.querySelector("#productionSheetPageMarkers") || null;
      this.pageCountEl = root?.querySelector("#productionSheetPageCount") || null;
      this.statusEl = root?.querySelector("#productionSheetSaveStatus") || null;
      this.undoButton = root?.querySelector("#productionSheetUndo") || null;
      this.redoButton = root?.querySelector("#productionSheetRedo") || null;
      this.signature = "";
      this.generatedHtml = "";
      this.layout = { ...DEFAULT_LAYOUT };
      this.undoStack = [];
      this.redoStack = [];
      this.lastSnapshot = "";
      this.selectedCell = null;
      this.selectedEditable = null;
      this.savedRange = null;
      this.inputBatchActive = false;
      this.inputBatchTimer = null;
      this.pageTimer = null;
      this.isOpen = false;
      this.bind();
      this.applyLayoutToControls();
      this.applyLayoutPreview();
    }

    bind(){
      if (!this.root || !this.documentEl) return;
      this.root.querySelector("#productionSheetClose")?.addEventListener("click", () => this.close());
      this.root.querySelector("#productionSheetUndo")?.addEventListener("click", () => this.undo());
      this.root.querySelector("#productionSheetRedo")?.addEventListener("click", () => this.redo());
      this.root.querySelector("#productionSheetReset")?.addEventListener("click", () => this.reset());
      this.root.querySelector("#productionSheetSave")?.addEventListener("click", () => this.save());
      this.root.querySelector("#productionSheetPrint")?.addEventListener("click", () => this.print());

      this.root.addEventListener("mousedown", event => {
        if (event.target.closest("button[data-production-sheet-command]")) event.preventDefault();
      });
      this.root.addEventListener("click", event => {
        const commandButton = event.target.closest("[data-production-sheet-command]");
        if (!commandButton) return;
        this.runCommand(commandButton.dataset.productionSheetCommand);
      });

      this.documentEl.addEventListener("focusin", event => this.rememberTarget(event.target));
      this.documentEl.addEventListener("click", event => this.rememberTarget(event.target));
      this.documentEl.addEventListener("beforeinput", () => this.beginInputBatch());
      this.documentEl.addEventListener("input", () => {
        if (!this.inputBatchActive){
          this.undoStack.push(this.lastSnapshot || this.generatedHtml);
          if (this.undoStack.length > 60) this.undoStack.shift();
          this.redoStack = [];
          this.inputBatchActive = true;
          this.updateHistoryButtons();
        }
        this.lastSnapshot = this.captureHtml();
        this.markDirty();
        this.schedulePageCount();
        this.scheduleInputBatchEnd();
      });
      document.addEventListener("selectionchange", () => this.captureSelection());

      const layoutBindings = [
        ["#productionSheetPaper", "paper", value => PAPER_MM[value] ? value : "A4"],
        ["#productionSheetOrientation", "orientation", value => value === "landscape" ? "landscape" : "portrait"],
        ["#productionSheetMarginTop", "marginTop", value => clampNumber(value, 5, 40, 15)],
        ["#productionSheetMarginBottom", "marginBottom", value => clampNumber(value, 5, 40, 15)],
        ["#productionSheetMarginLeft", "marginLeft", value => clampNumber(value, 5, 40, 15)],
        ["#productionSheetMarginRight", "marginRight", value => clampNumber(value, 5, 40, 15)],
        ["#productionSheetHeader", "header", value => String(value || "").slice(0, 180)],
        ["#productionSheetFooter", "footer", value => String(value || "").slice(0, 180)],
        ["#productionSheetPageNumbers", "pageNumbers", (_value, input) => !!input.checked],
        ["#productionSheetRepeatHeader", "repeatHeader", (_value, input) => !!input.checked],
      ];
      layoutBindings.forEach(([selector, key, normalize]) => {
        const input = this.root.querySelector(selector);
        input?.addEventListener("input", () => this.changeLayout(key, normalize(input.value, input)));
        input?.addEventListener("change", () => this.changeLayout(key, normalize(input.value, input)));
      });

      this.root.querySelector("#productionSheetFontFamily")?.addEventListener("change", event => this.applyInlineStyle("fontFamily", event.target.value));
      this.root.querySelector("#productionSheetFontSize")?.addEventListener("change", event => this.applyInlineStyle("fontSize", `${clampNumber(event.target.value, 8, 36, 12)}px`));
      this.root.querySelector("#productionSheetTextColor")?.addEventListener("input", event => this.applyInlineStyle("color", event.target.value));
      this.root.querySelector("#productionSheetLineSpacing")?.addEventListener("change", event => this.applyBlockStyle("lineHeight", event.target.value));
      this.root.querySelector("#productionSheetSpaceBefore")?.addEventListener("change", event => this.applyBlockStyle("marginTop", `${clampNumber(event.target.value, 0, 48, 0)}px`));
      this.root.querySelector("#productionSheetSpaceAfter")?.addEventListener("change", event => this.applyBlockStyle("marginBottom", `${clampNumber(event.target.value, 0, 48, 0)}px`));
      this.root.querySelector("#productionSheetVerticalAlign")?.addEventListener("change", event => this.applyCellStyle("verticalAlign", event.target.value));
      this.root.querySelector("#productionSheetBorder")?.addEventListener("change", event => {
        const borders = { none: "none", light: "1px solid #9ca3af", strong: "2px solid #111827" };
        this.applyCellStyle("border", borders[event.target.value] || borders.light);
      });
      this.root.querySelector("#productionSheetCellColor")?.addEventListener("input", event => this.applyCellStyle("backgroundColor", event.target.value));
      window.addEventListener("resize", () => this.schedulePageCount());
    }

    setSheet(preview, signature){
      const hasGroups = Array.isArray(preview?.groups) && preview.groups.length > 0;
      if (!signature || !hasGroups){
        this.signature = "";
        this.generatedHtml = "";
        this.documentEl.innerHTML = "";
        this.lastSnapshot = "";
        this.updateHistoryButtons();
        return;
      }
      if (signature === this.signature) return;

      this.signature = signature;
      this.generatedHtml = this.buildGeneratedHtml(preview);
      const saved = safeStorageRecords()[signature];
      this.layout = this.normalizeLayout(saved?.layout || DEFAULT_LAYOUT);
      const restoredHtml = saved?.html ? sanitizeSavedHtml(saved.html) : this.generatedHtml;
      this.documentEl.innerHTML = restoredHtml;
      this.lastSnapshot = this.captureHtml();
      this.undoStack = [];
      this.redoStack = [];
      this.applyLayoutToControls();
      this.applyLayoutPreview();
      this.setStatus(saved?.html ? `Saved ${new Date(saved.savedAt).toLocaleString()}` : "Generated sheet");
      this.updateHistoryButtons();
      this.schedulePageCount();
    }

    buildGeneratedHtml(preview){
      const firstLine = String(preview?.text || "").split(/\r?\n/)[0] || "Mother Sheet";
      const separator = preview?.meta?.decimalSeparator === "dot" ? "dot" : "comma";
      const formatNumber = (numeric, display) => {
        let value = display != null && String(display).trim() ? String(display).trim() : (Number.isFinite(Number(numeric)) ? String(numeric) : "");
        if (separator === "comma") value = value.replace(".", ",");
        return value;
      };
      const tableHtml = lines => {
        const rows = (Array.isArray(lines) ? lines : []).map(line => {
          const width = formatNumber(line?.width, line?.widthDisplay);
          const height = formatNumber(line?.height, line?.heightDisplay);
          const quantity = line?.qty != null ? line.qty : "";
          return `<tr data-origin="generated"><td contenteditable="true" data-origin="generated">${escapeHtml(line?.idx ?? "")}</td><td contenteditable="true" data-origin="generated">${escapeHtml(width)}</td><td contenteditable="true" data-origin="generated">${escapeHtml(height)}</td><td contenteditable="true" data-origin="generated">${escapeHtml(quantity)}</td></tr>`;
        }).join("");
        return `<div class="ps-table-wrap ps-block" data-origin="generated"><table><thead><tr data-origin="generated"><th contenteditable="true" data-origin="generated">No.</th><th contenteditable="true" data-origin="generated">Width</th><th contenteditable="true" data-origin="generated">Height</th><th contenteditable="true" data-origin="generated">Qty</th></tr></thead><tbody>${rows || '<tr data-origin="generated"><td contenteditable="true" colspan="4">No line items</td></tr>'}</tbody></table></div>`;
      };

      const groups = preview.groups.map(group => {
        const heading = group?.headerText || group?.display || "(Header not set)";
        const sections = Array.isArray(group?.sections) && group.sections.length
          ? group.sections.map(section => {
              const orderLine = section?.orderHeaderText || `[Order ${section?.orderId || "—"} — ${section?.client || "—"}]`;
              return `<div class="ps-order-block ps-block" data-origin="generated"><p class="ps-order-line" contenteditable="true" data-origin="generated">${escapeHtml(orderLine)}</p>${tableHtml(section?.lines)}</div>`;
            }).join("")
          : tableHtml(group?.lines);
        return `<section class="ps-group ps-block" data-origin="generated"><h2 contenteditable="true" data-origin="generated">${escapeHtml(heading)}</h2>${sections}</section>`;
      }).join("");

      return `<article class="ps-document-body"><div class="ps-origin-key" contenteditable="false">Generated production data</div><h1 class="ps-sheet-title ps-block" contenteditable="true" data-origin="generated">${escapeHtml(firstLine)}</h1>${groups}</article>`;
    }

    open(){
      if (!this.signature) return false;
      this.root.hidden = false;
      this.isOpen = true;
      this.options.onOpen?.();
      this.schedulePageCount();
      this.root.scrollIntoView({ behavior: "smooth", block: "start" });
      return true;
    }

    close(){
      this.root.hidden = true;
      this.isOpen = false;
      this.options.onClose?.();
    }

    hasSheet(){
      return !!this.signature;
    }

    captureHtml(){
      return this.documentEl?.innerHTML || "";
    }

    captureSelection(){
      const selection = window.getSelection?.();
      if (!selection || !selection.rangeCount) return;
      const anchor = selection.anchorNode?.nodeType === Node.ELEMENT_NODE ? selection.anchorNode : selection.anchorNode?.parentElement;
      if (!anchor || !this.documentEl.contains(anchor)) return;
      this.savedRange = selection.getRangeAt(0).cloneRange();
      this.rememberTarget(anchor);
    }

    restoreSelection(){
      if (!this.savedRange) return false;
      const selection = window.getSelection?.();
      if (!selection) return false;
      try{
        selection.removeAllRanges();
        selection.addRange(this.savedRange);
        return true;
      }catch{
        return false;
      }
    }

    rememberTarget(target){
      const element = target?.nodeType === Node.ELEMENT_NODE ? target : target?.parentElement;
      if (!element || !this.documentEl.contains(element)) return;
      const cell = element.closest("td,th");
      if (this.selectedCell && this.selectedCell !== cell) this.selectedCell.classList.remove("is-selected");
      this.selectedCell = cell;
      if (cell) cell.classList.add("is-selected");
      this.selectedEditable = element.closest('[contenteditable="true"],td,th') || cell;
    }

    currentTarget(){
      if (this.selectedEditable && this.documentEl.contains(this.selectedEditable)) return this.selectedEditable;
      return this.documentEl.querySelector('[contenteditable="true"]');
    }

    beginInputBatch(){
      if (!this.inputBatchActive){
        this.undoStack.push(this.captureHtml());
        if (this.undoStack.length > 60) this.undoStack.shift();
        this.redoStack = [];
        this.inputBatchActive = true;
        this.updateHistoryButtons();
      }
      clearTimeout(this.inputBatchTimer);
    }

    scheduleInputBatchEnd(){
      clearTimeout(this.inputBatchTimer);
      this.inputBatchTimer = setTimeout(() => {
        this.inputBatchActive = false;
        this.lastSnapshot = this.captureHtml();
      }, 500);
    }

    mutate(callback){
      this.inputBatchActive = false;
      clearTimeout(this.inputBatchTimer);
      const before = this.captureHtml();
      callback();
      const after = this.captureHtml();
      if (after === before) return;
      this.undoStack.push(before);
      if (this.undoStack.length > 60) this.undoStack.shift();
      this.redoStack = [];
      this.lastSnapshot = after;
      this.markDirty();
      this.updateHistoryButtons();
      this.schedulePageCount();
    }

    undo(){
      const previous = this.undoStack.pop();
      if (previous == null) return;
      this.redoStack.push(this.captureHtml());
      this.documentEl.innerHTML = previous;
      this.lastSnapshot = previous;
      this.markDirty();
      this.updateHistoryButtons();
      this.schedulePageCount();
    }

    redo(){
      const next = this.redoStack.pop();
      if (next == null) return;
      this.undoStack.push(this.captureHtml());
      this.documentEl.innerHTML = next;
      this.lastSnapshot = next;
      this.markDirty();
      this.updateHistoryButtons();
      this.schedulePageCount();
    }

    reset(){
      if (!this.generatedHtml) return;
      this.mutate(() => { this.documentEl.innerHTML = this.generatedHtml; });
      this.layout = { ...DEFAULT_LAYOUT };
      this.applyLayoutToControls();
      this.applyLayoutPreview();
      this.setStatus("Reset to generated sheet — save to keep this reset");
    }

    save(){
      if (!this.signature) return false;
      const records = safeStorageRecords();
      records[this.signature] = {
        html: sanitizeSavedHtml(this.captureHtml()),
        layout: this.normalizeLayout(this.layout),
        savedAt: Date.now(),
      };
      const saved = saveStorageRecords(records);
      this.setStatus(saved ? `Saved ${new Date().toLocaleTimeString()}` : "Could not save in this browser");
      this.options.onSave?.(saved);
      return saved;
    }

    markDirty(){
      this.setStatus("Unsaved changes");
      this.options.onDirty?.();
    }

    setStatus(message){
      if (this.statusEl) this.statusEl.textContent = message;
    }

    updateHistoryButtons(){
      if (this.undoButton) this.undoButton.disabled = this.undoStack.length === 0;
      if (this.redoButton) this.redoButton.disabled = this.redoStack.length === 0;
    }

    runCommand(command){
      const target = this.currentTarget();
      if (!target && command !== "add-note") return;
      if (["bold", "italic", "underline"].includes(command)){
        this.mutate(() => {
          this.restoreSelection();
          document.execCommand(command, false, null);
        });
        return;
      }
      if (command === "align-left") return this.applyBlockStyle("textAlign", "left");
      if (command === "align-center") return this.applyBlockStyle("textAlign", "center");
      if (command === "align-right") return this.applyBlockStyle("textAlign", "right");
      if (command === "font-smaller" || command === "font-larger"){
        const input = this.root.querySelector("#productionSheetFontSize");
        const delta = command === "font-larger" ? 1 : -1;
        const size = clampNumber(Number(input?.value || 12) + delta, 8, 36, 12);
        if (input) input.value = String(size);
        return this.applyInlineStyle("fontSize", `${size}px`);
      }
      if (command === "line-break"){
        return this.mutate(() => {
          this.restoreSelection();
          if (!document.execCommand("insertLineBreak", false, null)) document.execCommand("insertHTML", false, "<br>");
        });
      }
      if (command === "remove-line-breaks"){
        return this.mutate(() => {
          const active = this.currentTarget();
          active?.querySelectorAll("br").forEach(br => br.replaceWith(document.createTextNode(" ")));
        });
      }
      if (command === "clear-format"){
        return this.mutate(() => {
          this.restoreSelection();
          document.execCommand("removeFormat", false, null);
          const active = this.currentTarget();
          if (active) active.removeAttribute("style");
        });
      }
      if (command === "add-note") return this.addNote();
      if (command === "add-row") return this.addRow();
      if (command === "remove-row") return this.removeRow();
      if (command === "add-column") return this.addColumn();
      if (command === "remove-column") return this.removeColumn();
      if (command === "merge-right") return this.mergeRight();
      if (command === "split-cell") return this.splitCell();
      if (command === "page-break") return this.insertPageBreak();
    }

    applyInlineStyle(property, value){
      this.mutate(() => {
        const target = this.currentTarget();
        if (!target) return;
        const restored = this.restoreSelection();
        const selection = window.getSelection?.();
        if (restored && selection && !selection.isCollapsed && selection.rangeCount){
          const range = selection.getRangeAt(0);
          const span = document.createElement("span");
          span.style[property] = value;
          try{
            range.surroundContents(span);
            selection.removeAllRanges();
            const nextRange = document.createRange();
            nextRange.selectNodeContents(span);
            selection.addRange(nextRange);
            this.savedRange = nextRange.cloneRange();
            return;
          }catch{
            // Complex selections fall back to formatting the containing editable block.
          }
        }
        target.style[property] = value;
      });
    }

    applyBlockStyle(property, value){
      this.mutate(() => {
        const target = this.currentTarget()?.closest("td,th,h1,h2,p,.ps-note") || this.currentTarget();
        if (target) target.style[property] = value;
      });
    }

    applyCellStyle(property, value){
      this.mutate(() => {
        const cell = this.selectedCell || this.currentTarget()?.closest("td,th");
        if (cell) cell.style[property] = value;
      });
    }

    selectedTableCell(){
      return this.selectedCell && this.documentEl.contains(this.selectedCell) ? this.selectedCell : null;
    }

    addNote(){
      this.mutate(() => {
        const note = document.createElement("p");
        note.className = "ps-note ps-block";
        note.dataset.origin = "user";
        note.contentEditable = "true";
        note.textContent = "Operator note";
        const block = this.currentTarget()?.closest(".ps-block");
        if (block) block.insertAdjacentElement("afterend", note);
        else this.documentEl.querySelector(".ps-document-body")?.appendChild(note);
        note.focus();
        this.rememberTarget(note);
      });
    }

    addRow(){
      const cell = this.selectedTableCell();
      const row = cell?.closest("tr");
      const table = row?.closest("table");
      if (!table) return;
      this.mutate(() => {
        const columnCount = Array.from(table.rows[0]?.cells || []).reduce((sum, item) => sum + Number(item.colSpan || 1), 0) || 1;
        const newRow = document.createElement("tr");
        newRow.dataset.origin = "user";
        for (let index = 0; index < columnCount; index += 1){
          const newCell = document.createElement("td");
          newCell.contentEditable = "true";
          newCell.dataset.origin = "user";
          newCell.innerHTML = "<br>";
          newRow.appendChild(newCell);
        }
        (table.tBodies[0] || table.createTBody()).insertBefore(newRow, row?.parentElement === table.tBodies[0] ? row.nextSibling : null);
      });
    }

    removeRow(){
      const row = this.selectedTableCell()?.closest("tr");
      if (!row || row.parentElement?.rows?.length <= 1) return;
      this.mutate(() => row.remove());
    }

    addColumn(){
      const cell = this.selectedTableCell();
      const table = cell?.closest("table");
      if (!table) return;
      const index = cell.cellIndex;
      this.mutate(() => {
        Array.from(table.rows).forEach(row => {
          const newCell = document.createElement(row.parentElement?.tagName === "THEAD" ? "th" : "td");
          newCell.contentEditable = "true";
          newCell.dataset.origin = "user";
          newCell.innerHTML = row.parentElement?.tagName === "THEAD" ? "New column" : "<br>";
          row.insertBefore(newCell, row.cells[index + 1] || null);
        });
      });
    }

    removeColumn(){
      const cell = this.selectedTableCell();
      const table = cell?.closest("table");
      if (!table || table.rows[0]?.cells?.length <= 1) return;
      const index = cell.cellIndex;
      this.mutate(() => Array.from(table.rows).forEach(row => row.cells[index]?.remove()));
    }

    mergeRight(){
      const cell = this.selectedTableCell();
      const next = cell?.nextElementSibling;
      if (!cell || !next || !/^(TD|TH)$/.test(next.tagName)) return;
      this.mutate(() => {
        cell.colSpan = Number(cell.colSpan || 1) + Number(next.colSpan || 1);
        const nextText = next.textContent?.trim();
        if (nextText) cell.append(document.createElement("br"), document.createTextNode(nextText));
        next.remove();
      });
    }

    splitCell(){
      const cell = this.selectedTableCell();
      if (!cell || Number(cell.colSpan || 1) <= 1) return;
      this.mutate(() => {
        cell.colSpan = Number(cell.colSpan) - 1;
        const next = document.createElement(cell.tagName.toLowerCase());
        next.contentEditable = "true";
        next.dataset.origin = "user";
        next.innerHTML = "<br>";
        cell.insertAdjacentElement("afterend", next);
      });
    }

    insertPageBreak(){
      this.mutate(() => {
        const pageBreak = document.createElement("div");
        pageBreak.className = "ps-manual-page-break ps-block";
        pageBreak.dataset.origin = "user";
        pageBreak.contentEditable = "false";
        pageBreak.textContent = "Manual page break";
        const block = this.currentTarget()?.closest(".ps-block");
        if (block) block.insertAdjacentElement("afterend", pageBreak);
        else this.documentEl.querySelector(".ps-document-body")?.appendChild(pageBreak);
      });
    }

    changeLayout(key, value){
      if (this.layout[key] === value) return;
      this.layout[key] = value;
      this.markDirty();
      this.applyLayoutPreview();
    }

    normalizeLayout(layout){
      return {
        paper: PAPER_MM[layout?.paper] ? layout.paper : DEFAULT_LAYOUT.paper,
        orientation: layout?.orientation === "landscape" ? "landscape" : "portrait",
        marginTop: clampNumber(layout?.marginTop, 5, 40, DEFAULT_LAYOUT.marginTop),
        marginBottom: clampNumber(layout?.marginBottom, 5, 40, DEFAULT_LAYOUT.marginBottom),
        marginLeft: clampNumber(layout?.marginLeft, 5, 40, DEFAULT_LAYOUT.marginLeft),
        marginRight: clampNumber(layout?.marginRight, 5, 40, DEFAULT_LAYOUT.marginRight),
        header: String(layout?.header || "").slice(0, 180),
        footer: String(layout?.footer || "").slice(0, 180),
        pageNumbers: layout?.pageNumbers !== false,
        repeatHeader: layout?.repeatHeader !== false,
      };
    }

    paperDimensions(){
      const base = PAPER_MM[this.layout.paper] || PAPER_MM.A4;
      return this.layout.orientation === "landscape" ? [base[1], base[0]] : base.slice();
    }

    applyLayoutToControls(){
      const values = {
        productionSheetPaper: this.layout.paper,
        productionSheetOrientation: this.layout.orientation,
        productionSheetMarginTop: this.layout.marginTop,
        productionSheetMarginBottom: this.layout.marginBottom,
        productionSheetMarginLeft: this.layout.marginLeft,
        productionSheetMarginRight: this.layout.marginRight,
        productionSheetHeader: this.layout.header,
        productionSheetFooter: this.layout.footer,
      };
      Object.entries(values).forEach(([id, value]) => {
        const input = this.root?.querySelector(`#${id}`);
        if (input) input.value = String(value);
      });
      const pageNumbers = this.root?.querySelector("#productionSheetPageNumbers");
      const repeatHeader = this.root?.querySelector("#productionSheetRepeatHeader");
      if (pageNumbers) pageNumbers.checked = !!this.layout.pageNumbers;
      if (repeatHeader) repeatHeader.checked = !!this.layout.repeatHeader;
    }

    applyLayoutPreview(){
      if (!this.paperEl || !this.documentEl) return;
      const [width, height] = this.paperDimensions();
      this.paperEl.style.setProperty("--ps-page-width", `${width}mm`);
      this.paperEl.style.setProperty("--ps-page-height", `${height}mm`);
      this.paperEl.style.setProperty("--ps-margin-top", `${this.layout.marginTop}mm`);
      this.paperEl.style.setProperty("--ps-margin-bottom", `${this.layout.marginBottom}mm`);
      this.paperEl.style.setProperty("--ps-margin-left", `${this.layout.marginLeft}mm`);
      this.paperEl.style.setProperty("--ps-margin-right", `${this.layout.marginRight}mm`);
      this.schedulePageCount();
    }

    schedulePageCount(){
      clearTimeout(this.pageTimer);
      this.pageTimer = setTimeout(() => this.updatePageCount(), 40);
    }

    updatePageCount(){
      if (!this.documentEl || !this.paperEl) return;
      const [_width, heightMm] = this.paperDimensions();
      const usableHeight = Math.max(30, heightMm - this.layout.marginTop - this.layout.marginBottom);
      const contentHeight = Math.max(0, this.documentEl.scrollHeight - ((this.layout.marginTop + this.layout.marginBottom) * PX_PER_MM));
      const manualBreaks = this.documentEl.querySelectorAll(".ps-manual-page-break").length;
      const pages = Math.max(1, Math.ceil(Math.max(0, contentHeight - 4) / (usableHeight * PX_PER_MM)) + manualBreaks);
      this.paperEl.style.setProperty("--ps-page-count", String(pages));
      this.paperEl.style.minHeight = `calc(var(--ps-page-height) * ${pages})`;
      if (this.pageCountEl) this.pageCountEl.textContent = `${pages} printable page${pages === 1 ? "" : "s"}`;
      if (this.markersEl){
        this.markersEl.innerHTML = "";
        for (let page = 1; page <= pages; page += 1){
          const marker = document.createElement("div");
          marker.className = "production-sheet-page-marker";
          marker.style.top = `calc(var(--ps-page-height) * ${page})`;
          marker.textContent = `Page ${page}`;
          this.markersEl.appendChild(marker);
        }
      }
    }

    getLayout(){
      return this.normalizeLayout(this.layout);
    }

    getPlainText(){
      if (!this.hasSheet()) return "";
      const clone = this.documentEl.cloneNode(true);
      clone.querySelectorAll(".ps-origin-key,.ps-manual-page-break").forEach(node => node.remove());
      clone.querySelectorAll("tr").forEach(row => {
        const line = document.createTextNode(`${Array.from(row.cells).map(cell => cell.innerText.trim()).join(" | ")}\n`);
        row.replaceWith(line);
      });
      return clone.innerText.replace(/\n{3,}/g, "\n\n").trim();
    }

    print(){
      if (!this.hasSheet()) return false;
      const printWindow = window.open("", "_blank");
      if (!printWindow){
        this.setStatus("Allow pop-ups to open the print dialog");
        return false;
      }
      const clone = this.documentEl.cloneNode(true);
      clone.querySelectorAll("[contenteditable]").forEach(node => node.removeAttribute("contenteditable"));
      clone.querySelectorAll(".is-selected,.ps-origin-key").forEach(node => node.classList.remove("is-selected"));
      const paper = this.layout.paper;
      const orientation = this.layout.orientation;
      const header = cssString(this.layout.header);
      const footer = cssString(this.layout.footer);
      const pageNumberRule = this.layout.pageNumbers ? 'content:"Page " counter(page) " of " counter(pages);' : "content:'';";
      const repeatHeaderRule = this.layout.repeatHeader ? "table-header-group" : "table-row-group";
      const html = `<!doctype html><html><head><meta charset="utf-8"><title>Production Sheet</title><style>
        @page{size:${paper} ${orientation};margin:${this.layout.marginTop}mm ${this.layout.marginRight}mm ${this.layout.marginBottom}mm ${this.layout.marginLeft}mm;
          @top-center{content:"${header}";font:9pt Arial,sans-serif;color:#4b5563;}
          @bottom-center{content:"${footer}";font:9pt Arial,sans-serif;color:#4b5563;}
          @bottom-right{${pageNumberRule}font:9pt Arial,sans-serif;color:#4b5563;}
        }
        *{box-sizing:border-box}html,body{margin:0;padding:0;background:#fff;color:#111827;font:11pt/1.45 Arial,sans-serif}
        .ps-document-body{width:100%}.ps-origin-key{display:none}.ps-sheet-title{font-size:16pt;line-height:1.25;margin:0 0 7mm}.ps-group{margin:0 0 6mm;break-inside:auto}.ps-group h2{font-size:13pt;margin:0 0 2.5mm;break-after:avoid}.ps-order-line{font-weight:600;margin:3mm 0 2mm;break-after:avoid}.ps-table-wrap{overflow:visible;margin:0 0 3mm}table{border-collapse:collapse;width:100%;table-layout:fixed}thead{display:${repeatHeaderRule}}tr{break-inside:avoid}th,td{border:1px solid #9ca3af;padding:2mm 2.5mm;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eef2f7;font-weight:700}.ps-note{background:#fff7cc;border-left:3px solid #d6a700;padding:2.5mm;margin:3mm 0}.ps-manual-page-break{break-before:page;height:0;border:0;font-size:0;color:transparent}.is-selected{outline:none!important}
      </style></head><body>${clone.innerHTML}</body></html>`;
      printWindow.document.open();
      printWindow.document.write(html);
      printWindow.document.close();
      const invokePrint = () => {
        printWindow.focus();
        printWindow.print();
      };
      if (printWindow.document.readyState === "complete") setTimeout(invokePrint, 80);
      else printWindow.addEventListener("load", () => setTimeout(invokePrint, 80), { once: true });
      this.options.onPrint?.();
      return true;
    }
  }

  window.ProductionSheetEditor = ProductionSheetEditor;
})();
