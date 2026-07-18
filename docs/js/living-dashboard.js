(function attachLivingDashboard(globalObject, factory){
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (globalObject) globalObject.LivingDashboard = api;
})(typeof window !== "undefined" ? window : globalThis, function livingDashboardFactory(){
  "use strict";

  const SCHEMA_VERSION = 1;
  const DEFAULT_COLUMNS = 12;
  const MAX_STORAGE_BYTES = 16000;

  function cloneLayout(layout){
    return layout.map(item => ({
      id: item.id,
      x: item.x,
      y: item.y,
      w: item.w,
      h: item.h,
      collapsed: item.collapsed === true,
    }));
  }

  function effectiveHeight(item){
    return item.collapsed ? 1 : item.h;
  }

  function overlaps(first, second){
    return first.x < second.x + second.w
      && first.x + first.w > second.x
      && first.y < second.y + effectiveHeight(second)
      && first.y + effectiveHeight(first) > second.y;
  }

  function hasCollision(candidate, layout, ignoredId){
    return layout.some(item => item.id !== ignoredId && overlaps(candidate, item));
  }

  function specsById(widgetSpecs){
    return Object.fromEntries(widgetSpecs.map(spec => [spec.id, spec]));
  }

  function clampItem(item, spec, columns){
    const minW = Math.max(1, Number(spec.minW || 1));
    const maxW = Math.min(columns, Number(spec.maxW || columns));
    const minH = Math.max(1, Number(spec.minH || 1));
    const maxH = Math.max(minH, Number(spec.maxH || 20));
    const w = Math.max(minW, Math.min(maxW, Math.round(Number(item.w) || minW)));
    const h = Math.max(minH, Math.min(maxH, Math.round(Number(item.h) || minH)));
    return {
      id: item.id,
      x: Math.max(0, Math.min(columns - w, Math.round(Number(item.x) || 0))),
      y: Math.max(0, Math.min(500, Math.round(Number(item.y) || 0))),
      w,
      h,
      collapsed: item.collapsed === true,
    };
  }

  function findFreePosition(candidate, layout, columns, ignoredId){
    const maxExistingY = layout.reduce(
      (maximum, item) => Math.max(maximum, item.y + effectiveHeight(item)),
      0,
    );
    const maxSearchY = Math.max(maxExistingY + 30, candidate.y + 30);
    let best = null;
    let bestScore = Number.POSITIVE_INFINITY;
    for (let y = 0; y <= maxSearchY; y += 1){
      for (let x = 0; x <= columns - candidate.w; x += 1){
        const attempt = { ...candidate, x, y };
        if (hasCollision(attempt, layout, ignoredId)) continue;
        const score = Math.abs(y - candidate.y) * 3 + Math.abs(x - candidate.x);
        if (score < bestScore){
          best = attempt;
          bestScore = score;
        }
      }
    }
    return best || { ...candidate, x: 0, y: maxExistingY };
  }

  function compactLayout(layout){
    const placed = [];
    cloneLayout(layout)
      .sort((first, second) => first.y - second.y || first.x - second.x)
      .forEach(item => {
        let compacted = { ...item };
        while (compacted.y > 0){
          const moved = { ...compacted, y: compacted.y - 1 };
          if (hasCollision(moved, placed)) break;
          compacted = moved;
        }
        placed.push(compacted);
      });
    const order = new Map(layout.map((item, index) => [item.id, index]));
    return placed.sort((first, second) => order.get(first.id) - order.get(second.id));
  }

  function validateAndMergeLayout(payload, defaultLayout, widgetSpecs, columns = DEFAULT_COLUMNS){
    const defaults = cloneLayout(defaultLayout);
    const specs = specsById(widgetSpecs);
    if (!payload || typeof payload !== "object" || Array.isArray(payload)){
      return { valid: false, reason: "layout payload is not an object", layout: defaults };
    }
    if (payload.version !== SCHEMA_VERSION || !Array.isArray(payload.widgets)){
      return { valid: false, reason: "unsupported or missing layout schema", layout: defaults };
    }
    if (payload.widgets.length > widgetSpecs.length + 20){
      return { valid: false, reason: "layout contains too many widgets", layout: defaults };
    }

    const accepted = [];
    const seen = new Set();
    for (const raw of payload.widgets){
      if (!raw || typeof raw !== "object" || typeof raw.id !== "string"){
        return { valid: false, reason: "layout contains an invalid widget", layout: defaults };
      }
      if (!specs[raw.id]) continue;
      if (seen.has(raw.id)){
        return { valid: false, reason: `layout repeats widget ${raw.id}`, layout: defaults };
      }
      seen.add(raw.id);
      const numericFields = [raw.x, raw.y, raw.w, raw.h];
      if (!numericFields.every(Number.isInteger)){
        return { valid: false, reason: `layout has non-integer bounds for ${raw.id}`, layout: defaults };
      }
      const spec = specs[raw.id];
      const normalized = clampItem(raw, spec, columns);
      if (
        normalized.x !== raw.x
        || normalized.y !== raw.y
        || normalized.w !== raw.w
        || normalized.h !== raw.h
        || (raw.collapsed !== undefined && typeof raw.collapsed !== "boolean")
      ){
        return { valid: false, reason: `layout has out-of-bounds values for ${raw.id}`, layout: defaults };
      }
      accepted.push(normalized);
    }
    for (let index = 0; index < accepted.length; index += 1){
      if (hasCollision(accepted[index], accepted, accepted[index].id)){
        return { valid: false, reason: "layout contains overlapping widgets", layout: defaults };
      }
    }

    for (const defaultItem of defaults){
      if (seen.has(defaultItem.id)) continue;
      const candidate = clampItem(defaultItem, specs[defaultItem.id], columns);
      accepted.push(
        hasCollision(candidate, accepted)
          ? findFreePosition(candidate, accepted, columns)
          : candidate,
      );
    }
    return { valid: true, reason: "", layout: accepted };
  }

  function create(options){
    const root = options.root;
    const toolbar = options.toolbar;
    const editButton = options.editButton;
    const doneButton = options.doneButton;
    const resetButton = options.resetButton;
    const saveStatus = options.saveStatus;
    const storage = options.storage;
    const storageKey = options.storageKey;
    const widgetSpecs = options.widgets || [];
    const columns = options.columns || DEFAULT_COLUMNS;
    const rowHeight = options.rowHeight || 56;
    const gap = options.gap || 12;
    const defaults = cloneLayout(options.defaultLayout || []);
    const specs = specsById(widgetSpecs);
    const origins = new Map();
    const wrappers = new Map();
    let editMode = false;
    let destroyed = false;
    let saveTimer = null;
    let resizeFrame = null;
    let activePointer = null;
    let layout = cloneLayout(defaults);

    if (!root || !toolbar || !storage || !storageKey){
      throw new Error("Living dashboard is missing required initialization options.");
    }

    try{
      const raw = storage.getItem(storageKey);
      if (raw){
        if (raw.length > MAX_STORAGE_BYTES) throw new Error("saved layout exceeds the size limit");
        const result = validateAndMergeLayout(JSON.parse(raw), defaults, widgetSpecs, columns);
        if (!result.valid) throw new Error(result.reason);
        layout = result.layout;
      }
    }catch(error){
      layout = cloneLayout(defaults);
      console.warn("Living dashboard ignored an invalid saved layout:", error?.message || error);
    }

    const grid = document.createElement("div");
    grid.className = "living-dashboard-grid";
    grid.setAttribute("aria-label", "Configurable dashboard widgets");
    toolbar.insertAdjacentElement("afterend", grid);

    widgetSpecs.forEach(spec => {
      const element = root.querySelector(`[data-dashboard-widget="${spec.id}"]`);
      if (!element) return;
      origins.set(spec.id, { parent: element.parentNode, nextSibling: element.nextSibling });

      const wrapper = document.createElement("section");
      wrapper.className = "living-dashboard-widget";
      wrapper.dataset.widgetId = spec.id;
      wrapper.setAttribute("aria-label", spec.title);

      const header = document.createElement("div");
      header.className = "living-dashboard-widget-header";
      header.innerHTML = `<strong>${escapeText(spec.title)}</strong>`;

      const controls = document.createElement("div");
      controls.className = "living-dashboard-widget-controls";
      const dragHandle = document.createElement("button");
      dragHandle.type = "button";
      dragHandle.className = "living-dashboard-drag-handle";
      dragHandle.setAttribute("aria-label", `Move ${spec.title}`);
      dragHandle.title = `Move ${spec.title}`;
      dragHandle.textContent = "Drag";
      const collapseButton = document.createElement("button");
      collapseButton.type = "button";
      collapseButton.className = "living-dashboard-collapse";
      collapseButton.setAttribute("aria-label", `Collapse ${spec.title}`);
      collapseButton.textContent = "Collapse";
      controls.append(dragHandle, collapseButton);
      header.appendChild(controls);

      const body = document.createElement("div");
      body.className = "living-dashboard-widget-body";
      body.appendChild(element);

      const error = document.createElement("div");
      error.className = "living-dashboard-widget-error";
      error.hidden = true;
      error.innerHTML = `<strong>${escapeText(spec.title)}</strong><span>This widget could not be updated.</span><button type="button" class="btn small">Retry</button>`;
      error.querySelector("button").addEventListener("click", () => options.onRetry?.(spec.id));

      const resizeHandle = document.createElement("button");
      resizeHandle.type = "button";
      resizeHandle.className = "living-dashboard-resize-handle";
      resizeHandle.setAttribute("aria-label", `Resize ${spec.title}`);
      resizeHandle.title = `Resize ${spec.title}`;

      wrapper.append(header, body, error, resizeHandle);
      grid.appendChild(wrapper);
      wrappers.set(spec.id, { wrapper, body, dragHandle, collapseButton, resizeHandle, error });

      dragHandle.addEventListener("pointerdown", event => beginPointer(event, spec.id, "drag"));
      resizeHandle.addEventListener("pointerdown", event => beginPointer(event, spec.id, "resize"));
      collapseButton.addEventListener("click", () => toggleCollapsed(spec.id));
    });

    function escapeText(value){
      return String(value || "").replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
      })[character]);
    }

    function itemById(id){
      return layout.find(item => item.id === id);
    }

    function isDesktopEditable(){
      return window.matchMedia("(min-width: 1025px)").matches;
    }

    function render(){
      if (destroyed) return;
      const width = grid.clientWidth;
      if (!width){
        resizeFrame = window.requestAnimationFrame(render);
        return;
      }
      const columnWidth = (width - gap * (columns - 1)) / columns;
      let maximumBottom = 0;
      layout.forEach(item => {
        const parts = wrappers.get(item.id);
        if (!parts) return;
        const heightRows = effectiveHeight(item);
        parts.wrapper.style.left = `${item.x * (columnWidth + gap)}px`;
        parts.wrapper.style.top = `${item.y * (rowHeight + gap)}px`;
        parts.wrapper.style.width = `${item.w * columnWidth + (item.w - 1) * gap}px`;
        parts.wrapper.style.height = `${heightRows * rowHeight + (heightRows - 1) * gap}px`;
        parts.wrapper.classList.toggle("is-collapsed", item.collapsed);
        parts.collapseButton.textContent = item.collapsed ? "Expand" : "Collapse";
        parts.collapseButton.setAttribute("aria-label", `${item.collapsed ? "Expand" : "Collapse"} ${specs[item.id].title}`);
        maximumBottom = Math.max(maximumBottom, item.y + heightRows);
      });
      grid.style.height = maximumBottom
        ? `${maximumBottom * rowHeight + (maximumBottom - 1) * gap}px`
        : "0px";
    }

    function setEditMode(enabled){
      editMode = enabled === true && isDesktopEditable();
      root.classList.toggle("living-dashboard-editing", editMode);
      editButton.hidden = editMode;
      doneButton.hidden = !editMode;
      resetButton.hidden = !editMode;
      render();
      options.onEditModeChange?.(editMode);
    }

    function setSaveState(state, message){
      saveStatus.className = `overview-layout-save-status ${state ? `is-${state}` : ""}`.trim();
      saveStatus.textContent = message || "";
    }

    function persistLayout(){
      window.clearTimeout(saveTimer);
      setSaveState("saving", "Saving layout…");
      saveTimer = window.setTimeout(() => {
        try{
          const payload = JSON.stringify({ version: SCHEMA_VERSION, widgets: cloneLayout(layout) });
          if (payload.length > MAX_STORAGE_BYTES) throw new Error("layout exceeds storage size limit");
          storage.setItem(storageKey, payload);
          setSaveState("saved", "Layout saved");
        }catch(error){
          console.warn("Living dashboard could not save the layout:", error?.message || error);
          setSaveState("error", "Layout could not be saved");
        }
      }, 120);
    }

    function toggleCollapsed(id){
      const current = itemById(id);
      if (!current) return;
      const next = { ...current, collapsed: !current.collapsed };
      if (!next.collapsed){
        const free = findFreePosition(next, layout, columns, id);
        next.x = free.x;
        next.y = free.y;
      }
      layout = layout.map(item => item.id === id ? next : item);
      layout = compactLayout(layout);
      render();
      persistLayout();
    }

    function beginPointer(event, id, type){
      if (!editMode || !isDesktopEditable() || event.button !== 0) return;
      const item = itemById(id);
      const parts = wrappers.get(id);
      if (!item || !parts || item.collapsed && type === "resize") return;
      event.preventDefault();
      event.stopPropagation();
      const itemRect = parts.wrapper.getBoundingClientRect();
      activePointer = {
        id,
        type,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        offsetX: event.clientX - itemRect.left,
        offsetY: event.clientY - itemRect.top,
        original: { ...item },
      };
      parts.wrapper.classList.add(type === "drag" ? "is-dragging" : "is-resizing");
      event.currentTarget.setPointerCapture?.(event.pointerId);
      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", endPointer, { once: true });
      window.addEventListener("pointercancel", endPointer, { once: true });
    }

    function handlePointerMove(event){
      if (!activePointer || event.pointerId !== activePointer.pointerId) return;
      event.preventDefault();
      const gridRect = grid.getBoundingClientRect();
      const columnWidth = (grid.clientWidth - gap * (columns - 1)) / columns;
      const stepX = columnWidth + gap;
      const stepY = rowHeight + gap;
      const current = itemById(activePointer.id);
      if (!current) return;
      if (activePointer.type === "drag"){
        const spec = specs[current.id];
        const target = clampItem({
          ...current,
          x: Math.round((event.clientX - gridRect.left - activePointer.offsetX) / stepX),
          y: Math.round((event.clientY - gridRect.top - activePointer.offsetY) / stepY),
        }, spec, columns);
        const free = findFreePosition(target, layout, columns, current.id);
        layout = layout.map(item => item.id === current.id ? free : item);
      }else{
        const spec = specs[current.id];
        const candidate = clampItem({
          ...current,
          w: activePointer.original.w + Math.round((event.clientX - activePointer.startX) / stepX),
          h: activePointer.original.h + Math.round((event.clientY - activePointer.startY) / stepY),
        }, spec, columns);
        candidate.x = current.x;
        candidate.y = current.y;
        if (!hasCollision(candidate, layout, current.id)){
          layout = layout.map(item => item.id === current.id ? candidate : item);
        }
      }
      render();
    }

    function endPointer(event){
      if (!activePointer || event.pointerId !== activePointer.pointerId) return;
      const parts = wrappers.get(activePointer.id);
      parts?.wrapper.classList.remove("is-dragging", "is-resizing");
      activePointer = null;
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", endPointer);
      window.removeEventListener("pointercancel", endPointer);
      layout = compactLayout(layout);
      render();
      persistLayout();
    }

    function handleResize(){
      window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        if (editMode && !isDesktopEditable()) setEditMode(false);
        render();
      });
    }

    function resetLayout(){
      if (!window.confirm("Reset the dashboard layout to its default arrangement? No order or production data will be changed.")) return;
      layout = cloneLayout(defaults);
      render();
      persistLayout();
    }

    function showWidgetError(id){
      const parts = wrappers.get(id);
      if (!parts) return;
      parts.error.hidden = false;
      parts.body.hidden = true;
    }

    function clearWidgetError(id){
      const parts = wrappers.get(id);
      if (!parts) return;
      parts.error.hidden = true;
      parts.body.hidden = false;
    }

    function destroy(){
      if (destroyed) return;
      destroyed = true;
      window.clearTimeout(saveTimer);
      window.cancelAnimationFrame(resizeFrame);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("pointermove", handlePointerMove);
      editButton.removeEventListener("click", handleEdit);
      doneButton.removeEventListener("click", handleDone);
      resetButton.removeEventListener("click", resetLayout);
      [...widgetSpecs].reverse().forEach(spec => {
        const id = spec.id;
        const parts = wrappers.get(id);
        if (!parts) return;
        const original = parts.body.firstElementChild;
        const origin = origins.get(id);
        if (!original || !origin?.parent) return;
        const reference = origin.nextSibling?.parentNode === origin.parent
          ? origin.nextSibling
          : null;
        origin.parent.insertBefore(original, reference);
      });
      grid.remove();
      root.classList.remove("living-dashboard-enabled", "living-dashboard-editing");
      toolbar.hidden = true;
    }

    const handleEdit = () => setEditMode(true);
    const handleDone = () => setEditMode(false);
    editButton.addEventListener("click", handleEdit);
    doneButton.addEventListener("click", handleDone);
    resetButton.addEventListener("click", resetLayout);
    window.addEventListener("resize", handleResize);

    root.classList.add("living-dashboard-enabled");
    toolbar.hidden = false;
    setEditMode(false);
    setSaveState("", "");
    render();

    return {
      clearWidgetError,
      destroy,
      getLayout: () => cloneLayout(layout),
      isEditing: () => editMode,
      resetLayout,
      setEditMode,
      showWidgetError,
    };
  }

  return {
    SCHEMA_VERSION,
    cloneLayout,
    compactLayout,
    create,
    effectiveHeight,
    overlaps,
    validateAndMergeLayout,
  };
});
