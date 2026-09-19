/* Session-only preparation choices. Saved orders are never rewritten by grouping. */
(function(root){
  "use strict";
  const selected = new Set();
  const positive = value => (typeof value === "number" ||
    (typeof value === "string" && /^\d+(?:\.\d+)?$/.test(value))) && Number.isFinite(Number(value)) && Number(value) > 0;
  function buckets(rows){
    const result = [], bySize = new Map();
    rows.forEach((row,index) => {
      const valid = positive(row.width_mm) && positive(row.height_mm) && positive(row.quantity) &&
        Number.isSafeInteger(Number(row.quantity)) && (typeof row.quantity !== "string" || /^\d+$/.test(row.quantity));
      const key = valid ? JSON.stringify([Number(row.width_mm),Number(row.height_mm)]) : null;
      const existing = key === null ? null : bySize.get(key);
      if (existing && Number.isSafeInteger(existing.quantity + Number(row.quantity))){
        existing.indexes.push(index);
        existing.quantity += Number(row.quantity);
      }else{
        const group = { indexes:[index], quantity:valid ? Number(row.quantity) : row.quantity };
        result.push(group);
        if (key !== null) bySize.set(key,group);
      }
    });
    return result;
  }
  const api = {
    buckets,
    isGrouped: id => id != null && selected.has(String(id)),
    setGrouped(id, grouped){
      if (id == null) return;
      if (grouped) selected.add(String(id)); else selected.delete(String(id));
    },
  };
  root.ManualDimensionGroups = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window === "undefined" ? globalThis : window);
