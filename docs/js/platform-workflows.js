// Shared by the website and the bounded Python workflow worker.
// Business rules and PDF layouts have one implementation here.

window.dankoRound = function(valueMm) {
  if (!Number.isFinite(valueMm)) return valueMm;
  const lastDigit = valueMm % 10;
  if (lastDigit === 1 || lastDigit === 2) return valueMm - lastDigit;        // → 0
  if (lastDigit === 6 || lastDigit === 7) return valueMm - (lastDigit - 5);  // → 5
  if (lastDigit === 4) return valueMm + (5 - lastDigit);                     // → 5
  if (lastDigit === 9) return valueMm + (10 - lastDigit);                    // → next 0
  return valueMm; // keep unchanged
};

window.applyDankoRuleToDim = function(dimStr) {
  if (!dimStr || !/^\d+x\d+$/.test(dimStr)) {
    return { w: null, h: null, text: dimStr, changed: false };
  }
  const [w0, h0] = dimStr.split("x").map(n => parseInt(n, 10));
  const w = window.dankoRound(w0);
  const h = window.dankoRound(h0);
  const changed = (w !== w0) || (h !== h0);
  return { w, h, text: `${w}x${h}`, changed, original: `${w0}x${h0}` };
};

function pickFirstString(values, fallback = ""){
  if (values == null) return fallback;
  if (!Array.isArray(values)){
    const str = String(values).trim();
    return str || fallback;
  }
  for (const value of values){
    if (Array.isArray(value)){
      const nested = pickFirstString(value);
      if (nested) return nested;
    }else if (value != null){
      const str = String(value).trim();
      if (str) return str;
    }
  }
  return fallback;
}

function pickFirstNumber(values){
  if (values == null) return null;
  if (!Array.isArray(values)){
    const numeric = typeof values === "string" ? values.replace(",", ".") : values;
    const num = Number(numeric);
    return Number.isFinite(num) ? num : null;
  }
  for (const value of values){
    if (Array.isArray(value)){
      const nested = pickFirstNumber(value);
      if (nested != null) return nested;
    }else if (value != null && value !== ""){
      const normalized = typeof value === "string" ? value.replace(",", ".") : value;
      const num = Number(normalized);
      if (Number.isFinite(num)) return num;
    }
  }
  return null;
}

function extractOrderDate(order){
  if (!order || typeof order !== "object") return null;
  const candidates = [
    order.created_at,
    order.createdAt,
    order.created,
    order.inserted_at,
    order.insertedAt,
    order.timestamp,
    order.date,
  ];
  for (const candidate of candidates){
    if (candidate == null || candidate === "") continue;
    const date = parsePlatformDate(candidate);
    if (date) return date;
  }
  return null;
}

function getOrderLabel(order){
  if (!order) return "—";
  const numbers = Array.isArray(order.order_numbers)
    ? order.order_numbers.map(value => value == null ? "" : String(value).trim()).filter(Boolean)
    : [];
  if (numbers.length){
    return numbers.join(", ");
  }
  const fallback = pickFirstString([
    order.order_number,
    order.orderNumber,
    order.order_no,
    order.orderNo,
    order.order,
    order.orderId,
    order.order_id,
    order.id,
  ]);
  if (fallback){
    return fallback;
  }
  return order.id != null ? `#${order.id}` : "—";
}

function computeRowArea(row){
  if (!row) return 0;
  const directArea = pickFirstNumber([
    row.area,
    row.area_m2,
    row.areaM2,
    row.computed_area,
    row.computedArea,
  ]);
  if (directArea != null) return directArea;
  let width = normalizeDimensionNumber(row.width);
  let height = normalizeDimensionNumber(row.height);
  if (width == null || height == null){
    const dimensionLabel = pickFirstString([
      row.dimension,
      row.dimension_display,
      row.dim,
      row.dimensionDisplay,
    ]);
    if (dimensionLabel){
      const parsed = parseDimensionTokens(dimensionLabel);
      if (width == null && parsed.width != null) width = parsed.width;
      if (height == null && parsed.height != null) height = parsed.height;
    }
  }
  if (width != null && height != null){
    return (width * height) / 1000000;
  }
  return 0;
}

function normalizeHeaderForDisplay(raw, enableLPtoG = true){
  const base = (raw || "").trim().replace(/\s+/g, " ") || "(Header not set)";
  if (!enableLPtoG) return base;
  return base.replace(/\bLP\b/gi, "G");
}

function parseDimensionTokens(dimension){
  const dim = (dimension || "").trim();
  if (!dim) return { width: null, height: null, widthDisplay: "?", heightDisplay: "?", invalid: true };
  const compact = dim.replace(/\s+/g, "");
  const match = compact.match(/^(\d{1,5}(?:[.,]\d{1,3})?)(?:[xX×])(\d{1,5}(?:[.,]\d{1,3})?)$/);
  if (match){
    const parseToken = (value)=>{
      const normalized = value.replace(",", ".");
      const parsed = Number(normalized);
      return Number.isFinite(parsed) ? parsed : null;
    };
    const widthNum = parseToken(match[1]);
    const heightNum = parseToken(match[2]);
    return {
      width: widthNum,
      height: heightNum,
      widthDisplay: match[1],
      heightDisplay: match[2],
      invalid: !Number.isFinite(widthNum) || !Number.isFinite(heightNum),
    };
  }
  const tokens = dim.split(/[xX×]/);
  const widthDisplay = (tokens[0] || "?").trim() || "?";
  const heightDisplay = (tokens[1] || "?").trim() || "?";
  return {
    width: null,
    height: null,
    widthDisplay,
    heightDisplay,
    invalid: true,
  };
}

function formatProcessingNumber(value, fallback, separator){
  if (value == null || Number.isNaN(value)){
    return (fallback !== undefined && fallback !== null && String(fallback).length) ? String(fallback) : "?";
  }
  const str = String(value);
  if (separator === "comma"){
    return str.replace(".", ",");
  }
  return str;
}

function formatM2(value, separator){
  const numeric = Number.isFinite(value) ? value : 0;
  const fixed = numeric.toFixed(3);
  return separator === "comma" ? fixed.replace(".", ",") : fixed;
}

function toNumericArea(value){
  if (value == null) return null;
  const normalized = Number(String(value).replace(",", "."));
  return Number.isFinite(normalized) ? normalized : null;
}

function computeGroupArea(rows){
  const safeRows = Array.isArray(rows) ? rows : [];
  let total = 0;
  let hasValue = false;
  safeRows.forEach(row => {
    if (!row) return;
    const candidate = row.__original?.pdfAreaValue ?? row.__original?.area ?? row.area ?? row.area_m2 ?? row.areaM2;
    const areaValue = toNumericArea(candidate);
    if (areaValue != null){
      total += areaValue;
      hasValue = true;
    }
  });
  if (!hasValue) return null;
  return Math.round(total * 1000) / 1000;
}

function normalizeDimensionNumber(value){
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string"){
    const normalized = Number(value.replace(",", "."));
    if (Number.isFinite(normalized)) return normalized;
  }
  return null;
}

function decimalPlacesFromDisplay(display){
  if (!display) return 0;
  const normalized = String(display).trim().replace(",", ".");
  if (!normalized.length) return 0;
  const parts = normalized.split(".");
  if (parts.length !== 2) return 0;
  if (!/^\d+$/.test(parts[0]) || !/^\d+$/.test(parts[1])) return 0;
  return parts[1].length;
}

function calculateRepresentativeNumber(values, decimals){
  if (!values.length) return null;
  const precision = Math.min(Number.isFinite(decimals) ? decimals : 0, 3);
  const factor = Math.pow(10, precision);
  if (!factor || !Number.isFinite(factor)) return values[values.length - 1];
  const sum = values.reduce((acc, val)=> acc + val, 0);
  return Math.round((sum / values.length) * factor) / factor;
}

function selectRepresentativeDisplay(displays, numericValue, decimals){
  const cleaned = (displays || [])
    .map(value => String(value || "").trim())
    .filter(Boolean);
  if (cleaned.length){
    let winner = cleaned[0];
    let best = 0;
    const counts = new Map();
    cleaned.forEach(value=>{
      const count = (counts.get(value) || 0) + 1;
      counts.set(value, count);
      if (count > best){
        best = count;
        winner = value;
      }
    });
    return winner;
  }
  if (numericValue != null){
    const precision = Math.min(Number.isFinite(decimals) ? decimals : 0, 3);
    if (precision > 0){
      return numericValue.toFixed(precision);
    }
    return String(Math.round(numericValue));
  }
  return "?";
}

function collapseGroupDimensions(items, tolerance = DIMENSION_TOLERANCE_MM){
  const buckets = [];
  items.forEach(item => {
    const widthNum = normalizeDimensionNumber(item.width);
    const heightNum = normalizeDimensionNumber(item.height);
    const widthKey = (item.widthDisplay || "").trim().toLowerCase();
    const heightKey = (item.heightDisplay || "").trim().toLowerCase();
    const widthDecimals = decimalPlacesFromDisplay(item.widthDisplay);
    const heightDecimals = decimalPlacesFromDisplay(item.heightDisplay);

    let bucket = null;
    for (const candidate of buckets){
      // Manual sections and red indexes are separate manufacturing identities.
      if (candidate.section !== (item.section || "") || candidate.redIndex !== (item.red_index ?? null)) continue;
      if (
        candidate.hasNumeric &&
        widthNum != null &&
        heightNum != null &&
        Math.abs(candidate.widthReference - widthNum) <= tolerance &&
        Math.abs(candidate.heightReference - heightNum) <= tolerance
      ){
        bucket = candidate;
        break;
      }
      if (
        !candidate.hasNumeric &&
        widthNum == null &&
        heightNum == null &&
        candidate.widthKey &&
        candidate.widthKey === widthKey &&
        candidate.heightKey &&
        candidate.heightKey === heightKey
      ){
        bucket = candidate;
        break;
      }
    }

    if (!bucket){
      bucket = {
        section: item.section || "",
        redIndex: item.red_index ?? null,
        widthValues: [],
        heightValues: [],
        widthDisplays: [],
        heightDisplays: [],
        widthDecimals: 0,
        heightDecimals: 0,
        widthReference: widthNum != null ? widthNum : 0,
        heightReference: heightNum != null ? heightNum : 0,
        hasNumeric: widthNum != null && heightNum != null,
        widthKey: widthKey || null,
        heightKey: heightKey || null,
        qty: 0,
        invalid: false,
        orderIds: new Set(),
        clients: new Set(),
        positions: new Set(),
        originRows: [],
        areaTotal: 0,
      };
      buckets.push(bucket);
    }

    if (widthNum != null){
      bucket.widthValues.push(widthNum);
      bucket.widthDecimals = Math.max(bucket.widthDecimals, widthDecimals);
      bucket.widthReference = calculateRepresentativeNumber(bucket.widthValues, bucket.widthDecimals);
    }
    if (heightNum != null){
      bucket.heightValues.push(heightNum);
      bucket.heightDecimals = Math.max(bucket.heightDecimals, heightDecimals);
      bucket.heightReference = calculateRepresentativeNumber(bucket.heightValues, bucket.heightDecimals);
    }

    if (widthNum != null && heightNum != null){
      bucket.hasNumeric = true;
    }

    if (item.widthDisplay) bucket.widthDisplays.push(item.widthDisplay);
    if (item.heightDisplay) bucket.heightDisplays.push(item.heightDisplay);
    bucket.qty += Number(item.qty || 0);
    bucket.invalid = bucket.invalid || !!item.invalidDimension;
    if (item.orderId) bucket.orderIds.add(item.orderId);
    if (item.client) bucket.clients.add(item.client);
    const positionValue = pickPositionValue(item);
    if (positionValue) bucket.positions.add(positionValue);
    bucket.originRows.push(buildOriginRowPayload(item));
    const areaValue = toNumericArea(item.area ?? item.area_m2 ?? item.areaM2);
    if (areaValue != null){
      bucket.areaTotal += areaValue;
    }
  });

  return buckets.map(bucket => {
    const widthNumeric = bucket.widthValues.length ? calculateRepresentativeNumber(bucket.widthValues, bucket.widthDecimals) : null;
    const heightNumeric = bucket.heightValues.length ? calculateRepresentativeNumber(bucket.heightValues, bucket.heightDecimals) : null;
    const widthDisplay = selectRepresentativeDisplay(bucket.widthDisplays, widthNumeric, bucket.widthDecimals);
    const heightDisplay = selectRepresentativeDisplay(bucket.heightDisplays, heightNumeric, bucket.heightDecimals);
    const orderIdsList = Array.from(bucket.orderIds).filter(Boolean);
    const clientsList = Array.from(bucket.clients).filter(Boolean);
    const positionsList = dedupeAndSortPositions(Array.from(bucket.positions || []));
    const position = positionsList.length === 1
      ? positionsList[0]
      : (positionsList.length ? "(Grouped)" : "");
    const orderId = orderIdsList.join(", ");
    const client = clientsList.length === 1 ? clientsList[0] : (clientsList.length ? "(Mixed)" : "—");
    return {
      width: widthNumeric,
      height: heightNumeric,
      widthDisplay,
      heightDisplay,
      qty: bucket.qty,
      invalid: bucket.invalid,
      orderId,
      client,
      position,
      originRows: bucket.originRows.map(origin => ({ ...origin })),
      positions: positionsList,
      sortWidth: Number.isFinite(widthNumeric) ? widthNumeric : Number.POSITIVE_INFINITY,
      sortHeight: Number.isFinite(heightNumeric) ? heightNumeric : Number.POSITIVE_INFINITY,
      sortWidthKey: String(widthDisplay || "").toLowerCase(),
      sortHeightKey: String(heightDisplay || "").toLowerCase(),
      area: Number.isFinite(bucket.areaTotal) ? Number(bucket.areaTotal.toFixed(3)) : null,
    };
  });
}

function generateMotherSheet(rows, options){
  const {
    restartPerGroup,
    decimalSeparator,
    normalizeLPtoG,
    headerOverrides = {},
    groupDimensions = false,
    mergeAcrossOrders = false,
  } = options;
  const grouped = new Map();
  const orderSet = new Set();
  const clientSet = new Set();
  (rows || []).forEach(row => {
    if (!row) return;
    const key = (row.composition_raw || row.composition || "").trim() || "(Header not set)";
    if (!grouped.has(key)) grouped.set(key, []);
    const normalizedRow = {
      ...row,
      widthDisplay: row.widthDisplay ?? (Number.isFinite(row.width) ? String(row.width) : row.widthDisplay),
      heightDisplay: row.heightDisplay ?? (Number.isFinite(row.height) ? String(row.height) : row.heightDisplay),
    };
    grouped.get(key).push(normalizedRow);
    if (row.orderId) orderSet.add(row.orderId);
    const clientValue = row.client && String(row.client).trim();
    if (clientValue) clientSet.add(clientValue);
  });

  const sortCollapsedEntries = (entries)=>{
    return entries.sort((a, b)=>{
      const aWidthFinite = Number.isFinite(a.sortWidth);
      const bWidthFinite = Number.isFinite(b.sortWidth);
      if (aWidthFinite && bWidthFinite && a.sortWidth !== b.sortWidth){
        return a.sortWidth - b.sortWidth;
      }
      if (aWidthFinite && !bWidthFinite) return -1;
      if (!aWidthFinite && bWidthFinite) return 1;

      const aHeightFinite = Number.isFinite(a.sortHeight);
      const bHeightFinite = Number.isFinite(b.sortHeight);
      if (aHeightFinite && bHeightFinite && a.sortHeight !== b.sortHeight){
        return a.sortHeight - b.sortHeight;
      }
      if (aHeightFinite && !bHeightFinite) return -1;
      if (!aHeightFinite && bHeightFinite) return 1;

      if (a.sortWidthKey !== b.sortWidthKey){
        return a.sortWidthKey.localeCompare(b.sortWidthKey);
      }
      if (a.sortHeightKey !== b.sortHeightKey){
        return a.sortHeightKey.localeCompare(b.sortHeightKey);
      }
      return 0;
    });
  };

  const groups = [];
  let runningIndex = 1;
  let totalLines = 0;

  for (const [raw, items] of grouped.entries()){
    const override = headerOverrides[raw];
    const display = override && override.trim().length ? override.trim() : normalizeHeaderForDisplay(raw, normalizeLPtoG);
    let localIndex = restartPerGroup ? 1 : runningIndex;
    const assignIndex = ()=> restartPerGroup ? localIndex++ : runningIndex++;

    const groupLines = [];
    const sections = [];

    if (mergeAcrossOrders){
      let working = [];
      if (groupDimensions){
        working = collapseGroupDimensions(items, DIMENSION_TOLERANCE_MM);
        sortCollapsedEntries(working);
      }else{
        working = items.map(item => {
          const positionValue = pickPositionValue(item);
          return {
            width: item.width,
            height: item.height,
            widthDisplay: item.widthDisplay,
            heightDisplay: item.heightDisplay,
            qty: item.qty ?? 0,
            invalid: !!item.invalidDimension,
            orderId: item.orderId || "",
            client: item.client && String(item.client).trim().length ? item.client : UNKNOWN_CLIENT_LABEL,
            position: positionValue || "",
            positions: positionValue ? [positionValue] : [],
            sortWidth: Number.isFinite(item.width) ? item.width : Number.POSITIVE_INFINITY,
            sortHeight: Number.isFinite(item.height) ? item.height : Number.POSITIVE_INFINITY,
            sortWidthKey: String(item.widthDisplay || "").toLowerCase(),
            sortHeightKey: String(item.heightDisplay || "").toLowerCase(),
            danko: item.danko || null,
            area: item.area ?? null,
            areaDisplay: item.areaDisplay ?? null,
            areaSource: item.areaSource ?? null,
            areaRaw: item.areaRaw ?? null,
            originRows: [buildOriginRowPayload(item)],
          };
        });
      }

      working.forEach(entry => {
        const idx = assignIndex();
        const line = {
          idx,
          display_no: idx,
          width: entry.width,
          height: entry.height,
          widthDisplay: entry.widthDisplay,
          heightDisplay: entry.heightDisplay,
          qty: entry.qty ?? 0,
          orderId: entry.orderId || "",
          client: entry.client || UNKNOWN_CLIENT_LABEL,
          composition_raw: raw,
          composition_display: display,
          invalid: !!entry.invalid,
          danko: entry.danko || null,
          area: entry.area ?? null,
          areaDisplay: entry.areaDisplay ?? null,
          areaSource: entry.areaSource ?? null,
          areaRaw: entry.areaRaw ?? null,
          position: entry.position || "",
          positions: Array.isArray(entry.positions) && entry.positions.length
            ? entry.positions.slice()
            : (entry.position ? [entry.position] : []),
          originRows: Array.isArray(entry.originRows) && entry.originRows.length
            ? entry.originRows.map(origin => ({ ...origin }))
            : [buildOriginRowPayload(entry)],
        };
        groupLines.push(line);
      });

      if (restartPerGroup) runningIndex = localIndex;
      totalLines += groupLines.length;
      const groupAreaValue = computeGroupArea(groupLines);
      const groupAreaDisplay = groupAreaValue != null ? `${formatM2(groupAreaValue, decimalSeparator)} m²` : null;
      const groupHeaderText = groupAreaDisplay ? `${display} — Area: ${groupAreaDisplay}` : display;
      groups.push({
        raw,
        display,
        headerText: groupHeaderText,
        lines: groupLines,
        sections: null,
        areaValue: groupAreaValue,
        areaDisplay: groupAreaDisplay,
        areaSource: groupAreaDisplay ? "pdf" : null,
      });
      continue;
    }

    const orderBuckets = new Map();
    items.forEach(item => {
      const orderKey = (item.orderId || "").trim() || "—";
      if (!orderBuckets.has(orderKey)) orderBuckets.set(orderKey, []);
      orderBuckets.get(orderKey).push(item);
    });

    for (const [orderKey, orderItems] of orderBuckets.entries()){
      const clientName = orderItems.find(row => row.client && String(row.client).trim().length)?.client || "";
      const clientDisplay = clientName && clientName.trim().length ? clientName.trim() : UNKNOWN_CLIENT_LABEL;
      let entryList = [];
      if (groupDimensions){
        entryList = collapseGroupDimensions(orderItems, DIMENSION_TOLERANCE_MM).map(entry => {
          entry.orderId = orderKey;
          entry.client = clientDisplay;
          return entry;
        });
        sortCollapsedEntries(entryList);
      }else{
        entryList = orderItems.map(item => {
          const positionValue = pickPositionValue(item);
          return {
            width: item.width,
            height: item.height,
            widthDisplay: item.widthDisplay,
            heightDisplay: item.heightDisplay,
            qty: item.qty ?? 0,
            invalid: !!item.invalidDimension,
            orderId: item.orderId || orderKey,
            client: item.client && String(item.client).trim().length ? item.client : clientDisplay,
            position: positionValue || "",
            positions: positionValue ? [positionValue] : [],
            sortWidth: Number.isFinite(item.width) ? item.width : Number.POSITIVE_INFINITY,
            sortHeight: Number.isFinite(item.height) ? item.height : Number.POSITIVE_INFINITY,
            sortWidthKey: String(item.widthDisplay || "").toLowerCase(),
            sortHeightKey: String(item.heightDisplay || "").toLowerCase(),
            danko: item.danko || null,
            area: item.area ?? null,
            areaDisplay: item.areaDisplay ?? null,
            areaSource: item.areaSource ?? null,
            areaRaw: item.areaRaw ?? null,
            originRows: [buildOriginRowPayload(item)],
          };
        });
      }

      const section = {
        orderId: orderKey,
        client: clientDisplay,
        orderHeaderText: `[Order ${orderKey} — ${clientDisplay}]`,
        lines: [],
      };

      entryList.forEach(entry => {
        const idx = assignIndex();
        const line = {
          idx,
          display_no: idx,
          width: entry.width,
          height: entry.height,
          widthDisplay: entry.widthDisplay,
          heightDisplay: entry.heightDisplay,
          qty: entry.qty ?? 0,
          orderId: entry.orderId || orderKey,
          client: entry.client || clientDisplay,
          composition_raw: raw,
          composition_display: display,
          invalid: !!entry.invalid,
          danko: entry.danko || null,
          area: entry.area ?? null,
          areaDisplay: entry.areaDisplay ?? null,
          areaSource: entry.areaSource ?? null,
          areaRaw: entry.areaRaw ?? null,
          position: entry.position || "",
          positions: Array.isArray(entry.positions) && entry.positions.length
            ? entry.positions.slice()
            : (entry.position ? [entry.position] : []),
          originRows: Array.isArray(entry.originRows) && entry.originRows.length
            ? entry.originRows.map(origin => ({ ...origin }))
            : [buildOriginRowPayload(entry)],
        };
        section.lines.push(line);
        groupLines.push(line);
      });
      sections.push(section);
    }

    if (restartPerGroup) runningIndex = localIndex;
    totalLines += groupLines.length;
    const groupAreaValue = computeGroupArea(groupLines);
    const groupAreaDisplay = groupAreaValue != null ? `${formatM2(groupAreaValue, decimalSeparator)} m²` : null;
    const groupHeaderText = groupAreaDisplay ? `${display} — Area: ${groupAreaDisplay}` : display;
    groups.push({
      raw,
      display,
      headerText: groupHeaderText,
      lines: groupLines,
      sections,
      areaValue: groupAreaValue,
      areaDisplay: groupAreaDisplay,
      areaSource: groupAreaDisplay ? "pdf" : null,
    });
  }

  const orders = Array.from(orderSet).filter(Boolean);
  const clients = Array.from(clientSet);
  const clientLabel = clients.length === 1 ? clients[0] : (clients.length ? "(Mixed)" : UNKNOWN_CLIENT_LABEL);
  const today = new Date();
  const headerLine = `Mother Sheet – Client: ${clientLabel || UNKNOWN_CLIENT_LABEL} | Orders: ${orders.length ? orders.join(", ") : "—"} | Date: ${today.toLocaleDateString()}`;

  const linesOut = [headerLine, ""];
  groups.forEach((group, groupIndex) => {
    const groupHeader = group && group.headerText ? group.headerText : ((group && group.display) ? group.display : "(Header not set)");
    linesOut.push(groupHeader);
    if (!mergeAcrossOrders && group.sections && group.sections.length){
      group.sections.forEach((section, sectionIndex) => {
        const orderHeaderText = typeof section.orderHeaderText === "string" && section.orderHeaderText.trim().length
          ? section.orderHeaderText
          : (() => {
              const orderLabel = section.orderId && section.orderId.trim().length ? section.orderId : "—";
              const clientLabelForSection = section.client && section.client.trim().length ? section.client : UNKNOWN_CLIENT_LABEL;
              return `[Order ${orderLabel} — ${clientLabelForSection}]`;
            })();
        linesOut.push(orderHeaderText);
        section.lines.forEach(line => {
          const widthText = formatProcessingNumber(line.width, line.widthDisplay, decimalSeparator);
          const heightText = formatProcessingNumber(line.height, line.heightDisplay, decimalSeparator);
          const qtyText = Number(line.qty || 0);
          const warning = line.invalid ? "  ⚠" : "";
          const approx = line.danko && line.danko.changed ? " ≈" : "";
          linesOut.push(`${line.idx} – ${widthText} × ${heightText}${approx} × ${qtyText}${warning}`);
          if (line.danko && line.danko.changed && line.danko.original){
            linesOut.push(`   (Rounded from ${line.danko.original.replace(/x/g, "×")})`);
          }
        });
        if (sectionIndex < group.sections.length - 1){
          linesOut.push("");
        }
      });
    }else{
      group.lines.forEach(line => {
        const widthText = formatProcessingNumber(line.width, line.widthDisplay, decimalSeparator);
        const heightText = formatProcessingNumber(line.height, line.heightDisplay, decimalSeparator);
        const qtyText = Number(line.qty || 0);
        const warning = line.invalid ? "  ⚠" : "";
        const approx = line.danko && line.danko.changed ? " ≈" : "";
        linesOut.push(`${line.idx} – ${widthText} × ${heightText}${approx} × ${qtyText}${warning}`);
        if (line.danko && line.danko.changed && line.danko.original){
          linesOut.push(`   (Rounded from ${line.danko.original.replace(/x/g, "×")})`);
        }
      });
    }
    if (groupIndex < groups.length - 1){
      linesOut.push("");
    }
  });

  const meta = {
    clientLabel,
    orders,
    date: today,
    rows: totalLines,
    decimalSeparator,
  };

  return {
    text: linesOut.join("\n").trim(),
    groups,
    meta,
  };
}

function convertOrderToProcessingEntry(order){
  const id = Number(order.id) || order.id;
  const orderSource = String(order.source || "pdf").toLowerCase();
  const orderLabel = (Array.isArray(order.order_numbers) && order.order_numbers.length)
    ? order.order_numbers.join(", ")
    : (order.order_number || `#${order.id}`);
  const client = order.client_name || order.clientName || order.client || order.client_hint || "—";
  const rows = (order.rows || []).map((row, idx) => {
    const dimension = row.dimension || (
      row.width_mm != null && row.height_mm != null
        ? `${row.width_mm}x${row.height_mm}`
        : ""
    );
    const parsed = parseDimensionTokens(dimension);
    const areaRaw = row.final_area_m2 ?? row.area_display ?? row.area ?? row.area_m2 ?? row.areaM2 ?? null;
    const areaValue = toNumericArea(areaRaw);
    const areaDisplay = areaRaw != null ? String(areaRaw) : (areaValue != null ? areaValue.toFixed(3) : null);
    const positionValue = [
      row.position,
      row.position_label,
      row.positionLabel,
      row.pos,
      row.position_display,
    ].map(value => (value == null ? "" : String(value).trim())).find(value => value.length) || "";
    return {
      key: row.id || `${order.id}-${idx}`,
      composition_raw: (row.type || row.glass_type || "").trim() || "(Header not set)",
      width: parsed.width,
      height: parsed.height,
      widthDisplay: parsed.widthDisplay,
      heightDisplay: parsed.heightDisplay,
      qty: Number(row.quantity || 0) || 0,
      orderId: orderLabel,
      client,
      position: positionValue,
      section: row.section || "",
      client_position: row.client_position || "",
      red_index: row.index_number ?? row.red_index ?? null,
      row_notes: row.notes || "",
      invalidDimension: parsed.invalid,
      area: areaValue != null ? areaValue : null,
      areaDisplay,
      areaSource: areaRaw != null ? orderSource : null,
      areaRaw: areaRaw != null ? String(areaRaw) : null,
      source: orderSource,
    };
  });
  return {
    id,
    orderLabel,
    client,
    createdAt: order.created_at || null,
    rows,
  };
}

function deriveRawDimension(row){
  const candidates = [
    row.rawDimension,
    row.dimension,
    row.dimension_display,
    row.dim,
    row.widthDisplay && row.heightDisplay ? `${row.widthDisplay}x${row.heightDisplay}` : null,
    Number.isFinite(row.width) && Number.isFinite(row.height) ? `${Math.round(row.width)}x${Math.round(row.height)}` : null,
  ];
  const found = candidates.find(value => value && String(value).trim().length);
  return found ? String(found).replace(/\s+/g, "") : "";
}

function isProcessingRoundingActive(){
  return !!(appState.processing.options.autoDanko || appState.processing.rounding.manualApplied);
}

function applyProcessingRoundingToRows(){
  const rows = Array.isArray(appState.processing.rows) ? appState.processing.rows : [];
  const active = isProcessingRoundingActive();
  rows.forEach(row => {
    if (!row.__original){
      const dimension = deriveRawDimension(row);
      const widthNumeric = Number.isFinite(row.width) ? Number(row.width) : null;
      const heightNumeric = Number.isFinite(row.height) ? Number(row.height) : null;
      const pdfArea = toNumericArea(row.areaDisplay ?? row.areaRaw ?? row.area ?? row.area_m2 ?? row.areaM2 ?? null);
      const fallbackArea = (widthNumeric != null && heightNumeric != null)
        ? Number(((widthNumeric * heightNumeric) / 1_000_000).toFixed(3))
        : (Number.isFinite(row.area) ? Number(row.area) : null);
      const resolvedArea = pdfArea != null ? pdfArea : fallbackArea;
      const areaDisplay = row.areaDisplay != null
        ? String(row.areaDisplay)
        : (row.areaRaw != null ? String(row.areaRaw) : (resolvedArea != null ? resolvedArea.toFixed(3) : null));
      row.__original = {
        width: widthNumeric,
        height: heightNumeric,
        widthDisplay: row.widthDisplay != null ? String(row.widthDisplay) : (widthNumeric != null ? String(widthNumeric) : null),
        heightDisplay: row.heightDisplay != null ? String(row.heightDisplay) : (heightNumeric != null ? String(heightNumeric) : null),
        area: resolvedArea,
        pdfAreaValue: pdfArea,
        areaDisplay,
        dimension,
      };
    }
    const original = row.__original;
    const originalArea = original.pdfAreaValue != null ? original.pdfAreaValue : original.area;
    const originalAreaDisplay = original.areaDisplay != null
      ? original.areaDisplay
      : (originalArea != null ? originalArea.toFixed(3) : null);
    const dimensionString = original.dimension || deriveRawDimension(row) || "";
    const danko = window.applyDankoRuleToDim(dimensionString);
    const tooltipBase = dimensionString ? dimensionString.replace(/x/g, "×") : "";
    row.danko = {
      ...danko,
      original: danko.original || dimensionString,
      tooltip: danko.changed && tooltipBase ? `Rounded from ${tooltipBase}` : (tooltipBase ? `Original ${tooltipBase}` : ""),
    };
    if (active && danko && danko.changed){
      if (Number.isFinite(danko.w)){
        row.width = danko.w;
        row.widthDisplay = String(danko.w);
      }else{
        row.width = original.width;
        row.widthDisplay = original.widthDisplay;
      }
      if (Number.isFinite(danko.h)){
        row.height = danko.h;
        row.heightDisplay = String(danko.h);
      }else{
        row.height = original.height;
        row.heightDisplay = original.heightDisplay;
      }
      row.danko.changed = true;
    }else{
      row.width = original.width;
      row.height = original.height;
      row.widthDisplay = original.widthDisplay;
      row.heightDisplay = original.heightDisplay;
      if (row.danko){
        row.danko.changed = false;
      }
    }
    row.area = originalArea != null ? originalArea : (Number.isFinite(row.area) ? row.area : 0);
    row.areaDisplay = originalAreaDisplay;
    if (!row.areaSource && original.pdfAreaValue != null){
      row.areaSource = "pdf";
    }
  });
}

function deriveLabelDimension(row){
  const candidates = [
    row.dimension,
    row.dimension_display,
    row.dim,
  ];
  for (const candidate of candidates){
    if (candidate && String(candidate).trim().length){
      return String(candidate).trim().replace(/\s+/g, " ");
    }
  }
  const widthDisplay = row.widthDisplay != null ? String(row.widthDisplay).trim() : "";
  const heightDisplay = row.heightDisplay != null ? String(row.heightDisplay).trim() : "";
  const widthNumeric = Number.isFinite(row.width) ? String(row.width) : "";
  const heightNumeric = Number.isFinite(row.height) ? String(row.height) : "";
  if (widthDisplay && heightDisplay){
    return `${widthDisplay} × ${heightDisplay}`;
  }
  if (widthDisplay && heightNumeric){
    return `${widthDisplay} × ${heightNumeric}`;
  }
  if (widthNumeric && heightDisplay){
    return `${widthNumeric} × ${heightDisplay}`;
  }
  if (widthNumeric && heightNumeric){
    return `${widthNumeric} × ${heightNumeric}`;
  }
  if (widthDisplay || widthNumeric){
    return `${widthDisplay || widthNumeric} × ?`;
  }
  if (heightDisplay || heightNumeric){
    return `? × ${heightDisplay || heightNumeric}`;
  }
  return "";
}

function normalizePositionString(value){
  if (value == null) return "";
  return String(value).trim();
}

function pickPositionValue(row){
  if (!row || typeof row !== "object") return "";
  const candidates = [
    row.position,
    row.position_label,
    row.positionLabel,
    row.pos,
    row.position_display,
  ];
  for (const candidate of candidates){
    const normalized = normalizePositionString(candidate);
    if (normalized) return normalized;
  }
  return "";
}

function parsePositionParts(value){
  const normalized = normalizePositionString(value);
  if (!normalized) return { original: "", primary: null, secondary: null };
  const parts = normalized.split("-");
  const parseNumber = (token)=>{
    if (token == null) return null;
    const num = Number(token);
    return Number.isFinite(num) ? num : null;
  };
  return {
    original: normalized,
    primary: parseNumber(parts[0]),
    secondary: parseNumber(parts[1]),
  };
}

function comparePositionStrings(a, b){
  const pa = parsePositionParts(a);
  const pb = parsePositionParts(b);
  const aHasPrimary = pa.primary != null;
  const bHasPrimary = pb.primary != null;
  if (aHasPrimary && bHasPrimary && pa.primary !== pb.primary){
    return pa.primary - pb.primary;
  }
  if (aHasPrimary && !bHasPrimary) return -1;
  if (!aHasPrimary && bHasPrimary) return 1;

  const aHasSecondary = pa.secondary != null;
  const bHasSecondary = pb.secondary != null;
  if (aHasSecondary && bHasSecondary && pa.secondary !== pb.secondary){
    return pa.secondary - pb.secondary;
  }
  if (aHasSecondary && !bHasSecondary) return -1;
  if (!aHasSecondary && bHasSecondary) return 1;

  return pa.original.localeCompare(pb.original);
}

function dedupeAndSortPositions(values){
  if (!Array.isArray(values) || !values.length) return [];
  const seen = new Set();
  const unique = [];
  values.forEach(value=>{
    const normalized = normalizePositionString(value);
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    unique.push(normalized);
  });
  unique.sort(comparePositionStrings);
  return unique;
}

function summarizePositionsList(values, limit = 6){
  const unique = dedupeAndSortPositions(values);
  if (!unique.length) return "";
  if (unique.length === 1) return unique[0];
  const display = unique.slice(0, limit);
  let summary = display.join(", ");
  const remaining = unique.length - display.length;
  if (remaining > 0){
    summary += ` (+${remaining})`;
  }
  return summary;
}

function collectProcessingLinePositions(line){
  if (!line || typeof line !== "object") return [];
  const bucket = [];
  if (Array.isArray(line.positions) && line.positions.length){
    bucket.push(...line.positions);
  }
  const fallback = pickPositionValue(line);
  if (fallback && fallback !== "(Grouped)") bucket.push(fallback);
  return dedupeAndSortPositions(bucket);
}

function collectRowPositions(row){
  if (!row || typeof row !== "object") return [];
  const bucket = [];
  if (Array.isArray(row.processing_positions) && row.processing_positions.length){
    bucket.push(...row.processing_positions);
  }else if (Array.isArray(row.positions) && row.positions.length){
    bucket.push(...row.positions);
  }
  const fallback = pickPositionValue(row);
  if (fallback) bucket.push(fallback);
  return dedupeAndSortPositions(bucket);
}

function formatProcessingPositionsSummary(row){
  if (!row) return "";
  const preset = normalizePositionString(row.processing_position_summary);
  if (preset) return preset;
  const list = collectRowPositions(row);
  return summarizePositionsList(list);
}

function buildOriginRowPayload(item){
  if (!item) return {
    position: "—",
    quantity: 1,
    orderId: "",
    client: "",
    widthOriginal: null,
    heightOriginal: null,
  };
  const quantityRaw = Number(item.qty ?? item.quantity ?? 0);
  const quantity = Number.isFinite(quantityRaw) && quantityRaw > 0 ? quantityRaw : 1;
  const original = item.__original || {};
  const positionValue = pickPositionValue(item) || "—";
  return {
    position: positionValue,
    section: item.section || "",
    client_position: item.client_position || "",
    red_index: item.red_index ?? null,
    row_notes: item.row_notes || "",
    source_row_id: item.key ?? null,
    positions: positionValue && positionValue !== "(Grouped)" ? [positionValue] : [],
    quantity,
    orderId: (item.orderId && String(item.orderId).trim()) || "",
    client: (item.client && String(item.client).trim()) || "",
    widthOriginal: Number.isFinite(original.width) ? original.width : (Number.isFinite(item.width) ? item.width : null),
    heightOriginal: Number.isFinite(original.height) ? original.height : (Number.isFinite(item.height) ? item.height : null),
    rawDimension: item.rawDimension || original.dimension || "",
  };
}

function formatProcessingDimensionLabel(line){
  if (!line) return "—";
  const widthDisplay = line.widthDisplay != null ? String(line.widthDisplay).trim() : "";
  const heightDisplay = line.heightDisplay != null ? String(line.heightDisplay).trim() : "";
  const widthText = widthDisplay.length ? widthDisplay : (Number.isFinite(line.width) ? String(line.width) : "?");
  const heightText = heightDisplay.length ? heightDisplay : (Number.isFinite(line.height) ? String(line.height) : "?");
  return `${widthText || "?"} × ${heightText || "?"}`;
}

function buildProcessingLabelRows(line, orderId, clientLabel){
  if (!line) return [];
  const orderNumber = (line.orderId && String(line.orderId).trim()) || orderId || "—";
  const clientValue = (line.client && String(line.client).trim()) || clientLabel || UNKNOWN_CLIENT_LABEL;
  const qty = Number(line.qty || line.quantity || 0);
  const msIndex = Number(line.idx);
  const rawArea = Number(line.area ?? line.areaValue ?? line.areaRaw);
  const areaValue = Number.isFinite(rawArea) ? rawArea : 0;
  const positionsList = collectProcessingLinePositions(line);
  const fallbackPosition = normalizePositionString(line.position);
  const dimensionLabel = formatProcessingDimensionLabel(line);
  const rowsOut = [];
  const originRows = Array.isArray(line.originRows) && line.originRows.length
    ? line.originRows
    : [{
        position: fallbackPosition && fallbackPosition !== "(Grouped)" ? fallbackPosition : (positionsList[0] || line.position || "—"),
        quantity: line.qty ?? 1,
        orderId: orderNumber,
        client: clientValue,
      }];
  originRows.forEach(origin=>{
    const uniquePositions = dedupeAndSortPositions(origin.positions || [origin.position]);
    const targets = uniquePositions.length
      ? uniquePositions
      : (origin.position && origin.position !== "(Grouped)" ? [origin.position] : []);
    const positionTargets = targets.length ? targets : ["—"];
    const perPositionQtyRaw = Number(origin.quantity);
    const perPositionQty = Number.isFinite(perPositionQtyRaw) && perPositionQtyRaw > 0 ? perPositionQtyRaw : 1;
    positionTargets.forEach(positionValue=>{
      const safePosition = positionValue || "—";
      for (let i = 0; i < perPositionQty; i++){
        rowsOut.push({
          order_number: origin.orderId || orderNumber,
          orderId: origin.orderId || orderNumber,
          client: origin.client || clientValue,
          type: line.composition_display || line.composition_raw || "",
          dimension: dimensionLabel,
          dimension_display: dimensionLabel,
          widthDisplay: line.widthDisplay,
          heightDisplay: line.heightDisplay,
          quantity: 1,
          qty: 1,
          position: safePosition,
          area: areaValue,
          source: "processing",
          ms_index: Number.isFinite(msIndex) ? msIndex : null,
          processing_positions: [safePosition],
        });
      }
    });
  });
  return rowsOut;
}

async function buildProcessingPdfBlob(){
  const preview = appState.processing.preview;
  const plainText = typeof preview?.text === "string" ? preview.text : "";
  if (!plainText.trim()){
    throw new Error("Nothing to export.");
  }
  await ensurePdfLib();
  const { PDFDocument, StandardFonts, rgb } = PDFLib;
  const mmToPt = 2.83464567;
  const pdfDoc = await PDFDocument.create();
  const regularFont = await pdfDoc.embedFont(StandardFonts.Helvetica);

  const layout = {
    pageWidth: 210 * mmToPt,
    pageHeight: 297 * mmToPt,
    margin: 12.7 * mmToPt,
    columnGap: 8 * mmToPt,
    fontSize: 11,
  };
  layout.lineHeight = layout.fontSize * 1.3;
  layout.contentWidth = layout.pageWidth - (layout.margin * 2);
  layout.columnWidth = (layout.contentWidth - layout.columnGap) / 2;
  layout.columnTop = layout.pageHeight - layout.margin;
  layout.columnBottom = layout.margin;

  const textColor = rgb(0, 0, 0);
  const toPdfText = value => String(value ?? "")
    .replace(/\u26a0/g, "!")
    .replace(/[^\x09\x0a\x0d\x20-\x7e\u00a0-\u00ff\u2013\u2014\u2018-\u201d\u2022\u2026\u20ac]/g, "?");
  const textWidth = text => regularFont.widthOfTextAtSize(toPdfText(text), layout.fontSize);

  let page = null;
  let columnIndex = 0;
  let cursorY = layout.columnTop - layout.fontSize;

  const currentColumnX = () => layout.margin + (columnIndex * (layout.columnWidth + layout.columnGap));
  const addPage = () => {
    page = pdfDoc.addPage([layout.pageWidth, layout.pageHeight]);
    columnIndex = 0;
    cursorY = layout.columnTop - layout.fontSize;
  };
  const advanceColumn = () => {
    if (columnIndex === 0){
      columnIndex = 1;
      cursorY = layout.columnTop - layout.fontSize;
    }else{
      addPage();
    }
  };
  const ensureLineSpace = () => {
    if (cursorY < layout.columnBottom){
      advanceColumn();
    }
  };
  const wrapLine = line => {
    const text = String(line ?? "");
    if (text === "") return [""];
    const indent = (text.match(/^\s*/) || [""])[0];
    const maxWidth = layout.columnWidth;
    if (textWidth(text) <= maxWidth) return [text];

    const words = text.trimEnd().split(/(\s+)/);
    const wrapped = [];
    let current = "";
    words.forEach(part => {
      if (!part) return;
      const next = current ? `${current}${part}` : part;
      if (current && textWidth(next) > maxWidth){
        wrapped.push(current.trimEnd());
        current = indent && part.trim() ? `${indent}${part.trimStart()}` : part.trimStart();
      }else{
        current = next;
      }
      while (current && textWidth(current) > maxWidth){
        let cut = current.length;
        while (cut > 1 && textWidth(current.slice(0, cut)) > maxWidth){
          cut -= 1;
        }
        wrapped.push(current.slice(0, cut));
        current = `${indent}${current.slice(cut)}`;
      }
    });
    if (current || !wrapped.length){
      wrapped.push(current.trimEnd());
    }
    return wrapped;
  };
  const drawTextLine = line => {
    ensureLineSpace();
    if (line !== ""){
      page.drawText(toPdfText(line), {
        x: currentColumnX(),
        y: cursorY,
        size: layout.fontSize,
        font: regularFont,
        color: textColor,
      });
    }
    cursorY -= layout.lineHeight;
  };

  addPage();
  plainText.split(/\r?\n/).forEach(sourceLine => {
    wrapLine(sourceLine).forEach(drawTextLine);
  });

  const pdfBytes = await pdfDoc.save();
  return new Blob([pdfBytes], { type: "application/pdf" });
}

function normalizeTypeKey(str) {
  return (str || "")
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/\++/g, "+");
}

function applyTypeCorrections(rawType) {
  const key = normalizeTypeKey(rawType);
  for (const entry of TYPE_CORRECTIONS) {
    if (normalizeTypeKey(entry.raw) === key) {
      return entry.corrected;
    }
  }
  return rawType;
}

function fixGlassTypos(text, rawSource){
  if (!text) return text;
  const replaced = text.replace(/stainat/gi, "satinat");
  if (replaced !== text){
    console.log("[GlassNormalize] raw:", rawSource, "→ normalized:", replaced);
  }
  return replaced;
}

function normalizeGlassKey(raw){
  const original = raw == null ? "" : String(raw);
  let str = original.toLowerCase().trim().replace(/^\s*\d+\s*vetri?\s*/i, "").replace(/\bvetri?\b/gi, "");
  if (!str) return "";
  str = str.replace(/\s+/g, " ").trim();
  str = fixGlassTypos(str, original);
  return str.replace(/\s+/g, "");
}

function dedupeAndSortOrderNumbers(values){
  const unique = [];
  (values || []).forEach(num=>{
    const normalized = String(num || "").trim();
    if (!normalized) return;
    const lower = normalized.toLowerCase();
    if (!unique.some(existing => existing.toLowerCase() === lower)){
      unique.push(normalized);
    }
  });
  return unique;
}

function extractOrderUnits(order){
  return Number(pickFirstNumber([
    order?.units_total,
    order?.total_units,
    order?.parsed_units,
    order?.units,
  ]) || 0);
}

function extractOrderArea(order){
  return Number(pickFirstNumber([
    order?.area_total,
    order?.total_area,
    order?.parsed_area,
    order?.area,
  ]) || 0);
}

function createOrderMetadata(options = {}){
  return {
    orderNumber: options.orderNumber || "—",
    clientName: options.clientName || "—",
    createdAt: options.createdAt || new Date().toISOString(),
    units: Number(options.units || 0),
    area: Number(options.area || 0),
    amount: Number(options.amount || 0),
  };
}

function ensureJobOrders(job){
  if (!job) return;
  if (!Array.isArray(job.orderNumbers)){
    job.orderNumbers = [];
  }
  if (!Array.isArray(job.orders)){
    job.orders = [];
  }
  const metaMap = new Map();
  job.orders.forEach(entry=>{
    if (!entry) return;
    const key = String(entry.orderNumber || "").trim().toLowerCase();
    if (!key) return;
    metaMap.set(key, {
      orderNumber: entry.orderNumber || "—",
      clientName: entry.clientName || entry.client || "—",
      createdAt: entry.createdAt || entry.date || job.createdAt || new Date().toISOString(),
      units: Number(entry.units || 0),
      area: Number(entry.area || 0),
      amount: Number(entry.amount || entry.total || 0),
    });
  });
  const fallbackClient = formatInvoiceClientLabel(job) || job.client || "—";
  const fallbackDate = job.createdAt || new Date().toISOString();
  job.orders = job.orderNumbers.map(orderNumber=>{
    const key = String(orderNumber || "").trim();
    const normalizedKey = key.toLowerCase();
    const existing = metaMap.get(normalizedKey);
    if (existing){
      if (!existing.clientName) existing.clientName = fallbackClient;
      if (!existing.createdAt) existing.createdAt = fallbackDate;
      existing.orderNumber = key || existing.orderNumber || "—";
      return existing;
    }
    return createOrderMetadata({
      orderNumber: key || "—",
      clientName: fallbackClient,
      createdAt: fallbackDate,
    });
  });
}

function updateJobOrdersFromCalc(job, calc){
  if (!job) return;
  ensureJobOrders(job);
  const stats = new Map();
  const lines = Array.isArray(calc?.lines) ? calc.lines : Array.isArray(job.calculated?.lines) ? job.calculated.lines : [];
  lines.forEach(line=>{
    const orderKeyRaw = (line.orderId && String(line.orderId).trim()) || (job.orderNumbers && job.orderNumbers[0]) || "";
    const orderKey = orderKeyRaw.toLowerCase();
    if (!orderKey) return;
    if (!stats.has(orderKey)){
      stats.set(orderKey, { units: 0, area: 0, amount: 0 });
    }
    const entry = stats.get(orderKey);
    entry.units += Number(line.quantity || 0);
    const areaValue = Number(line.area || line.lineAreaTotal || 0);
    entry.area += areaValue;
    entry.amount += Number(line.lineTotal || 0);
  });
  const fallbackClient = formatInvoiceClientLabel(job) || job.client || "—";
  const fallbackDate = job.createdAt || new Date().toISOString();
  const normalizedOrders = job.orderNumbers.map(orderNumber=>{
    const normalizedKey = String(orderNumber || "").trim().toLowerCase();
    const existing = (job.orders || []).find(meta => String(meta.orderNumber || "").toLowerCase() === normalizedKey) || createOrderMetadata({
      orderNumber: orderNumber || "—",
      clientName: fallbackClient,
      createdAt: fallbackDate,
    });
    const stat = stats.get(normalizedKey) || { units: 0, area: 0, amount: 0 };
    return {
      ...existing,
      orderNumber: orderNumber || existing.orderNumber || "—",
      clientName: existing.clientName || fallbackClient,
      createdAt: existing.createdAt || fallbackDate,
      units: Number(stat.units || 0),
      area: Number(stat.area || 0),
      amount: Number(stat.amount || 0),
    };
  });
  job.orders = normalizedOrders;
}

function formatInvoiceClientLabel(job){
  if (!job) return "—";
  let source = Array.isArray(job.clients) && job.clients.length
    ? job.clients
    : [];
  if (!source.length && job.client){
    source = [job.client];
  }
  if (!source.length && Array.isArray(job.orders)){
    source = job.orders.map(entry => entry?.clientName).filter(Boolean);
  }
  if (!source.length){
    source = ["—"];
  }
  const normalized = source.map(name => {
    const text = String(name || "").trim();
    return text || "—";
  });
  const unique = [];
  normalized.forEach(name=>{
    const lower = name.toLowerCase();
    if (!unique.some(existing => existing.toLowerCase() === lower)){
      unique.push(name);
    }
  });
  return unique.join(", ");
}

function normalizeOrderKey(numbers, fallback){
  const joined = Array.isArray(numbers) ? numbers.map(v => String(v || "").trim()).filter(Boolean).join(", ") : "";
  const base = joined || (fallback ? String(fallback).trim() : "");
  return base.toLowerCase();
}

function parseInvoiceOrderNumbers(order){
  if (!order) return [];
  if (Array.isArray(order.order_numbers) && order.order_numbers.length){
    return order.order_numbers.map(val => String(val || "").trim()).filter(Boolean);
  }
  const label = getOrderLabel(order);
  if (label && label !== "—"){
    return label.split(/\s*,\s*/).map(token => token.trim()).filter(Boolean);
  }
  const fallback = pickFirstString([order.order_number, order.orderNumber, order.id]);
  return fallback ? [fallback] : [];
}

function tokenizeComposition(typeString){
  const raw = String(typeString || "");
  const compact = raw.replace(/^\s*\d+\s*vetri?\s*/i, "").replace(/\s+/g, " ").trim();
  const withoutMm = compact.replace(/\b\d+\s*mm\b/gi, "").replace(/\bmm\b/gi, "");
  const tokens = withoutMm.split("+").map(t => t.trim()).filter(Boolean);
  return tokens;
}

function stripThermalDescriptors(token){
  return String(token || "")
    .replace(/\b(termico|caldo|c\.?caldo)\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeGlassToken(token, opts = {}){
  const { forcePane = false } = opts;
  let cleaned = stripThermalDescriptors(token);
  if (!cleaned) return "";
  const numericOnly = /^\d+(?:[.,]\d+)?$/.test(cleaned);
  if (numericOnly && forcePane){
    cleaned = `${cleaned}F`;
  }
  return cleaned;
}

function getSpacerKind(typeString){
  const text = (typeString || "").toLowerCase();
  if (text.includes("termico") || text.includes("warm edge") || text.includes("warm-edge") || text.includes("warm") || text.includes("tgi")){
    return "thermal";
  }
  return "normal";
}

function parseInvoiceComposition(typeString){
  const tokens = tokenizeComposition(typeString);
  const panes = [];
  const spacerThicknesses = [];
  const glassTable = appState?.invoices?.priceLists?.glass || {};
  tokens.forEach(token=>{
    const hasThermalWord = /\b(termico|caldo|c\.?caldo)\b/i.test(token);
    const cleanedToken = normalizeGlassToken(token, { forcePane: hasThermalWord });
    if (!cleanedToken) return;
    const numericCandidate = cleanedToken.replace(/\s+/g, "");
    const numeric = Number(numericCandidate.replace(",", "."));
    const normalizedGlass = normalizeGlassKey(cleanedToken);
    if (Number.isFinite(numeric) && !hasThermalWord){
      spacerThicknesses.push(numeric);
    }else if (normalizedGlass && normalizedGlass.length && !normalizedGlass.includes("vetri")){
      panes.push(cleanedToken);
      if (!glassTable.hasOwnProperty(normalizedGlass)){
        // will trigger missing price prompt via lookup later
      }
    }
  });
  return {
    panes,
    spacerThicknesses,
    spacerKind: getSpacerKind(typeString),
  };
}

function mapInvoiceRow(row){
  if (!row) return null;
  const orderId = pickFirstString([
    row.orderId,
    row.order_id,
    row.order_number,
    row.orderNumber,
    row.order,
    row.orderLabel,
  ], null);
  const originalType = pickFirstString([
    row.type,
    row.composition_display,
    row.composition_raw,
    row.glass_type,
    row.glassType,
    row.description,
  ], "");
  const type = applyTypeCorrections(originalType);
  const dimension = deriveLabelDimension(row) || "—";
  const position = pickPositionValue(row) || "—";
  const qty = Math.max(0, Number(pickFirstNumber([
    row.quantity,
    row.qty,
    row.Qty,
    row.QTY,
    row.total_qty,
  ]) || 0));
  const area = (() => {
    const candidate = toNumericArea(row.area ?? row.area_m2 ?? row.areaM2 ?? row.computed_area);
    if (candidate != null) return candidate;
    return computeRowArea(row);
  })();
  return {
    orderId: orderId || null,
    rawType: originalType || null,
    type,
    dimension,
    position,
    quantity: qty,
    area: Number.isFinite(area) ? Number(area) : 0,
  };
}

function buildInvoiceJobFromOrder(order){
  const orderNumbers = dedupeAndSortOrderNumbers(parseInvoiceOrderNumbers(order));
  const normalizedKey = normalizeOrderKey(orderNumbers, order?.id);
  const clientName = pickFirstString([
    order.client,
    order.client_name,
    order.clientName,
    order.client_hint,
    order.customer,
  ], "—");
  const defaultOrderId = orderNumbers[0] || null;
  const mappedRows = Array.isArray(order?.rows)
    ? order.rows.map(row => mapInvoiceRow({
        orderId: row.orderId ?? row.order_id ?? row.order_number ?? defaultOrderId,
        order_number: row.order_number ?? defaultOrderId,
        ...row,
      })).filter(Boolean)
    : [];
  const createdAt = extractOrderDate(order) || new Date();
  const orderEntries = (orderNumbers.length ? orderNumbers : [defaultOrderId || "—"]).map(num => createOrderMetadata({
    orderNumber: num || "—",
    clientName,
    createdAt: createdAt.toISOString(),
    units: extractOrderUnits(order),
    area: extractOrderArea(order),
    amount: 0,
  }));
  return {
    id: order?.id ? `inv-${order.id}` : `inv-${Date.now()}`,
    key: normalizedKey,
    source: String(order?.source || "").toLowerCase() === "manual" ? "manual" : "pdf",
    orderNumbers,
    client: clientName || "—",
    clients: [clientName || "—"],
    orders: orderEntries,
    createdAt: createdAt.toISOString(),
    rows: mappedRows.map(r => ({ ...r })),
    rawRows: mappedRows.map(r => ({ ...r })),
    status: "draft",
    discountMode: "percent",
    discountValue: 0,
    vatRate: 0,
    invoiceNumber: order?.invoiceNumber || null,
    hidden: false,
    totalUnits: 0,
    totalArea: 0,
    totalAmount: 0,
  };
}

function getSpacerPrice(thickness, kind){
  const table = appState.invoices.priceLists.spacer || {};
  if (!Number.isFinite(thickness)){
    return { price: 0, missing: { reason: "thickness", thickness: null, kind } };
  }
  const entry = table[thickness];
  if (!entry){
    return { price: 0, missing: { reason: "thickness", thickness, kind } };
  }
  const normalizedKind = kind === "thermal" ? "thermal" : "normal";
  const candidate = entry[normalizedKind];
  if (!Number.isFinite(candidate)){
    return { price: 0, missing: { reason: "kind", thickness, kind: normalizedKind } };
  }
  return { price: Number(candidate), missing: null };
}

function ensureInvoiceRawRows(job){
  if (!job) return [];
  if (!Array.isArray(job.rawRows)){
    job.rawRows = Array.isArray(job.rows) ? job.rows.map(r => ({ ...r })) : [];
  }
  const defaultOrderId = Array.isArray(job.orderNumbers) && job.orderNumbers.length === 1 ? job.orderNumbers[0] : null;
  if (defaultOrderId){
    job.rawRows.forEach(row=>{
      if (!row || typeof row !== "object") return;
      const hasOrderId = row.orderId || row.order_id || row.order_number || row.orderNumber;
      if (!hasOrderId){
        row.orderId = defaultOrderId;
        row.order_number = defaultOrderId;
      }
    });
  }
  job.rows = job.rawRows;
  return job.rawRows;
}

async function buildInvoiceLinesFromRaw(rawRows, options = {}){
  const promptAllowedForJob = !!options.promptAllowedForJob;
  const allowAi = options.allowAi !== false;
  const groups = new Map();
  (rawRows || []).forEach((row, idx)=>{
    if (!row || typeof row !== "object") return;
    const orderId = pickFirstString([
      row.orderId,
      row.order_id,
      row.order_number,
      row.orderNumber,
      row.order,
      row.orderLabel,
    ], "");
    const normalizedOrder = orderId ? orderId.toLowerCase() : "";
    const originalTypeText = String(row.type || "").trim();
    const rawOriginal = row.rawType != null ? String(row.rawType) : null;
    const correctedTypeText = applyTypeCorrections(originalTypeText);
    if (row.type !== correctedTypeText){
      row.type = correctedTypeText;
    }
    const typeText = correctedTypeText;
    const normalizedType = normalizeGlassKey(typeText) || typeText.toLowerCase();
    const key = `${normalizedOrder}|${normalizedType}`;
    if (!groups.has(key)){
      groups.set(key, {
        displayType: typeText,
        normalizedType,
        orderId: orderId || null,
        qty: 0,
        area: 0,
        rawIndexes: [],
        spacerKind: row.spacerKind || null,
        rawType: rawOriginal && rawOriginal.trim() ? rawOriginal : null,
        aiAssisted: !!row.aiAssisted,
      });
    }
    const group = groups.get(key);
    if (!group.rawType && rawOriginal && rawOriginal.trim()){
      group.rawType = rawOriginal;
    }
    if (!group.spacerKind && row.spacerKind){
      group.spacerKind = row.spacerKind;
    }
    if (row.aiAssisted){
      group.aiAssisted = true;
    }
    const qty = Math.max(0, Number(row.quantity || 0));
    const areaPerUnit = (()=> {
      const direct = Number(row.area ?? row.area_m2 ?? row.areaM2);
      if (Number.isFinite(direct)) return direct;
      const computed = computeRowArea(row);
      return Number.isFinite(computed) ? computed : 0;
    })();
    group.qty += qty;
    group.area += areaPerUnit * qty;
    group.rawIndexes.push(idx);
  });

  const lines = [];
  const groupEntries = Array.from(groups.values());
  for (let index = 0; index < groupEntries.length; index += 1){
    const line = await computeInvoiceLineFromGroup(groupEntries[index], index, { promptAllowedForJob, allowAi });
    lines.push(line);
  }
  return lines;
}

async function computeInvoiceLineFromGroup(group, index, options = {}){
  const promptAllowedForJob = !!options.promptAllowedForJob;
  const allowAi = options.allowAi !== false;
  const glassPriceTable = appState.invoices.priceLists.glass || {};
  const completeProductKey = normalizeGlassKey(group.displayType || "");
  const completeProductMatch = completeProductKey
    && Object.prototype.hasOwnProperty.call(glassPriceTable, completeProductKey)
    ? completeProductKey
    : null;
  let composition = parseInvoiceComposition(group.displayType || "");
  if (completeProductMatch){
    composition = {
      ...composition,
      panes: [completeProductMatch],
      spacerThicknesses: [],
    };
  }
  // Manual-order catalog names such as "TERMIK + TRANSPARENT 24 mm" are
  // complete products, not always pane + spacer formulas. If the formula
  // parser cannot identify a glass pane, price the complete product name
  // instead of silently producing a zero-value invoice line.
  if (!composition.panes.length && String(group.displayType || "").trim()){
    composition = {
      ...composition,
      panes: [String(group.displayType).trim()],
      spacerThicknesses: [],
    };
  }
  let glassPricePerM2 = 0;
  let aiAssisted = !!group.aiAssisted;
  let lineIsLaminated = false;
  let lineDisplayType = group.displayType || "";
  let pricingUnderstanding = completeProductMatch ? {
    status: "matched",
    matchSource: "exact_catalog",
    matchedKey: completeProductMatch,
    confidence: 1,
    safeToPrice: true,
    pricingMode: "finished_product",
    explanation: `Exact catalog product matched: ${completeProductMatch}.`,
    alternatives: [],
    warnings: [],
  } : null;
  const glassPriceKeys = Object.keys(glassPriceTable);
  const glassIssues = [];
  const matchedGlassKeys = [];
  const matchedGlassSources = [];
  for (const pane of composition.panes){
    const result = await resolveGlassType(pane, glassPriceKeys, { allowAi: false });
    if (result?.fromAi){
      aiAssisted = true;
    }
    const matchedKey = result?.match;
    const normalizedKey = matchedKey || normalizeGlassKey(pane);
    const priceKey = normalizedKey;
    const price = priceKey ? glassPriceTable[priceKey] : undefined;
    if (!Number.isFinite(price)){
      const canonicalLabel = priceKey || matchedKey || normalizeGlassKey(pane) || pane;
      console.log("[GlassPriceMissing] rawName:", pane, "normalizedKey:", canonicalLabel, "availableKeys:", Object.keys(glassPriceTable || {}));
      glassIssues.push({
        missing: { type: "glass", label: pane, pane: canonicalLabel, normalized: canonicalLabel },
        prompt: { kind: "glass", label: pane, normalized: canonicalLabel },
      });
    }else{
      glassPricePerM2 += Number(price);
      matchedGlassKeys.push(priceKey);
      matchedGlassSources.push(result?.matchSource || "exact_catalog");
    }
  }
  if (!glassIssues.length && !pricingUnderstanding && matchedGlassKeys.length){
    pricingUnderstanding = {
      status: "matched",
      matchSource: matchedGlassSources.every(source => source === "exact_catalog")
        ? "exact_catalog_components"
        : "normalized_catalog_alias",
      matchedKey: matchedGlassKeys.join(" + "),
      confidence: 1,
      safeToPrice: true,
      pricingMode: "component",
      explanation: matchedGlassSources.every(source => source === "exact_catalog")
        ? `Matched ${matchedGlassKeys.length === 1 ? "the glass product" : "all glass components"} directly to the price catalog.`
        : `Normalized spelling and formatting, then matched ${matchedGlassKeys.length === 1 ? "the glass product" : "all glass components"} to the price catalog.`,
      alternatives: [],
      warnings: [],
    };
  }
  const inferredKind = (composition.spacerKind === "thermal" || composition.spacerKind === "normal")
    ? composition.spacerKind
    : null;
  let spacerModeResolved = false;
  let spacerMode = null;
  if (group.spacerKind === "thermal" || group.spacerKind === "normal"){
    spacerMode = group.spacerKind;
    spacerModeResolved = true;
  }else if (inferredKind){
    spacerMode = inferredKind;
    spacerModeResolved = true;
  }
  const spacerText = `${group.displayType || ""} ${group.rawType || ""}`;
  const caldoDetected = /\bcaldo\b/i.test(spacerText);
  if (caldoDetected){
    spacerMode = "thermal";
    spacerModeResolved = true;
    aiAssisted = true;
  }
  if (!spacerMode){
    spacerMode = "normal";
  }
  const thicknesses = Array.isArray(composition.spacerThicknesses) && composition.spacerThicknesses.length
    ? composition.spacerThicknesses
    : [];
  const needsGlassAi = glassIssues.length > 0;
  const needsSpacerAi = !spacerModeResolved;
  const needsHistoricalAi = group.aiAssisted === true;
  const rawLineText = group.rawType || group.displayType || "";
  if (allowAi && (needsGlassAi || needsSpacerAi || needsHistoricalAi) && rawLineText){
    console.log("[GlassAI] Calling AI for raw line:", rawLineText);
    const analysis = await analyzeInvoiceLineWithAI(rawLineText, glassPriceKeys);
    console.log("[GlassAI] Result:", analysis);
    if (analysis){
      const confident = Number.isFinite(analysis.confidence) ? Number(analysis.confidence) >= 0.8 : false;
      pricingUnderstanding = {
        status: analysis.matchStatus || "unresolved",
        matchSource: "gpt-5.6-terra",
        matchedKey: analysis.glassKey || null,
        confidence: Number.isFinite(analysis.confidence) ? Number(analysis.confidence) : 0,
        safeToPrice: Boolean(analysis.safeToPrice && confident),
        pricingMode: analysis.pricingMode || "unresolved",
        explanation: analysis.pricingExplanation || analysis.reason || "AI could not safely resolve this description.",
        alternatives: Array.isArray(analysis.alternativeGlassKeys) ? analysis.alternativeGlassKeys.slice(0, 3) : [],
        recognizedTerms: Array.isArray(analysis.recognizedTerms) ? analysis.recognizedTerms.slice(0, 12) : [],
        warnings: Array.isArray(analysis.warnings) ? analysis.warnings.slice(0, 8) : [],
        overallThicknessMm: analysis.overallThicknessMm ?? null,
      };
      if (confident){
        aiAssisted = true;
        if (analysis.normalizedType){
          const cleaned = analysis.normalizedType.trim();
          if (cleaned){
            lineDisplayType = cleaned;
          }
        }
        lineIsLaminated = Boolean(analysis.isLaminated);
        if (analysis.glassKey && analysis.safeToPrice && glassIssues.length){
          const aiGlassKeyMatch = glassPriceKeys.find(key => key.toLowerCase() === String(analysis.glassKey).toLowerCase());
          const aiGlassKey = aiGlassKeyMatch || analysis.glassKey;
          const aiPrice = glassPriceTable[aiGlassKey];
          if (Number.isFinite(aiPrice)){
            if (analysis.pricingMode === "finished_product"){
              glassPricePerM2 = Number(aiPrice);
              glassIssues.length = 0;
              composition.spacerThicknesses = [];
            }else if (analysis.pricingMode === "component" && glassIssues.length === 1){
              glassPricePerM2 += Number(aiPrice);
              glassIssues.length = 0;
            }else{
              pricingUnderstanding.safeToPrice = false;
              pricingUnderstanding.status = "ambiguous";
              pricingUnderstanding.warnings = [
                ...(pricingUnderstanding.warnings || []),
                "More than one unresolved component remains; confirm the construction before pricing.",
              ];
            }
          }else{
            const normalizedAiKey = normalizeGlassKey(aiGlassKey) || aiGlassKey;
            console.log("[GlassPriceMissing] rawName:", analysis.glassKey, "normalizedKey:", normalizedAiKey, "availableKeys:", Object.keys(glassPriceTable || {}));
            glassIssues.push({
              missing: { type: "glass", label: analysis.glassKey, pane: normalizedAiKey, normalized: normalizedAiKey },
              prompt: { kind: "glass", label: analysis.glassKey, normalized: normalizedAiKey },
            });
          }
        }
        if (analysis.spacerMode === "thermal" || analysis.spacerMode === "normal"){
          spacerMode = analysis.spacerMode;
          spacerModeResolved = true;
        }
      }
    }
  }
  let spacerPricePerM2 = 0;
  const spacerIssues = [];
  const pricedThicknesses = pricingUnderstanding?.safeToPrice
    && pricingUnderstanding?.pricingMode === "finished_product"
    ? []
    : thicknesses;
  pricedThicknesses.forEach(th=>{
    const spacer = getSpacerPrice(th, spacerMode);
    if (spacer.missing){
      spacerIssues.push({
        missing: { type: "spacer", details: spacer.missing },
        prompt: {
          kind: "spacer",
          thickness: th,
          spacerKind: spacerMode,
          description: `Spacer ${th ?? "?"}mm (${spacerMode || "normal"})`,
        },
      });
    }
    spacerPricePerM2 += spacer.price || 0;
  });
  const missingEntries = [...glassIssues, ...spacerIssues];
  let lineMissing = null;
  let lineMissingPrice = false;
  if (missingEntries.length){
    lineMissing = missingEntries[0].missing;
    lineMissingPrice = missingEntries.some(entry => entry?.missing?.type === "glass");
    if (promptAllowedForJob){
      missingEntries.forEach(entry=>{
        enqueueInvoicePrompt(entry.prompt);
      });
    }
    if (pricingUnderstanding){
      pricingUnderstanding.safeToPrice = false;
      pricingUnderstanding.warnings = [
        ...(pricingUnderstanding.warnings || []),
        "A required catalog or spacer price is still missing.",
      ];
    }
  }
  const unitPricePerM2 = glassPricePerM2 + spacerPricePerM2;
  const lineTotal = unitPricePerM2 * group.area;
  const normalizedDisplayType = (() => {
    const text = lineDisplayType || group.displayType || "";
    const normalized = normalizeGlassKey(text);
    return normalized || text.toLowerCase();
  })();
  return {
    index: index + 1,
    orderId: group.orderId || null,
    type: lineDisplayType,
    displayType: lineDisplayType,
    normalizedType: normalizedDisplayType,
    quantity: group.qty,
    area: group.area,
    glassPricePerM2,
    spacerPricePerM2,
    spacerKind: spacerMode,
    spacerMode,
    spacerThicknesses: pricedThicknesses.slice(),
    spacerOverride: null,
    unitPricePerM2,
    lineTotal,
    lineAreaTotal: group.area,
    rawIndexes: group.rawIndexes || [],
    missing: lineMissing,
    missingPrice: lineMissingPrice,
    rawType: group.rawType || null,
    isLaminated: lineIsLaminated,
    aiAssisted,
    pricingUnderstanding,
  };
}

function finalizeInvoiceTotals(job, lines){
  let subtotal = 0;
  let units = 0;
  let areaTotal = 0;
  let missing = null;
  lines.forEach(line=>{
    const lineTotal = Number(line.lineTotal || 0);
    const qty = Number(line.quantity || 0);
    const area = Number(line.area || line.lineAreaTotal || 0);
    subtotal += lineTotal;
    units += qty;
    areaTotal += area;
    if (!missing && line.missing){
      missing = line.missing;
    }
  });
  const discountRaw = Math.max(0, Number(job.discountValue || 0));
  const discountValue = job.discountMode === "absolute"
    ? Math.min(discountRaw, subtotal)
    : subtotal * (discountRaw / 100);
  const taxable = Math.max(0, subtotal - discountValue);
  const vatValue = taxable * (Math.max(0, Number(job.vatRate || 0)) / 100);
  const total = taxable + vatValue;
  job.calculated = {
    lines,
    subtotal,
    discountValue,
    vatValue,
    total,
    units,
    area: areaTotal,
    missing,
  };
  applyJobTotals(job, job.calculated);
}

function applyJobTotals(job, calc){
  if (!job) return;
  const source = calc || job.calculated || {};
  job.totalUnits = Number(source.units || 0);
  job.totalArea = Number(source.area || 0);
  job.totalAmount = Number(source.total || 0);
  updateJobOrdersFromCalc(job, source);
}

function normalizeGlassAliasToken(value){
  const original = String(value || "");
  const text = original.toLowerCase().replace(/\+/g, " ").replace(/\bmm\b/g, " ");
  const fixed = fixGlassTypos(text, original);
  return fixed.replace(/\s+/g, " ").trim();
}

function getGlassAliasCacheKey(value){
  const normalized = normalizeGlassAliasToken(value);
  return normalized.replace(/\s+/g, "");
}

function findLocalGlassMatch(targetCompact, candidates){
  if (!targetCompact) return null;
  const exact = candidates.find(entry => entry.compact === targetCompact);
  if (exact) return exact.key;
  const numberSignature = value => (String(value || "").match(/\d+(?:[.,]\d+)?/g) || []).join("|");
  const hasCompatibleNumbers = entry => numberSignature(targetCompact) === numberSignature(entry.compact);
  const startsWith = candidates.find(entry => hasCompatibleNumbers(entry)
    && (targetCompact.startsWith(entry.compact) || entry.compact.startsWith(targetCompact)));
  if (startsWith) return startsWith.key;
  const contains = candidates.find(entry => hasCompatibleNumbers(entry)
    && (targetCompact.includes(entry.compact) || entry.compact.includes(targetCompact)));
  if (contains) return contains.key;
  return null;
}

async function resolveGlassType(rawName, knownTypes = [], options = {}){
  const allowAi = options.allowAi !== false;
  if (!rawName || !Array.isArray(knownTypes) || !knownTypes.length){
    return { match: null, fromAi: false };
  }
  const cacheKey = getGlassAliasCacheKey(rawName);
  if (!cacheKey){
    return { match: null, fromAi: false };
  }
  const factoryAliasTarget = appState?.invoices?.componentAliases?.[normalizeGlassKey(rawName)] || null;
  if (factoryAliasTarget){
    const aliasMatch = knownTypes.find(type => normalizeGlassKey(type) === factoryAliasTarget);
    if (aliasMatch){
      const result = { match: aliasMatch, fromAi: false, matchSource: "factory_alias" };
      glassAliasCache[cacheKey] = result;
      return result;
    }
  }
  if (Object.prototype.hasOwnProperty.call(glassAliasCache, cacheKey)){
    return glassAliasCache[cacheKey];
  }
  const candidates = knownTypes.map(type => ({
    key: type,
    normalized: normalizeGlassAliasToken(type),
    compact: getGlassAliasCacheKey(type),
  }));
  const localMatch = findLocalGlassMatch(cacheKey, candidates);
  if (localMatch){
    const isExact = candidates.some(entry => entry.key === localMatch && entry.compact === cacheKey);
    const result = { match: localMatch, fromAi: false, matchSource: isExact ? "exact_catalog" : "normalized_alias" };
    glassAliasCache[cacheKey] = result;
    return result;
  }
  if (!allowAi){
    return { match: null, fromAi: false };
  }
  if (glassAliasPending.has(cacheKey)){
    return glassAliasPending.get(cacheKey);
  }
  const pending = (async ()=>{
    try{
      console.log("[GlassAI] Calling AI glass matcher for:", rawName);
      const aiMatch = await callAiGlassMatch(rawName, knownTypes);
      console.log("[GlassAI] Result:", aiMatch);
      if (aiMatch){
        const resolved = knownTypes.find(type => type.toLowerCase() === aiMatch.toLowerCase()) || aiMatch;
        const result = { match: resolved, fromAi: true };
        glassAliasCache[cacheKey] = result;
        return result;
      }
    }catch(error){
      console.warn("AI glass resolver failed", error);
    }
    const fallback = { match: null, fromAi: false };
    glassAliasCache[cacheKey] = fallback;
    return fallback;
  })().finally(()=>{
    glassAliasPending.delete(cacheKey);
  });
  glassAliasPending.set(cacheKey, pending);
  return pending;
}

async function generateLabelsPdf(rows){
  await ensurePdfLib();
  const { PDFDocument, StandardFonts } = PDFLib;
  const pdfDoc = await PDFDocument.create();
  const bold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);
  const regular = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const { keliBytes, ceBytes } = await preloadLogos();
  let keliImg, ceImg;
  if (keliBytes){
    const data = keliBytes instanceof Uint8Array ? keliBytes : new Uint8Array(keliBytes);
    keliImg = await pdfDoc.embedPng(data);
  }
  if (ceBytes){
    const data = ceBytes instanceof Uint8Array ? ceBytes : new Uint8Array(ceBytes);
    ceImg = await pdfDoc.embedPng(data);
  }
  const today = new Date().toLocaleDateString("en-GB").replaceAll("/", ".");
  for (const row of rows){
    const qty = Number(row.quantity || 0) || 1;
    const labelSource = String(row.source || "").toLowerCase();
    const msIndexValue = Number(row.ms_index ?? row.msIndex);
    const isProcessingLabel = labelSource === "processing";
    for (let i = 0; i < qty; i++){
      const page = pdfDoc.addPage([pageSize.w, pageSize.h]);
      const margin = 16;
      const spacing = 12;
      let y = pageSize.h - margin;
      if (keliImg) page.drawImage(keliImg, { x: margin, y: y - 20, width: 60, height: 20 });
      if (ceImg) page.drawImage(ceImg, { x: (pageSize.w - 30) / 2, y: y - 18, width: 30, height: 18 });
      page.drawText(today, { x: pageSize.w - margin - 60, y: y - 14, size: 10, font: bold });
      page.drawLine({ start: { x: margin, y: y - 24 }, end: { x: pageSize.w - margin, y: y - 24 }, thickness: 0.8 });
      y -= (24 + spacing);
      const dimension = row.dimension && row.dimension.length ? row.dimension : "—";
      let positionText = row.position || "—";
      if (isProcessingLabel){
        const summary = formatProcessingPositionsSummary(row);
        if (summary){
          positionText = summary;
        }else if (!positionText || positionText === "(Grouped)"){
          positionText = "—";
        }
      }
      const orderNo = row.order_number || "—";
      const type = row.type || "";
      page.drawText(`Order No.: ${orderNo}   Pos: ${positionText}   Dim: ${dimension}`, {
        x: margin, y, size: 9, font: bold, maxWidth: pageSize.w - margin * 2
      });
      y -= 14;
      page.drawText(`Glass Type: ${type}`, { x: margin, y, size: 10, font: bold, maxWidth: pageSize.w - margin * 2 });
      y -= 16;
      const showProcessingMsIndex = isProcessingLabel && Number.isFinite(msIndexValue);
      if (showProcessingMsIndex){
        const footerText = String(Math.trunc(msIndexValue));
        const footerSize = 18;
        const msTopSpacing = 8;
        const msBottomSpacing = 12;
        const msVerticalOffset = -16; // push down visually by 20px
        const textWidth = bold.widthOfTextAtSize(footerText, footerSize);
        const footerX = (pageSize.w - textWidth) / 2;
        const footerY = y + (16 - msTopSpacing) + msVerticalOffset;
        page.drawText(footerText, { x: footerX, y: footerY, size: footerSize, font: bold });
        y = footerY - msBottomSpacing;
      }else{
        page.drawText("KELI ALBANIA PVC", { x: margin, y, size: 10, font: regular });
      }
    }
  }
  return pdfDoc.save();
}

function manualOrderToShared(order, { perUnitArea = false } = {}){
  const statusMap = {
    draft: "draft",
    approved: "approved",
    processing: "in_production",
    finished: "completed",
    cancelled: "archived",
  };
  return {
    id: `manual-${order.id ?? order.manual_order_id}`,
    source: "manual",
    client_name: order.client_name,
    client: order.client_name,
    order_number: order.order_number,
    order_numbers: [order.order_number],
    order_date: order.order_date,
    created_at: order.created_at || order.order_date,
    status: statusMap[order.status] || "draft",
    manual_format: order.manual_format || "standard",
    units_total: order.total_quantity || (order.rows || []).reduce((sum, row) => sum + Number(row.quantity || 0), 0),
    area_total: order.total_area_m2 || (order.rows || []).reduce((sum, row) => sum + Number(row.final_area_m2 || 0), 0),
    rows: (order.rows || []).map(row => {
      const quantity = Number(row.quantity || 0);
      const finalArea = Number(row.final_area_m2 ?? row.area ?? 0);
      return {
        id: row.id,
        order_number: order.order_number,
        position: row.position || "",
        section: row.section || "",
        client_position: row.client_position || "",
        index_number: row.index_number ?? null,
        manual_format: order.manual_format || row.manual_format || "standard",
        type: row.glass_type || row.type || "",
        glass_type: row.glass_type || row.type || "",
        width_mm: Number(row.width_mm),
        height_mm: Number(row.height_mm),
        dimension: row.dimension || `${Number(row.width_mm)}x${Number(row.height_mm)}`,
        quantity,
        area: perUnitArea && quantity > 0 ? finalArea / quantity : finalArea,
        final_area_m2: finalArea,
        source: "manual",
        notes: row.notes || "",
      };
    }),
  };
}

function parsePlatformDate(value){
  if (value instanceof Date){
    return Number.isNaN(value.getTime()) ? null : new Date(value.getTime());
  }
  if (typeof value === "number"){
    const numericDate = new Date(value);
    return Number.isNaN(numericDate.getTime()) ? null : numericDate;
  }
  const text = String(value ?? "").trim();
  if (!text) return null;

  // Date-only values represent a local calendar day. Backend date-times are
  // stored in UTC; older SQLite responses may be missing their UTC suffix.
  let normalized = text;
  const isDateOnly = /^\d{4}-\d{2}-\d{2}$/.test(text);
  const isIsoDateTime = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/.test(text);
  const hasTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(text);
  if (isDateOnly){
    normalized = `${text}T00:00:00`;
  }else if (isIsoDateTime && !hasTimezone){
    normalized = `${text}Z`;
  }

  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
}
