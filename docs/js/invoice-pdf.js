/* A4 invoice layout with measured wrapping, repeated headings, and page numbers. */
(() => {
  async function build(job) {
    await ensurePdfLib();
    const { PDFDocument, StandardFonts, rgb } = window.PDFLib;
    const pdf = await PDFDocument.create();
    const regular = await pdf.embedFont(StandardFonts.Helvetica);
    const bold = await pdf.embedFont(StandardFonts.HelveticaBold);
    const navy = rgb(.09, .16, .27), muted = rgb(.38, .43, .50), blue = rgb(.12, .33, .68), rule = rgb(.83, .87, .91), pale = rgb(.95, .97, .99);
    const W = 595.28, H = 841.89, margin = 42, right = W - margin;
    const document = window.InvoiceWorkspace.documentData(job);
    const calc = job.calculated;
    const number = job.invoiceNumber;
    let page, y;
    pdf.setTitle(`Invoice ${number}`);
    pdf.setAuthor(document.businessName || '');
    pdf.setSubject(`Invoice for ${document.clientName || job.client}`);

    function text(value, x, top, size = 10, font = regular, color = navy) {
      try { page.drawText(String(value ?? ''), { x, y: H - top - size, size, font, color }); }
      catch (error) {
        if (/encode/i.test(error.message)) throw new Error('The PDF font cannot display a character in this invoice. Use Latin text for this export. Your invoice is still saved.');
        throw error;
      }
    }
    function wrapped(value, width, size = 10, font = regular) {
      const lines = [];
      for (const paragraph of String(value ?? '').replace(/\r/g, '').split('\n')) {
        if (!paragraph) { lines.push(''); continue; }
        let line = '';
        for (const word of paragraph.split(/\s+/)) {
          const candidate = line ? `${line} ${word}` : word;
          if (font.widthOfTextAtSize(candidate, size) <= width) { line = candidate; continue; }
          if (line) lines.push(line);
          line = '';
          for (const char of word) {
            if (font.widthOfTextAtSize(line + char, size) > width && line) { lines.push(line); line = ''; }
            line += char;
          }
        }
        lines.push(line);
      }
      return lines;
    }
    function block(value, x, top, width, size = 10, font = regular, color = navy) {
      const lines = wrapped(value, width, size, font);
      lines.forEach((line, i) => text(line, x, top + i * (size + 4), size, font, color));
      return top + lines.length * (size + 4);
    }
    function line(top, x = margin, end = right) {
      page.drawLine({ start: { x, y: H - top }, end: { x: end, y: H - top }, thickness: .7, color: rule });
    }
    function rect(top, height, color, x = margin, width = right - margin) {
      page.drawRectangle({ x, y: H - top - height, width, height, color });
    }
    function rightText(value, edge, top, size = 10, font = regular, color = navy) {
      text(value, edge - font.widthOfTextAtSize(String(value), size), top, size, font, color);
    }
    function newPage() {
      page = pdf.addPage([W, H]);
      rect(0, 5, blue, 0, W);
      text('INVOICE', margin, 29, 11, bold, blue);
      rightText(number, right, 30, 9, bold);
      line(55);
      y = 74;
    }
    function tableHeader() {
      rect(y, 27, navy);
      text('DESCRIPTION', margin + 10, y + 8, 8, bold, rgb(1,1,1));
      rightText(job.source === 'invoice' ? 'QTY' : 'AREA / m²', 360, y + 8, 8, bold, rgb(1,1,1));
      rightText('RATE / ALL', 443, y + 8, 8, bold, rgb(1,1,1));
      rightText('AMOUNT / ALL', right - 10, y + 8, 8, bold, rgb(1,1,1));
      y += 36;
    }
    const money = value => Number(value || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const date = value => value ? new Date(`${value}T12:00:00`).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) : '—';
    newPage();
    const leftBottom = block(document.businessName || '', margin, y, 275, 18, bold);
    let leftY = block(document.businessDetails || '', margin, leftBottom + 8, 265, 9, regular, muted);
    if (document.businessTaxId) leftY = block(`Tax ID: ${document.businessTaxId}`, margin, leftY + 4, 265, 9, regular, muted);
    text(job.status === 'finalized' ? 'FINALIZED' : 'DRAFT', 390, y + 2, 10, bold, blue);
    text('Issue date', 390, y + 29, 9, regular, muted);
    rightText(date(document.issueDate), right, y + 45, 10, bold);
    text('Due date', 390, y + 68, 9, regular, muted);
    rightText(date(document.dueDate), right, y + 84, 10, bold);
    y = Math.max(leftY + 24, y + 120);
    line(y); y += 20;
    text('BILL TO', margin, y, 8, bold, muted); y += 18;
    y = block(document.clientName || job.client || '', margin, y, right - margin, 13, bold);
    if (document.clientDetails) y = block(document.clientDetails, margin, y + 5, right - margin, 9, regular, muted);
    if (document.clientTaxId) y = block(`Tax ID: ${document.clientTaxId}`, margin, y + 5, right - margin, 9, regular, muted);
    if (job.orderNumbers?.length) y = block(`Order reference: ${job.orderNumbers.join(', ')}`, margin, y + 10, right - margin, 9, regular, muted);
    y += 24;
    if (y > H - 160) newPage();
    tableHeader();
    for (const item of calc.lines) {
      const description = item.orderId && job.orderNumbers?.length > 1 ? `${item.type}\nOrder ${item.orderId}` : item.type;
      const lines = wrapped(description, 250, 10);
      const quantity = job.source === 'invoice' ? item.quantity : item.area;
      let offset = 0;
      do {
        if (y + 30 > H - 65) { newPage(); tableHeader(); }
        const capacity = Math.max(1, Math.floor((H - 80 - y) / 14));
        const segment = lines.slice(offset, offset + capacity);
        segment.forEach((value, index) => text(value, margin + 10, y + index * 14));
        if (!offset) {
          rightText(Number(quantity || 0).toLocaleString('en-US', { maximumFractionDigits: 3 }), 360, y);
          rightText(money(item.unitPricePerM2), 443, y);
          rightText(money(item.lineTotal), right - 10, y, 10, bold);
        }
        y += segment.length * 14 + 16;
        offset += segment.length;
        line(y - 6);
        if (offset < lines.length) { newPage(); tableHeader(); }
      } while (offset < lines.length);
    }
    if (y + 165 > H - 65) newPage();
    y += 14;
    for (const [label, value] of [
      ['Subtotal', calc.subtotal],
      [`Discount${job.discountMode === 'percent' ? ` (${job.discountValue || 0}%)` : ''}`, -calc.discountValue],
      [`VAT (${job.vatRate || 0}%)`, calc.vatValue],
    ]) { text(label, 320, y, 10, regular, muted); rightText(money(value), right - 10, y, 10); y += 24; }
    rect(y, 43, pale, 309, right - 309);
    text('Total / ALL', 320, y + 13, 12, bold, blue);
    rightText(money(calc.total), right - 10, y + 12, 15, bold, blue);
    y += 68;
    if (document.notes) {
      if (y + 50 > H - 65) newPage();
      text('PAYMENT DETAILS & NOTES', margin, y, 8, bold, muted); y += 18;
      for (const value of wrapped(document.notes, right - margin, 9)) {
        if (y + 15 > H - 65) newPage();
        text(value, margin, y, 9, regular, muted); y += 13;
      }
    }
    const pages = pdf.getPages();
    pages.forEach((item, index) => {
      page = item;
      line(H - 44);
      text('Thank you for your business.', margin, H - 32, 8, regular, muted);
      rightText(`${job.status === 'finalized' ? '' : 'DRAFT · '}${index + 1} / ${pages.length}`, right, H - 32, 8, regular, muted);
    });
    return pdf.save();
  }

  async function download(job) {
    let bytes;
    try { bytes = await build(job); }
    catch (error) {
      if (/encode/i.test(error.message)) throw new Error('A character in this invoice is not supported by the PDF font. Use Latin text for this export. Your invoice is still saved.');
      throw error;
    }
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/pdf' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${String(job.invoiceNumber).replace(/[^a-zA-Z0-9._-]/g, '-')}.pdf`;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  window.InvoicePDF = { build, download };
})();
