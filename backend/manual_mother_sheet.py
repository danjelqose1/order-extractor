"""Read-only Mother Sheet printout used exclusively by Manual Orders' Print action."""

from collections import OrderedDict
from datetime import date
from decimal import Decimal
from io import BytesIO
from xml.sax.saxutils import escape

import fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _dimension_cm(value):
    # Decimal avoids rounding fractional millimetres or introducing float noise.
    text = format(Decimal(str(value)) / Decimal(10), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace(".", ",")


def build_manual_mother_sheet_pdf(order, layout="a4_portrait"):
    if layout == "a4_portrait":
        return _build_single_copy(order)
    if layout not in {"a4_portrait_2up", "a4_landscape_2up"}:
        raise ValueError("Unsupported Mother Sheet print layout.")
    portrait = layout == "a4_portrait_2up"
    width, height = A4 if portrait else landscape(A4)
    column_width = width / 2
    copy_size = (column_width, height)
    content = _fit_two_copy_sheet(order, copy_size, portrait=portrait)
    slots = [fitz.Rect(0, 0, column_width, height), fitz.Rect(column_width, 0, width, height)]
    with fitz.open(stream=content, filetype="pdf") as source, fitz.open() as output:
        page = output.new_page(width=width, height=height)
        for slot in slots:
            page.show_pdf_page(slot, source, 0)
        output.set_metadata({"title": source.metadata.get("title", "Mother Sheet")})
        return output.tobytes(garbage=4, deflate=True)


def _fit_two_copy_sheet(order, copy_size, *, portrait):
    content = _build_single_copy(order, copy_size, compact_scale=1, column_header=portrait)
    with fitz.open(stream=content, filetype="pdf") as source:
        if len(source) != 1:
            raise ValueError("This order is too long for two complete copies on one page. Choose 1 copy or try the other orientation.")
    if not portrait:
        return content

    # Use the largest scale that fits after wrapping, up to 22 pt for item text.
    low, high = 21, 42
    while low <= high:
        step = (low + high) // 2
        candidate = _build_single_copy(order, copy_size, compact_scale=step / 20, column_header=True)
        with fitz.open(stream=candidate, filetype="pdf") as source:
            if len(source) == 1:
                content = candidate
                low = step + 1
            else:
                high = step - 1
    return content


def _build_single_copy(order, page_size=A4, compact_scale=None, *, column_header=False):
    rows = order.get("rows") or []
    if not rows:
        raise ValueError("Manual order has no rows")

    order_number = str(order.get("order_number") or "")
    client = str(order.get("client_name") or "")
    saved_date = str(order.get("order_date") or "")
    display_date = date.fromisoformat(saved_date).strftime("%d/%m/%Y") if saved_date else ""
    groups = OrderedDict()
    for row in rows:
        # Thickness, when supplied, is already part of the saved glass description.
        glass = str(row.get("glass_type") or "Unspecified glass")
        groups.setdefault(glass, []).append(row)

    compact = compact_scale is not None
    output = BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=page_size,
        leftMargin=(10 if compact else 20) * mm, rightMargin=(10 if compact else 20) * mm,
        topMargin=(10 if compact else 18) * mm, bottomMargin=(10 if compact else 18) * mm,
        title=f"Mother Sheet – {order_number}", author="",
    )
    header = ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=16,
                            leading=22, textColor=colors.black, spaceAfter=22, keepWithNext=True)
    heading = ParagraphStyle("glass", fontName="Helvetica-Bold", fontSize=14,
                             leading=20, textColor=colors.black, spaceAfter=12, keepWithNext=True)
    reference = ParagraphStyle("reference", fontName="Helvetica", fontSize=12,
                               leading=18, textColor=colors.black, spaceAfter=6, keepWithNext=True)
    item = ParagraphStyle("item", fontName="Helvetica", fontSize=13,
                          leading=20, textColor=colors.black, spaceAfter=3)
    if compact:
        for style, size, leading, after, max_scale in [
            (header, 12, 16, 12, 1.25), (heading, 11, 15, 7, 1.45),
            (reference, 10, 14, 4, 1.3), (item, 10.5, 14, 1, 2.1),
        ]:
            scale = min(compact_scale, max_scale)
            style.fontSize = size * scale
            style.leading = leading * scale
            style.spaceAfter = after * scale
        if column_header:
            item.spaceAfter = 16 * (compact_scale - 1) + 1

    header_text = escape(
        f"Mother Sheet – Client: {client} | Order: {order_number} | "
        f"Date: {display_date} | Dimensions: cm"
    )
    if column_header:
        header_text = "<br/>".join(map(escape, (
            "Mother Sheet", f"Client: {client}", f"Order: {order_number}",
            f"Date: {display_date}", "Dimensions: cm",
        )))
    story = [Paragraph(header_text, header)]
    for glass, members in groups.items():
        if len(story) > 1:
            story.append(Spacer(1, (6 * compact_scale if compact else 10) * mm))
        # Use saved final areas, including overrides, rather than recomputing geometry.
        area = sum((Decimal(str(row["final_area_m2"])) for row in members), Decimal(0))
        area_text = format(area, ".3f").replace(".", ",")
        glass_heading = escape(f"{glass} — Area: {area_text} m²")
        if column_header:
            glass_heading = f"{escape(glass)}<br/>Area: {area_text} m²"
        story.append(Paragraph(glass_heading, heading))
        story.append(Paragraph(escape(f"[Order {order_number} — {client}]"), reference))
        for row in members:
            index = row.get("index_number")
            position = str(index if index not in (None, "") else row.get("position") or "")
            client_position = str(row.get("client_position") or "").strip()
            if client_position:
                position += f" ({client_position})"
            dimensions = (f'{_dimension_cm(row["width_mm"])} × '
                          f'{_dimension_cm(row["height_mm"])} × {row["quantity"]}')
            if column_header:
                formatted_dimensions = escape(dimensions).replace(" ", "&#160;")
                line = f"{escape(position)} – {formatted_dimensions}"
            else:
                line = escape(f"{position} – {dimensions}")
            story.append(Paragraph(line, item))
    doc.build(story)
    return output.getvalue()
