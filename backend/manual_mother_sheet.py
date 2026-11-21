"""Read-only Mother Sheet printout used exclusively by Manual Orders' Print action."""

from collections import OrderedDict
from datetime import date
from decimal import Decimal
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
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


def build_manual_mother_sheet_pdf(order):
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

    output = BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
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
    story = [Paragraph(escape(
        f"Mother Sheet – Client: {client} | Order: {order_number} | "
        f"Date: {display_date} | Dimensions: cm"
    ), header)]
    for glass, members in groups.items():
        # Use saved final areas, including overrides, rather than recomputing geometry.
        area = sum((Decimal(str(row["final_area_m2"])) for row in members), Decimal(0))
        area_text = format(area, ".3f").replace(".", ",")
        story.append(Paragraph(escape(f"{glass} — Area: {area_text} m²"), heading))
        story.append(Paragraph(escape(f"[Order {order_number} — {client}]"), reference))
        for row in members:
            index = row.get("index_number")
            position = str(index if index not in (None, "") else row.get("position") or "")
            client_position = str(row.get("client_position") or "").strip()
            if client_position:
                position += f" ({client_position})"
            line = (f'{position} – {_dimension_cm(row["width_mm"])} × '
                    f'{_dimension_cm(row["height_mm"])} × {row["quantity"]}')
            story.append(Paragraph(escape(line), item))
        story.append(Spacer(1, 10 * mm))
    doc.build(story)
    return output.getvalue()
