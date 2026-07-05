from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, Iterable, List, Sequence

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


MANUAL_PROCESSING_PAGE_SIZE = (100 * mm, 210 * mm)
MANUAL_LABEL_PAGE_SIZE = (100 * mm, 40 * mm)


def _pdf_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else fallback).strip()
    return text.encode("cp1252", "replace").decode("cp1252")


def _display_date(value: Any) -> str:
    text = _pdf_text(value)
    parts = text.split("-")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return ".".join(reversed(parts))
    return text


def _format_mm(value: Any) -> str:
    number = float(value or 0)
    if abs(number - round(number)) < 0.001:
        return str(int(round(number)))
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _format_cm(value: Any) -> str:
    number = float(value or 0) / 10
    if abs(number - round(number)) < 0.001:
        return str(int(round(number)))
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _draw_fitted_text(
    pdf: canvas.Canvas,
    text: str,
    *,
    x: float,
    y: float,
    max_width: float,
    font: str,
    size: float,
    min_size: float = 6,
    align: str = "left",
) -> float:
    safe = _pdf_text(text, "-")
    fitted = size
    while fitted > min_size and stringWidth(safe, font, fitted) > max_width:
        fitted -= 0.5
    if stringWidth(safe, font, fitted) > max_width:
        while safe and stringWidth(f"{safe}...", font, fitted) > max_width:
            safe = safe[:-1]
        safe = f"{safe}..." if safe else "-"
    if align == "center":
        pdf.drawCentredString(x, y, safe)
    elif align == "right":
        pdf.drawRightString(x, y, safe)
    else:
        pdf.drawString(x, y, safe)
    return fitted


def _group_rows(rows: Iterable[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]]]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for row in rows:
        glass_type = _pdf_text(row.get("glass_type") or row.get("type"), "Unspecified glass")
        grouped.setdefault(glass_type, []).append(row)
    return list(grouped.items())


def build_manual_processing_pdf(order: Dict[str, Any]) -> bytes:
    """Build the Manual Orders-only workshop slip.

    Dimensions are shown in centimetres to match the handwritten factory format.
    Each glass type starts on its own 100 x 210 mm slip.
    """

    rows = list(order.get("rows") or [])
    if not rows:
        raise ValueError("Manual order has no rows")

    groups = _group_rows(rows)
    rows_per_page = 15
    page_specs: List[tuple[str, Sequence[Dict[str, Any]], int, int]] = []
    for glass_type, glass_rows in groups:
        chunks = [
            glass_rows[index : index + rows_per_page]
            for index in range(0, len(glass_rows), rows_per_page)
        ]
        for chunk_index, chunk in enumerate(chunks, start=1):
            page_specs.append((glass_type, chunk, chunk_index, len(chunks)))

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=MANUAL_PROCESSING_PAGE_SIZE, pageCompression=1)
    page_width, page_height = MANUAL_PROCESSING_PAGE_SIZE
    margin = 7 * mm
    usable_width = page_width - (margin * 2)
    order_number = _pdf_text(order.get("order_number"), "Manual order")
    client_name = _pdf_text(order.get("client_name"), "-")
    order_date = _display_date(order.get("order_date"))

    for page_index, (glass_type, page_rows, group_page, group_pages) in enumerate(page_specs, start=1):
        pdf.setFillColor(colors.HexColor("#FFFFFF"))
        pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

        y = page_height - margin
        pdf.setFillColor(colors.HexColor("#667085"))
        pdf.setFont("Helvetica-Bold", 7.5)
        pdf.drawString(margin, y, "MANUAL PROCESSING")
        pdf.setFont("Helvetica", 7.5)
        pdf.drawRightString(page_width - margin, y, order_date)

        y -= 7 * mm
        pdf.setFillColor(colors.HexColor("#101828"))
        pdf.setFont("Helvetica-Bold", 12)
        _draw_fitted_text(
            pdf,
            order_number,
            x=margin,
            y=y,
            max_width=usable_width,
            font="Helvetica-Bold",
            size=12,
        )

        y -= 5.5 * mm
        pdf.setFillColor(colors.HexColor("#344054"))
        pdf.setFont("Helvetica", 9)
        _draw_fitted_text(
            pdf,
            client_name,
            x=margin,
            y=y,
            max_width=usable_width,
            font="Helvetica",
            size=9,
        )

        y -= 4 * mm
        pdf.setStrokeColor(colors.HexColor("#101828"))
        pdf.setLineWidth(0.8)
        pdf.line(margin, y, page_width - margin, y)

        y -= 6 * mm
        pdf.setFillColor(colors.HexColor("#667085"))
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawString(margin, y, "GLASS TYPE")
        if group_pages > 1:
            pdf.drawRightString(page_width - margin, y, f"{group_page}/{group_pages}")

        y -= 6.5 * mm
        pdf.setFillColor(colors.HexColor("#101828"))
        pdf.setFont("Helvetica-Bold", 14)
        _draw_fitted_text(
            pdf,
            glass_type,
            x=margin,
            y=y,
            max_width=usable_width,
            font="Helvetica-Bold",
            size=14,
            min_size=8,
        )

        y -= 6 * mm
        pdf.setFillColor(colors.HexColor("#667085"))
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawString(margin, y, "POSITION")
        pdf.drawString(margin + 21 * mm, y, "DIMENSIONS (CM)")
        pdf.drawRightString(page_width - margin, y, "QTY")

        y -= 4 * mm
        pdf.setStrokeColor(colors.HexColor("#D0D5DD"))
        pdf.setLineWidth(0.5)
        pdf.line(margin, y, page_width - margin, y)

        for row in page_rows:
            y -= 8.5 * mm
            position = _pdf_text(row.get("position"), "-")
            dimensions = f"{_format_cm(row.get('width_mm'))} x {_format_cm(row.get('height_mm'))}"
            quantity = max(1, int(row.get("quantity") or 1))

            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont("Helvetica-Bold", 11)
            _draw_fitted_text(
                pdf,
                f"{position})",
                x=margin,
                y=y,
                max_width=18 * mm,
                font="Helvetica-Bold",
                size=11,
                min_size=8,
            )
            pdf.setFont("Courier-Bold", 12)
            _draw_fitted_text(
                pdf,
                dimensions,
                x=margin + 21 * mm,
                y=y,
                max_width=50 * mm,
                font="Courier-Bold",
                size=12,
                min_size=9,
            )
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawRightString(page_width - margin, y, f"x {quantity}")

            notes = _pdf_text(row.get("notes"))
            if notes:
                y -= 3.5 * mm
                pdf.setFillColor(colors.HexColor("#667085"))
                pdf.setFont("Helvetica-Oblique", 7)
                _draw_fitted_text(
                    pdf,
                    notes,
                    x=margin + 21 * mm,
                    y=y,
                    max_width=usable_width - 21 * mm,
                    font="Helvetica-Oblique",
                    size=7,
                )
                y += 3.5 * mm

            pdf.setStrokeColor(colors.HexColor("#EAECF0"))
            pdf.setLineWidth(0.35)
            pdf.line(margin, y - 2.5 * mm, page_width - margin, y - 2.5 * mm)

        pdf.setFillColor(colors.HexColor("#98A2B3"))
        pdf.setFont("Helvetica", 6.5)
        pdf.drawString(margin, 5 * mm, "Manual Orders")
        pdf.drawRightString(page_width - margin, 5 * mm, f"{page_index}/{len(page_specs)}")
        pdf.showPage()

    pdf.save()
    return output.getvalue()


def build_manual_labels_pdf(order: Dict[str, Any]) -> bytes:
    """Build one dedicated 100 x 40 mm label per manual-order piece."""

    rows = list(order.get("rows") or [])
    if not rows:
        raise ValueError("Manual order has no rows")

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=MANUAL_LABEL_PAGE_SIZE, pageCompression=1)
    page_width, page_height = MANUAL_LABEL_PAGE_SIZE
    margin = 5.5 * mm
    usable_width = page_width - (margin * 2)
    order_number = _pdf_text(order.get("order_number"), "Manual order")
    client_name = _pdf_text(order.get("client_name"), "-")
    order_date = _display_date(order.get("order_date"))

    for row in rows:
        quantity = max(1, int(row.get("quantity") or 1))
        position = _pdf_text(row.get("position"), "-")
        glass_type = _pdf_text(row.get("glass_type") or row.get("type"), "-")
        dimensions = f"{_format_mm(row.get('width_mm'))} x {_format_mm(row.get('height_mm'))} mm"

        for _piece_index in range(1, quantity + 1):
            pdf.setFillColor(colors.white)
            pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

            top_y = page_height - margin - 1.5 * mm
            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont("Helvetica-Bold", 10)
            _draw_fitted_text(
                pdf,
                order_number,
                x=margin,
                y=top_y,
                max_width=51 * mm,
                font="Helvetica-Bold",
                size=10,
                min_size=7,
            )
            pdf.setFillColor(colors.HexColor("#475467"))
            pdf.setFont("Helvetica", 7)
            pdf.drawRightString(page_width - margin, top_y + 0.5, order_date)

            second_y = top_y - 4.5 * mm
            pdf.setFont("Helvetica", 8)
            _draw_fitted_text(
                pdf,
                client_name,
                x=margin,
                y=second_y,
                max_width=57 * mm,
                font="Helvetica",
                size=8,
                min_size=6,
            )
            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont("Helvetica-Bold", 10.5)
            pdf.drawRightString(
                page_width - margin,
                second_y,
                f"POS {position}",
            )

            rule_y = second_y - 2.2 * mm
            pdf.setStrokeColor(colors.HexColor("#98A2B3"))
            pdf.setLineWidth(0.55)
            pdf.line(margin, rule_y, page_width - margin, rule_y)

            dimension_y = 14.5 * mm
            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont("Helvetica-Bold", 16)
            _draw_fitted_text(
                pdf,
                dimensions,
                x=page_width / 2,
                y=dimension_y,
                max_width=usable_width,
                font="Helvetica-Bold",
                size=16,
                min_size=11,
                align="center",
            )

            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont("Helvetica-Bold", 9)
            _draw_fitted_text(
                pdf,
                glass_type,
                x=margin,
                y=margin - 0.5 * mm,
                max_width=70 * mm,
                font="Helvetica-Bold",
                size=9,
                min_size=7,
            )
            pdf.setFillColor(colors.HexColor("#667085"))
            pdf.setFont("Helvetica-Bold", 6.5)
            pdf.drawRightString(page_width - margin, margin - 0.3 * mm, "MANUAL")
            pdf.showPage()

    pdf.save()
    return output.getvalue()


__all__ = [
    "MANUAL_LABEL_PAGE_SIZE",
    "MANUAL_PROCESSING_PAGE_SIZE",
    "build_manual_labels_pdf",
    "build_manual_processing_pdf",
]
