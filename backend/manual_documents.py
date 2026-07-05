from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


MANUAL_PROCESSING_PAGE_SIZE = (100 * mm, 210 * mm)
MANUAL_LABEL_PAGE_SIZE = (100 * mm, 40 * mm)

DEFAULT_MANUAL_PRINT_SETTINGS: Dict[str, Any] = {
    "label_font_family": "Helvetica",
    "label_margin_mm": 5.5,
    "label_order_size": 9.5,
    "label_client_size": 12.5,
    "label_position_size": 10.5,
    "label_dimension_size": 16.0,
    "label_glass_size": 9.0,
    "label_client_bold": True,
    "label_dimension_bold": True,
    "label_show_date": True,
    "label_show_manual_marker": True,
    "label_show_divider": True,
    "processing_font_family": "Helvetica",
    "processing_page_width_mm": 100.0,
    "processing_page_height_mm": 210.0,
    "processing_margin_mm": 7.0,
    "processing_header_size": 7.5,
    "processing_order_size": 12.0,
    "processing_client_size": 9.0,
    "processing_glass_size": 14.0,
    "processing_row_size": 12.0,
    "processing_dimension_unit": "cm",
    "processing_glass_bold": True,
    "processing_rows_bold": True,
    "processing_show_date": True,
    "processing_show_client": True,
    "processing_show_notes": True,
    "processing_row_separators": True,
    "processing_show_footer": True,
}

_FONT_FAMILIES = {
    "Helvetica": {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "italic": "Helvetica-Oblique",
    },
    "Times": {
        "regular": "Times-Roman",
        "bold": "Times-Bold",
        "italic": "Times-Italic",
    },
    "Courier": {
        "regular": "Courier",
        "bold": "Courier-Bold",
        "italic": "Courier-Oblique",
    },
}

_NUMBER_LIMITS = {
    "label_margin_mm": (3.0, 8.0),
    "label_order_size": (7.0, 13.0),
    "label_client_size": (8.0, 16.0),
    "label_position_size": (8.0, 14.0),
    "label_dimension_size": (11.0, 21.0),
    "label_glass_size": (7.0, 13.0),
    "processing_page_width_mm": (70.0, 140.0),
    "processing_page_height_mm": (100.0, 297.0),
    "processing_margin_mm": (4.0, 15.0),
    "processing_header_size": (6.0, 11.0),
    "processing_order_size": (8.0, 18.0),
    "processing_client_size": (7.0, 15.0),
    "processing_glass_size": (9.0, 20.0),
    "processing_row_size": (9.0, 18.0),
}

_BOOLEAN_KEYS = {
    key for key, value in DEFAULT_MANUAL_PRINT_SETTINGS.items() if isinstance(value, bool)
}


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return fallback


def normalize_manual_print_settings(values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = values if isinstance(values, dict) else {}
    normalized = dict(DEFAULT_MANUAL_PRINT_SETTINGS)
    for key, fallback in DEFAULT_MANUAL_PRINT_SETTINGS.items():
        if key not in source:
            continue
        value = source[key]
        if key in _BOOLEAN_KEYS:
            normalized[key] = _as_bool(value, fallback)
        elif key in _NUMBER_LIMITS:
            minimum, maximum = _NUMBER_LIMITS[key]
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float(fallback)
            normalized[key] = round(max(minimum, min(maximum, number)), 2)
        elif key.endswith("_font_family"):
            normalized[key] = value if value in _FONT_FAMILIES else fallback
        elif key == "processing_dimension_unit":
            normalized[key] = "mm" if str(value).lower() == "mm" else "cm"

    max_processing_margin = max(
        4.0,
        min(
            normalized["processing_page_width_mm"] / 4,
            normalized["processing_page_height_mm"] / 6,
        ),
    )
    normalized["processing_margin_mm"] = min(
        normalized["processing_margin_mm"],
        round(max_processing_margin, 2),
    )
    return normalized


def _font(settings: Dict[str, Any], document: str, style: str = "regular") -> str:
    family = settings.get(f"{document}_font_family", "Helvetica")
    return _FONT_FAMILIES.get(family, _FONT_FAMILIES["Helvetica"])[style]


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


def _row_final_area_m2(row: Dict[str, Any]) -> float:
    for key in ("final_area_m2", "area_override_m2"):
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    try:
        width_mm = float(row.get("width_mm") or 0)
        height_mm = float(row.get("height_mm") or 0)
        quantity = max(1, int(row.get("quantity") or 1))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, width_mm * height_mm * quantity / 1_000_000)


def _format_area_m2(rows: Iterable[Dict[str, Any]]) -> str:
    total = sum(_row_final_area_m2(row) for row in rows)
    return f"{total:.3f} m²"


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


def build_manual_processing_pdf(
    order: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build the Manual Orders-only workshop slip.

    Dimensions use the configured workshop unit (centimetres by default).
    Glass-type sections share a slip whenever they fit.
    """

    rows = list(order.get("rows") or [])
    if not rows:
        raise ValueError("Manual order has no rows")

    config = normalize_manual_print_settings(settings)
    page_width = config["processing_page_width_mm"] * mm
    page_height = config["processing_page_height_mm"] * mm
    margin = config["processing_margin_mm"] * mm
    page_size = (page_width, page_height)
    row_step_mm = max(8.5, config["processing_row_size"] * 0.55)
    groups = _group_rows(rows)

    def row_height_mm(row: Dict[str, Any]) -> float:
        notes_height = (
            3.5
            if config["processing_show_notes"] and _pdf_text(row.get("notes"))
            else 0
        )
        return row_step_mm + notes_height

    section_header_mm = 22.5
    bottom_guard_mm = max(config["processing_margin_mm"], 10.0)
    page_content_mm = max(
        section_header_mm + row_step_mm,
        config["processing_page_height_mm"]
        - config["processing_margin_mm"]
        - 16.5
        - bottom_guard_mm,
    )
    page_specs: List[List[tuple[str, Sequence[Dict[str, Any]]]]] = []
    current_page: List[tuple[str, Sequence[Dict[str, Any]]]] = []
    remaining_mm = page_content_mm

    for glass_type, glass_rows in groups:
        row_index = 0
        while row_index < len(glass_rows):
            minimum_section_mm = section_header_mm + row_height_mm(glass_rows[row_index])
            if current_page and remaining_mm < minimum_section_mm:
                page_specs.append(current_page)
                current_page = []
                remaining_mm = page_content_mm

            chunk: List[Dict[str, Any]] = []
            chunk_height_mm = section_header_mm
            while row_index < len(glass_rows):
                next_row_height_mm = row_height_mm(glass_rows[row_index])
                if chunk and chunk_height_mm + next_row_height_mm > remaining_mm:
                    break
                if not chunk and chunk_height_mm + next_row_height_mm > remaining_mm:
                    break
                chunk.append(glass_rows[row_index])
                chunk_height_mm += next_row_height_mm
                row_index += 1

            if not chunk:
                if current_page:
                    page_specs.append(current_page)
                    current_page = []
                    remaining_mm = page_content_mm
                    continue
                chunk.append(glass_rows[row_index])
                chunk_height_mm += row_height_mm(glass_rows[row_index])
                row_index += 1

            current_page.append((glass_type, chunk))
            remaining_mm -= chunk_height_mm

            if row_index < len(glass_rows):
                page_specs.append(current_page)
                current_page = []
                remaining_mm = page_content_mm

    if current_page:
        page_specs.append(current_page)

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=page_size, pageCompression=1)
    usable_width = page_width - (margin * 2)
    regular_font = _font(config, "processing", "regular")
    bold_font = _font(config, "processing", "bold")
    italic_font = _font(config, "processing", "italic")
    glass_font = bold_font if config["processing_glass_bold"] else regular_font
    row_font = bold_font if config["processing_rows_bold"] else regular_font
    order_number = _pdf_text(order.get("order_number"), "Manual order")
    client_name = _pdf_text(order.get("client_name"), "-")
    order_date = _display_date(order.get("order_date"))

    for page_index, page_sections in enumerate(page_specs, start=1):
        pdf.setFillColor(colors.HexColor("#FFFFFF"))
        pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

        y = page_height - margin
        pdf.setFillColor(colors.HexColor("#667085"))
        pdf.setFont(bold_font, config["processing_header_size"])
        pdf.drawString(margin, y, "MANUAL PROCESSING")
        if config["processing_show_date"]:
            pdf.setFont(regular_font, config["processing_header_size"])
            pdf.drawRightString(page_width - margin, y, order_date)

        y -= 7 * mm
        pdf.setFillColor(colors.HexColor("#101828"))
        pdf.setFont(bold_font, config["processing_order_size"])
        _draw_fitted_text(
            pdf,
            order_number,
            x=margin,
            y=y,
            max_width=usable_width,
            font=bold_font,
            size=config["processing_order_size"],
        )

        y -= 5.5 * mm
        if config["processing_show_client"]:
            pdf.setFillColor(colors.HexColor("#344054"))
            pdf.setFont(regular_font, config["processing_client_size"])
            _draw_fitted_text(
                pdf,
                client_name,
                x=margin,
                y=y,
                max_width=usable_width,
                font=regular_font,
                size=config["processing_client_size"],
            )

        y -= 4 * mm
        pdf.setStrokeColor(colors.HexColor("#101828"))
        pdf.setLineWidth(0.8)
        pdf.line(margin, y, page_width - margin, y)

        for glass_type, page_rows in page_sections:
            y -= 6 * mm
            pdf.setFillColor(colors.HexColor("#667085"))
            pdf.setFont(bold_font, config["processing_header_size"])
            pdf.drawString(margin, y, "GLASS TYPE")
            pdf.drawRightString(page_width - margin, y, "AREA")

            y -= 6.5 * mm
            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont(glass_font, config["processing_glass_size"])
            area_width = usable_width * 0.30
            _draw_fitted_text(
                pdf,
                glass_type,
                x=margin,
                y=y,
                max_width=usable_width - area_width - (3 * mm),
                font=glass_font,
                size=config["processing_glass_size"],
                min_size=8,
            )
            area_size = max(9.0, config["processing_glass_size"] - 2)
            pdf.setFont(bold_font, area_size)
            _draw_fitted_text(
                pdf,
                _format_area_m2(page_rows),
                x=page_width - margin,
                y=y,
                max_width=area_width,
                font=bold_font,
                size=area_size,
                min_size=8,
                align="right",
            )

            y -= 6 * mm
            pdf.setFillColor(colors.HexColor("#667085"))
            pdf.setFont(bold_font, config["processing_header_size"])
            pdf.drawString(margin, y, "POSITION")
            dimension_unit = config["processing_dimension_unit"].upper()
            dimension_x = margin + (usable_width * 0.24)
            pdf.drawString(dimension_x, y, f"DIMENSIONS ({dimension_unit})")
            pdf.drawRightString(page_width - margin, y, "QTY")

            y -= 4 * mm
            pdf.setStrokeColor(colors.HexColor("#D0D5DD"))
            pdf.setLineWidth(0.5)
            pdf.line(margin, y, page_width - margin, y)

            for row in page_rows:
                y -= row_step_mm * mm
                position = _pdf_text(row.get("position"), "-")
                formatter = _format_mm if config["processing_dimension_unit"] == "mm" else _format_cm
                dimensions = f"{formatter(row.get('width_mm'))} x {formatter(row.get('height_mm'))}"
                quantity = max(1, int(row.get("quantity") or 1))

                pdf.setFillColor(colors.HexColor("#101828"))
                position_size = max(8.0, config["processing_row_size"] - 1)
                pdf.setFont(row_font, position_size)
                _draw_fitted_text(
                    pdf,
                    f"{position})",
                    x=margin,
                    y=y,
                    max_width=usable_width * 0.20,
                    font=row_font,
                    size=position_size,
                    min_size=8,
                )
                pdf.setFont(row_font, config["processing_row_size"])
                _draw_fitted_text(
                    pdf,
                    dimensions,
                    x=dimension_x,
                    y=y,
                    max_width=usable_width * 0.57,
                    font=row_font,
                    size=config["processing_row_size"],
                    min_size=9,
                )
                pdf.setFont(row_font, config["processing_row_size"])
                pdf.drawRightString(page_width - margin, y, f"x {quantity}")

                notes = _pdf_text(row.get("notes"))
                if notes and config["processing_show_notes"]:
                    y -= 3.5 * mm
                    pdf.setFillColor(colors.HexColor("#667085"))
                    note_size = max(6.0, config["processing_header_size"])
                    pdf.setFont(italic_font, note_size)
                    _draw_fitted_text(
                        pdf,
                        notes,
                        x=dimension_x,
                        y=y,
                        max_width=(page_width - margin) - dimension_x,
                        font=italic_font,
                        size=note_size,
                    )

                if config["processing_row_separators"]:
                    pdf.setStrokeColor(colors.HexColor("#EAECF0"))
                    pdf.setLineWidth(0.35)
                    pdf.line(margin, y - 2.5 * mm, page_width - margin, y - 2.5 * mm)

        if config["processing_show_footer"]:
            pdf.setFillColor(colors.HexColor("#98A2B3"))
            pdf.setFont(regular_font, 6.5)
            footer_y = max(3 * mm, margin * 0.7)
            pdf.drawString(margin, footer_y, "Manual Orders")
            pdf.drawRightString(page_width - margin, footer_y, f"{page_index}/{len(page_specs)}")
        pdf.showPage()

    pdf.save()
    return output.getvalue()


def build_manual_labels_pdf(
    order: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build one dedicated 100 x 40 mm label per manual-order piece."""

    rows = list(order.get("rows") or [])
    if not rows:
        raise ValueError("Manual order has no rows")

    config = normalize_manual_print_settings(settings)
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=MANUAL_LABEL_PAGE_SIZE, pageCompression=1)
    page_width, page_height = MANUAL_LABEL_PAGE_SIZE
    margin = config["label_margin_mm"] * mm
    usable_width = page_width - (margin * 2)
    regular_font = _font(config, "label", "regular")
    bold_font = _font(config, "label", "bold")
    client_font = bold_font if config["label_client_bold"] else regular_font
    dimension_font = bold_font if config["label_dimension_bold"] else regular_font
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
            pdf.setFont(bold_font, config["label_order_size"])
            _draw_fitted_text(
                pdf,
                order_number,
                x=margin,
                y=top_y,
                max_width=26 * mm,
                font=bold_font,
                size=config["label_order_size"],
                min_size=7,
            )
            pdf.setFont(client_font, config["label_client_size"])
            _draw_fitted_text(
                pdf,
                client_name,
                x=page_width / 2,
                y=top_y,
                max_width=36 * mm,
                font=client_font,
                size=config["label_client_size"],
                min_size=8,
                align="center",
            )
            pdf.setFont(bold_font, config["label_position_size"])
            _draw_fitted_text(
                pdf,
                f"POS {position}",
                x=page_width - margin,
                y=top_y,
                max_width=22 * mm,
                font=bold_font,
                size=config["label_position_size"],
                min_size=8,
                align="right",
            )

            second_y = top_y - 4.5 * mm
            if config["label_show_date"]:
                pdf.setFillColor(colors.HexColor("#475467"))
                pdf.setFont(regular_font, 7)
                pdf.drawString(margin, second_y, order_date)

            rule_y = second_y - 2.2 * mm
            if config["label_show_divider"]:
                pdf.setStrokeColor(colors.HexColor("#98A2B3"))
                pdf.setLineWidth(0.55)
                pdf.line(margin, rule_y, page_width - margin, rule_y)

            dimension_y = 14.5 * mm
            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont(dimension_font, config["label_dimension_size"])
            _draw_fitted_text(
                pdf,
                dimensions,
                x=page_width / 2,
                y=dimension_y,
                max_width=usable_width,
                font=dimension_font,
                size=config["label_dimension_size"],
                min_size=11,
                align="center",
            )

            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont(bold_font, config["label_glass_size"])
            _draw_fitted_text(
                pdf,
                glass_type,
                x=margin,
                y=margin - 0.5 * mm,
                max_width=70 * mm,
                font=bold_font,
                size=config["label_glass_size"],
                min_size=7,
            )
            if config["label_show_manual_marker"]:
                pdf.setFillColor(colors.HexColor("#667085"))
                pdf.setFont(bold_font, 6.5)
                pdf.drawRightString(page_width - margin, margin - 0.3 * mm, "MANUAL")
            pdf.showPage()

    pdf.save()
    return output.getvalue()


__all__ = [
    "MANUAL_LABEL_PAGE_SIZE",
    "MANUAL_PROCESSING_PAGE_SIZE",
    "DEFAULT_MANUAL_PRINT_SETTINGS",
    "build_manual_labels_pdf",
    "build_manual_processing_pdf",
    "normalize_manual_print_settings",
]
