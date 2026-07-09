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
A4_PORTRAIT_PAGE_SIZE = (210 * mm, 297 * mm)
A4_LANDSCAPE_PAGE_SIZE = (297 * mm, 210 * mm)
PROCESSING_PRINT_LAYOUT_SLIP = "slip"
PROCESSING_PRINT_LAYOUT_A4_PORTRAIT = "a4_portrait"
PROCESSING_PRINT_LAYOUT_A4_2UP = "a4_landscape_2up"

DEFAULT_MANUAL_PRINT_SETTINGS: Dict[str, Any] = {
    "label_font_family": "Helvetica",
    "label_margin_mm": 5.5,
    "label_order_size": 9.5,
    "label_client_size": 14.0,
    "label_position_size": 10.5,
    "label_dimension_size": 14.5,
    "label_glass_size": 9.0,
    "label_date_size": 7.0,
    "label_section_size": 6.5,
    "label_index_size": 14.0,
    "label_top_offset_mm": 3.0,
    "label_second_line_gap_mm": 4.5,
    "label_dimension_y_mm": 15.0,
    "label_bottom_offset_mm": 0.5,
    "label_order_width_mm": 26.0,
    "label_client_width_mm": 36.0,
    "label_position_width_mm": 28.0,
    "label_section_width_mm": 35.0,
    "label_glass_width_mm": 70.0,
    "label_index_width_mm": 22.0,
    "label_client_bold": True,
    "label_dimension_bold": True,
    "label_show_date": True,
    "label_show_manual_marker": False,
    "label_show_divider": True,
    "processing_font_family": "Helvetica",
    "processing_print_layout": PROCESSING_PRINT_LAYOUT_A4_PORTRAIT,
    "processing_show_cut_guide": True,
    "processing_page_width_mm": 100.0,
    "processing_page_height_mm": 210.0,
    "processing_margin_mm": 7.0,
    "processing_header_size": 7.5,
    "processing_order_size": 12.0,
    "processing_client_size": 14.0,
    "processing_client_bold": True,
    "processing_glass_size": 14.0,
    "processing_section_size": 9.5,
    "processing_section_before_gap_mm": 5.0,
    "processing_section_after_gap_mm": 4.0,
    "processing_row_size": 12.0,
    "processing_row_spacing_mm": 0.0,
    "processing_dimension_unit": "cm",
    "processing_glass_bold": True,
    "processing_rows_bold": True,
    "processing_repeat_headers_per_section": False,
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
    "label_date_size": (5.0, 10.0),
    "label_section_size": (5.0, 10.0),
    "label_index_size": (9.0, 20.0),
    "label_top_offset_mm": (0.0, 8.0),
    "label_second_line_gap_mm": (2.0, 8.0),
    "label_dimension_y_mm": (10.0, 22.0),
    "label_bottom_offset_mm": (0.0, 5.0),
    "label_order_width_mm": (16.0, 42.0),
    "label_client_width_mm": (24.0, 60.0),
    "label_position_width_mm": (16.0, 42.0),
    "label_section_width_mm": (16.0, 50.0),
    "label_glass_width_mm": (30.0, 88.0),
    "label_index_width_mm": (12.0, 36.0),
    "processing_page_width_mm": (70.0, 210.0),
    "processing_page_height_mm": (100.0, 297.0),
    "processing_margin_mm": (4.0, 15.0),
    "processing_header_size": (6.0, 11.0),
    "processing_order_size": (8.0, 18.0),
    "processing_client_size": (7.0, 15.0),
    "processing_glass_size": (9.0, 20.0),
    "processing_section_size": (6.0, 14.0),
    "processing_section_before_gap_mm": (0.0, 12.0),
    "processing_section_after_gap_mm": (0.0, 12.0),
    "processing_row_size": (9.0, 18.0),
    "processing_row_spacing_mm": (0.0, 6.0),
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
        elif key == "processing_print_layout":
            normalized[key] = (
                value
                if value in {
                    PROCESSING_PRINT_LAYOUT_SLIP,
                    PROCESSING_PRINT_LAYOUT_A4_PORTRAIT,
                    PROCESSING_PRINT_LAYOUT_A4_2UP,
                }
                else fallback
            )
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
    if (
        "processing_client_bold" not in source
        and source.get("processing_client_size") == 9
    ):
        normalized["processing_client_size"] = 14.0
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


def _group_red_index_rows(rows: Iterable[Dict[str, Any]]) -> List[tuple[str, List[tuple[str, List[Dict[str, Any]]]]]]:
    grouped: "OrderedDict[str, OrderedDict[str, List[Dict[str, Any]]]]" = OrderedDict()
    for row in rows:
        section = _pdf_text(row.get("section"))
        glass_type = _pdf_text(row.get("glass_type") or row.get("type"), "Unspecified glass")
        grouped.setdefault(glass_type, OrderedDict()).setdefault(section, []).append(row)
    return [
        (glass_type, list(section_rows.items()))
        for glass_type, section_rows in grouped.items()
    ]


def _processing_page_geometry(config: Dict[str, Any]) -> tuple[bool, float, float, float, float, tuple[float, float]]:
    layout = config["processing_print_layout"]
    is_a4_two_up = layout == PROCESSING_PRINT_LAYOUT_A4_2UP
    if is_a4_two_up:
        processing_width_mm = 100.0
        processing_height_mm = 210.0
        output_page_size = A4_LANDSCAPE_PAGE_SIZE
    elif layout == PROCESSING_PRINT_LAYOUT_A4_PORTRAIT:
        processing_width_mm = 210.0
        processing_height_mm = 297.0
        output_page_size = A4_PORTRAIT_PAGE_SIZE
    else:
        processing_width_mm = config["processing_page_width_mm"]
        processing_height_mm = config["processing_page_height_mm"]
        output_page_size = (processing_width_mm * mm, processing_height_mm * mm)
    page_width = processing_width_mm * mm
    page_height = processing_height_mm * mm
    return is_a4_two_up, processing_width_mm, processing_height_mm, page_width, page_height, output_page_size


def _build_red_index_processing_pdf(
    order: Dict[str, Any],
    config: Dict[str, Any],
) -> bytes:
    rows = list(order.get("rows") or [])
    (
        is_a4_two_up,
        processing_width_mm,
        processing_height_mm,
        page_width,
        page_height,
        output_page_size,
    ) = _processing_page_geometry(config)
    margin = config["processing_margin_mm"] * mm
    row_step_mm = max(
        5.5,
        config["processing_row_size"] * 0.46 + config["processing_row_spacing_mm"],
    )
    groups = _group_red_index_rows(rows)

    def row_height_mm(row: Dict[str, Any]) -> float:
        notes_height = (
            3.0
            if config["processing_show_notes"] and _pdf_text(row.get("notes"))
            else 0
        )
        return row_step_mm + notes_height

    def chunk_header_height_mm(show_glass: bool, section: str) -> float:
        height = 19.0 if show_glass else 0.0
        if section:
            height += config["processing_section_before_gap_mm"]
            height += max(3.5, config["processing_section_size"] * 0.35)
            height += config["processing_section_after_gap_mm"]
        if show_glass or config["processing_repeat_headers_per_section"]:
            height += 8.5
        return max(5.0, height)

    bottom_guard_mm = max(config["processing_margin_mm"], 10.0)
    minimum_header_mm = max(
        chunk_header_height_mm(True, section)
        for _glass, section_groups in groups
        for section, _rows in section_groups
    ) if groups else 19.0
    page_content_mm = max(
        minimum_header_mm + row_step_mm,
        processing_height_mm
        - config["processing_margin_mm"]
        - 16.5
        - bottom_guard_mm,
    )
    page_specs: List[List[tuple[str, str, Sequence[Dict[str, Any]], bool]]] = []
    current_page: List[tuple[str, str, Sequence[Dict[str, Any]], bool]] = []
    remaining_mm = page_content_mm

    for glass_type, section_groups in groups:
        show_glass_for_next_chunk = True
        for section, glass_rows in section_groups:
            row_index = 0
            while row_index < len(glass_rows):
                header_height_mm = chunk_header_height_mm(show_glass_for_next_chunk, section)
                minimum_section_mm = header_height_mm + row_height_mm(glass_rows[row_index])
                if current_page and remaining_mm < minimum_section_mm:
                    page_specs.append(current_page)
                    current_page = []
                    remaining_mm = page_content_mm
                    show_glass_for_next_chunk = True
                    header_height_mm = chunk_header_height_mm(show_glass_for_next_chunk, section)

                chunk: List[Dict[str, Any]] = []
                chunk_height_mm = header_height_mm
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
                        show_glass_for_next_chunk = True
                        continue
                    chunk.append(glass_rows[row_index])
                    chunk_height_mm += row_height_mm(glass_rows[row_index])
                    row_index += 1

                current_page.append((section, glass_type, chunk, show_glass_for_next_chunk))
                remaining_mm -= chunk_height_mm
                show_glass_for_next_chunk = False

                if row_index < len(glass_rows):
                    page_specs.append(current_page)
                    current_page = []
                    remaining_mm = page_content_mm
                    show_glass_for_next_chunk = True

    if current_page:
        page_specs.append(current_page)

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=output_page_size, pageCompression=1)
    usable_width = page_width - (margin * 2)
    regular_font = _font(config, "processing", "regular")
    bold_font = _font(config, "processing", "bold")
    italic_font = _font(config, "processing", "italic")
    row_font = bold_font if config["processing_rows_bold"] else regular_font
    client_font = bold_font if config["processing_client_bold"] else regular_font
    glass_font = bold_font if config["processing_glass_bold"] else regular_font
    order_number = _pdf_text(order.get("order_number"), "Manual order")
    client_name = _pdf_text(order.get("client_name"), "-")
    order_date = _display_date(order.get("order_date"))
    dimension_unit = config["processing_dimension_unit"].upper()

    for page_index, page_sections in enumerate(page_specs, start=1):
        form_name = f"manual_red_index_slip_{page_index}"
        if is_a4_two_up:
            pdf.beginForm(form_name, 0, 0, page_width, page_height)

        pdf.setFillColor(colors.white)
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
            pdf.setFont(client_font, config["processing_client_size"])
            _draw_fitted_text(
                pdf,
                client_name,
                x=margin,
                y=y,
                max_width=usable_width,
                font=client_font,
                size=config["processing_client_size"],
            )

        y -= 4 * mm
        pdf.setStrokeColor(colors.HexColor("#101828"))
        pdf.setLineWidth(0.8)
        pdf.line(margin, y, page_width - margin, y)

        for section, glass_type, page_rows, show_glass in page_sections:
            y -= 10 * mm if show_glass else config["processing_section_before_gap_mm"] * mm
            if show_glass:
                pdf.setFillColor(colors.HexColor("#101828"))
                pdf.setFont(glass_font, config["processing_glass_size"])
                area_width = usable_width * 0.26
                glass_total_rows = [
                    row
                    for _page_section, page_glass, section_rows, _show_glass in page_sections
                    if page_glass == glass_type
                    for row in section_rows
                ]
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
                    _format_area_m2(glass_total_rows),
                    x=page_width - margin,
                    y=y,
                    max_width=area_width,
                    font=bold_font,
                    size=area_size,
                    min_size=8,
                    align="right",
                )
                y -= 5 * mm
            if section:
                # The first section follows a glass heading; give it the same
                # configurable separation used by later section headings.
                if show_glass:
                    y -= config["processing_section_before_gap_mm"] * mm
                pdf.setFillColor(colors.HexColor("#667085"))
                section_size = config["processing_section_size"]
                pdf.setFont(bold_font, section_size)
                _draw_fitted_text(
                    pdf,
                    section,
                    x=margin,
                    y=y,
                    max_width=usable_width,
                    font=bold_font,
                    size=section_size,
                    min_size=7,
                )
                y -= config["processing_section_after_gap_mm"] * mm

            pos_x = margin
            index_x = margin + (usable_width * 0.15)
            dimension_x = margin + (usable_width * 0.31)
            if show_glass or config["processing_repeat_headers_per_section"]:
                y -= 5.5 * mm
                pdf.setFillColor(colors.HexColor("#667085"))
                pdf.setFont(bold_font, config["processing_header_size"])
                pdf.drawString(pos_x, y, "POS")
                pdf.drawString(index_x, y, "INDEX")
                pdf.drawString(dimension_x, y, f"DIMENSIONS ({dimension_unit})")
                pdf.drawRightString(page_width - margin, y, "QTY")
                y -= 3 * mm
                pdf.setStrokeColor(colors.HexColor("#D0D5DD"))
                pdf.setLineWidth(0.5)
                pdf.line(margin, y, page_width - margin, y)

            for row in page_rows:
                y -= row_step_mm * mm
                position_size = max(8.0, config["processing_row_size"] - 1)
                pdf.setFillColor(colors.HexColor("#101828"))
                pdf.setFont(row_font, position_size)
                _draw_fitted_text(
                    pdf,
                    _pdf_text(row.get("client_position"), "-"),
                    x=pos_x,
                    y=y,
                    max_width=index_x - pos_x - (3 * mm),
                    font=row_font,
                    size=position_size,
                    min_size=8,
                )
                pdf.setFillColor(colors.HexColor("#DC2626"))
                pdf.setFont(row_font, position_size)
                _draw_fitted_text(
                    pdf,
                    _pdf_text(row.get("index_number"), "-"),
                    x=index_x,
                    y=y,
                    max_width=dimension_x - index_x - (3 * mm),
                    font=row_font,
                    size=position_size,
                    min_size=8,
                )
                formatter = _format_mm if config["processing_dimension_unit"] == "mm" else _format_cm
                dimensions = f"{formatter(row.get('width_mm'))} x {formatter(row.get('height_mm'))}"
                pdf.setFillColor(colors.HexColor("#101828"))
                pdf.setFont(row_font, config["processing_row_size"])
                _draw_fitted_text(
                    pdf,
                    dimensions,
                    x=dimension_x,
                    y=y,
                    max_width=usable_width * 0.45,
                    font=row_font,
                    size=config["processing_row_size"],
                    min_size=9,
                )
                quantity = max(1, int(row.get("quantity") or 1))
                pdf.drawRightString(page_width - margin, y, f"x {quantity}")

                notes = _pdf_text(row.get("notes"))
                if notes and config["processing_show_notes"]:
                    y -= 3 * mm
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
                    pdf.line(margin, y - 2 * mm, page_width - margin, y - 2 * mm)

        if config["processing_show_footer"]:
            pdf.setFillColor(colors.HexColor("#98A2B3"))
            pdf.setFont(regular_font, 6.5)
            footer_y = max(3 * mm, margin * 0.7)
            pdf.drawString(margin, footer_y, "Manual Orders")
            pdf.drawRightString(page_width - margin, footer_y, f"{page_index}/{len(page_specs)}")

        if is_a4_two_up:
            pdf.endForm()
            output_width, output_height = A4_LANDSCAPE_PAGE_SIZE
            copies_gap = 8 * mm
            copies_width = (page_width * 2) + copies_gap
            first_copy_x = (output_width - copies_width) / 2
            second_copy_x = first_copy_x + page_width + copies_gap
            for copy_x in (first_copy_x, second_copy_x):
                pdf.saveState()
                pdf.translate(copy_x, 0)
                pdf.doForm(form_name)
                pdf.restoreState()
            if config["processing_show_cut_guide"]:
                pdf.saveState()
                pdf.setStrokeColor(colors.HexColor("#999999"))
                pdf.setLineWidth(0.2 * mm)
                pdf.setDash(2 * mm, 2 * mm)
                cut_x = output_width / 2
                pdf.line(cut_x, 3 * mm, cut_x, output_height - (3 * mm))
                pdf.restoreState()
        pdf.showPage()

    pdf.save()
    return output.getvalue()


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
    if order.get("manual_format") == "client_positions_red_index":
        return _build_red_index_processing_pdf(order, config)
    (
        is_a4_two_up,
        processing_width_mm,
        processing_height_mm,
        page_width,
        page_height,
        output_page_size,
    ) = _processing_page_geometry(config)
    margin = config["processing_margin_mm"] * mm
    row_step_mm = max(
        7.0,
        config["processing_row_size"] * 0.50 + config["processing_row_spacing_mm"],
    )
    groups = _group_rows(rows)

    def row_height_mm(row: Dict[str, Any]) -> float:
        notes_height = (
            3.5
            if config["processing_show_notes"] and _pdf_text(row.get("notes"))
            else 0
        )
        return row_step_mm + notes_height

    section_header_mm = 21.0
    bottom_guard_mm = max(config["processing_margin_mm"], 10.0)
    page_content_mm = max(
        section_header_mm + row_step_mm,
        processing_height_mm
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
    pdf = canvas.Canvas(output, pagesize=output_page_size, pageCompression=1)
    usable_width = page_width - (margin * 2)
    regular_font = _font(config, "processing", "regular")
    bold_font = _font(config, "processing", "bold")
    italic_font = _font(config, "processing", "italic")
    glass_font = bold_font if config["processing_glass_bold"] else regular_font
    row_font = bold_font if config["processing_rows_bold"] else regular_font
    client_font = bold_font if config["processing_client_bold"] else regular_font
    order_number = _pdf_text(order.get("order_number"), "Manual order")
    client_name = _pdf_text(order.get("client_name"), "-")
    order_date = _display_date(order.get("order_date"))

    for page_index, page_sections in enumerate(page_specs, start=1):
        form_name = f"manual_processing_slip_{page_index}"
        if is_a4_two_up:
            pdf.beginForm(form_name, 0, 0, page_width, page_height)

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
            pdf.setFont(client_font, config["processing_client_size"])
            _draw_fitted_text(
                pdf,
                client_name,
                x=margin,
                y=y,
                max_width=usable_width,
                font=client_font,
                size=config["processing_client_size"],
            )

        y -= 4 * mm
        pdf.setStrokeColor(colors.HexColor("#101828"))
        pdf.setLineWidth(0.8)
        pdf.line(margin, y, page_width - margin, y)

        for glass_type, page_rows in page_sections:
            y -= 11 * mm
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

        if is_a4_two_up:
            pdf.endForm()
            output_width, output_height = A4_LANDSCAPE_PAGE_SIZE
            copies_gap = 8 * mm
            copies_width = (page_width * 2) + copies_gap
            first_copy_x = (output_width - copies_width) / 2
            second_copy_x = first_copy_x + page_width + copies_gap
            for copy_x in (first_copy_x, second_copy_x):
                pdf.saveState()
                pdf.translate(copy_x, 0)
                pdf.doForm(form_name)
                pdf.restoreState()

            if config["processing_show_cut_guide"]:
                pdf.saveState()
                pdf.setStrokeColor(colors.HexColor("#999999"))
                pdf.setLineWidth(0.2 * mm)
                pdf.setDash(2 * mm, 2 * mm)
                cut_x = output_width / 2
                pdf.line(cut_x, 3 * mm, cut_x, output_height - (3 * mm))
                pdf.restoreState()
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
    red_index_format = order.get("manual_format") == "client_positions_red_index"

    for row in rows:
        quantity = max(1, int(row.get("quantity") or 1))
        position = _pdf_text(row.get("position"), "-")
        glass_type = _pdf_text(row.get("glass_type") or row.get("type"), "-")
        dimensions = f"{_format_mm(row.get('width_mm'))} x {_format_mm(row.get('height_mm'))} mm"

        for _piece_index in range(1, quantity + 1):
            pdf.setFillColor(colors.white)
            pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

            top_y = page_height - margin - config["label_top_offset_mm"] * mm
            pdf.setFillColor(colors.HexColor("#101828"))
            pdf.setFont(bold_font, config["label_order_size"])
            _draw_fitted_text(
                pdf,
                order_number,
                x=margin,
                y=top_y,
                max_width=config["label_order_width_mm"] * mm,
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
                max_width=config["label_client_width_mm"] * mm,
                font=client_font,
                size=config["label_client_size"],
                min_size=8,
                align="center",
            )
            if red_index_format:
                pdf.setFont(bold_font, config["label_position_size"])
                _draw_fitted_text(
                    pdf,
                    f"POS {_pdf_text(row.get('client_position'), '-')}",
                    x=page_width - margin,
                    y=top_y,
                    max_width=config["label_position_width_mm"] * mm,
                    font=bold_font,
                    size=config["label_position_size"],
                    min_size=7,
                    align="right",
                )
            else:
                pdf.setFont(bold_font, config["label_position_size"])
                _draw_fitted_text(
                    pdf,
                    f"POS {position}",
                    x=page_width - margin,
                    y=top_y,
                    max_width=config["label_position_width_mm"] * mm,
                    font=bold_font,
                    size=config["label_position_size"],
                    min_size=8,
                    align="right",
                )

            second_y = top_y - config["label_second_line_gap_mm"] * mm
            if config["label_show_date"]:
                pdf.setFillColor(colors.HexColor("#475467"))
                pdf.setFont(regular_font, config["label_date_size"])
                pdf.drawString(margin, second_y, order_date)
            section = _pdf_text(row.get("section"))
            if red_index_format and section:
                pdf.setFillColor(colors.HexColor("#475467"))
                pdf.setFont(bold_font, config["label_section_size"])
                _draw_fitted_text(
                    pdf,
                    section,
                    x=page_width - margin,
                    y=second_y,
                    max_width=config["label_section_width_mm"] * mm,
                    font=bold_font,
                    size=config["label_section_size"],
                    min_size=6,
                    align="right",
                )

            rule_y = second_y - 2.2 * mm
            if config["label_show_divider"]:
                pdf.setStrokeColor(colors.HexColor("#98A2B3"))
                pdf.setLineWidth(0.55)
                pdf.line(margin, rule_y, page_width - margin, rule_y)

            dimension_y = config["label_dimension_y_mm"] * mm
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
                y=margin - config["label_bottom_offset_mm"] * mm,
                max_width=config["label_glass_width_mm"] * mm,
                font=bold_font,
                size=config["label_glass_size"],
                min_size=7,
            )
            if red_index_format:
                index_size = config["label_index_size"]
                pdf.setFillColor(colors.HexColor("#DC2626"))
                pdf.setFont(bold_font, index_size)
                _draw_fitted_text(
                    pdf,
                    f"#{_pdf_text(row.get('index_number'), '-')}",
                    x=page_width - margin,
                    y=margin - config["label_bottom_offset_mm"] * mm,
                    max_width=config["label_index_width_mm"] * mm,
                    font=bold_font,
                    size=index_size,
                    min_size=9,
                    align="right",
                )
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
