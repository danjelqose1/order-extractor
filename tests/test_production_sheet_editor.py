from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "docs" / "index.html"
APP_JS = ROOT / "docs" / "js" / "app.js"
EDITOR_JS = ROOT / "docs" / "js" / "production-sheet-editor.js"
STYLES_CSS = ROOT / "docs" / "css" / "styles.css"


def _sources():
    return (
        INDEX_HTML.read_text(encoding="utf-8"),
        APP_JS.read_text(encoding="utf-8"),
        EDITOR_JS.read_text(encoding="utf-8"),
        STYLES_CSS.read_text(encoding="utf-8"),
    )


def test_processing_exposes_print_first_editor_after_generation():
    html, app_js, editor_js, _css = _sources()

    assert 'id="processingEditSheet" disabled>Edit Sheet</button>' in html
    assert 'id="processingPrintSheet" disabled>Print</button>' in html
    assert 'id="productionSheetPrint">Print</button>' in html
    assert app_js.index("processingEditSheetBtn.addEventListener") < app_js.index("processingExportPdfBtn.addEventListener")
    assert "if (!productionSheetEditor?.open())" in app_js
    assert "if (!productionSheetEditor?.print())" in app_js
    assert "window.open(\"\", \"_blank\")" in editor_js
    assert "printWindow.print()" in editor_js


def test_generated_sheet_is_a_separate_resettable_snapshot():
    _html, app_js, editor_js, _css = _sources()

    assert "this.generatedHtml = this.buildGeneratedHtml(preview);" in editor_js
    assert "this.documentEl.innerHTML = this.generatedHtml;" in editor_js
    assert 'data-origin="generated"' in editor_js
    assert 'note.dataset.origin = "user"' in editor_js
    assert "appState.processing.rows = combined.map" in app_js
    assert "productionSheetEditor?.close();" in app_js


def test_direct_edits_update_the_preview_and_support_history():
    html, _app_js, editor_js, _css = _sources()

    assert 'id="productionSheetDocument"' in html
    assert 'contenteditable="true"' in editor_js
    assert 'this.documentEl.addEventListener("input"' in editor_js
    assert "this.schedulePageCount();" in editor_js
    assert "this.undoStack.push" in editor_js
    assert "this.redoStack.push" in editor_js
    assert "document.execCommand(command" in editor_js


def test_table_tools_cover_structural_edits_and_resizing():
    html, _app_js, editor_js, css = _sources()

    for command in (
        "add-row",
        "remove-row",
        "add-column",
        "remove-column",
        "merge-right",
        "split-cell",
    ):
        assert f'data-production-sheet-command="{command}"' in html
    assert "cell.colSpan = Number(cell.colSpan || 1)" in editor_js
    assert "cell.colSpan = Number(cell.colSpan) - 1" in editor_js
    assert "resize:both" in css
    assert 'thead{display:${repeatHeaderRule}}' in editor_js


def test_layout_controls_feed_preview_print_and_pdf():
    html, app_js, editor_js, css = _sources()

    for element_id in (
        "productionSheetPaper",
        "productionSheetOrientation",
        "productionSheetMarginTop",
        "productionSheetMarginBottom",
        "productionSheetMarginLeft",
        "productionSheetMarginRight",
        "productionSheetHeader",
        "productionSheetFooter",
        "productionSheetPageNumbers",
        "productionSheetRepeatHeader",
    ):
        assert f'id="{element_id}"' in html
    assert '@page{size:${paper} ${orientation}' in editor_js
    assert 'counter(page) " of " counter(pages)' in editor_js
    assert "break-before:page" in editor_js
    assert "--ps-page-height" in css
    assert "production-sheet-page-marker" in css
    assert "productionSheetEditor.getLayout()" in app_js
    assert "orientedPaperMm" in app_js


def test_saved_edits_and_processing_source_survive_reload():
    _html, app_js, editor_js, _css = _sources()

    assert 'const STORAGE_KEY = "loe.productionSheetEditor.v1"' in editor_js
    assert "localStorage.setItem(STORAGE_KEY" in editor_js
    assert "safeStorageRecords()[signature]" in editor_js
    assert 'const PROCESSING_WORKSPACE_STORAGE_KEY = "loe.processing.workspace.v1"' in app_js
    assert "persistProcessingWorkspace();" in app_js
    assert "restoreProcessingWorkspace();" in app_js


def test_exports_remain_available_with_edited_pdf_and_source_csv():
    html, app_js, _editor_js, _css = _sources()

    assert 'id="processingExportPdf">Export PDF</button>' in html
    assert 'id="processingExportCsv">Export CSV</button>' in html
    assert "productionSheetEditor.getPlainText()" in app_js
    csv_start = app_js.index("function exportProcessingCsv")
    csv_source = app_js[csv_start : csv_start + 2200]
    assert "appState.processing.rows" in csv_source
    assert 'a.download = "mother-sheet.csv"' in csv_source

