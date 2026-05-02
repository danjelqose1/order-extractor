from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _html() -> str:
    return FRONTEND.read_text(encoding="utf-8")


def test_pdf_editor_navigation_and_upload_ui_present():
    html = _html()

    assert 'data-tab="pdfeditor"' in html
    assert 'id="tabPdfEditor"' in html
    assert 'id="pdfEditorFileInput"' in html
    assert 'id="pdfEditorDropZone"' in html
    assert "loadPdfIntoEditor(file)" in html


def test_workspace_factory_and_smart_chat_sections_are_separate():
    html = _html()

    assert "Factory Assistant" in html
    assert "Smart Chat" in html
    assert 'id="workspaceChatLog"' in html
    assert 'id="workspaceSmartChatLog"' in html
    assert 'id="workspaceCommandInput"' in html
    assert 'id="workspaceSmartChatInput"' in html
    assert 'API_BASE + "/api/agent/smart-chat"' in html
    assert "runSmartChat(message)" in html
    assert "runWorkspaceCommand(message)" in html
    assert "data-smart-chat-action=\"send-to-factory\"" in html
    assert "const backendMessage = typeof response?.message === \"string\" ? response.message : \"\";" in html
    assert 'appendSmartChatMessage("assistant", response.message || "Done.");' not in html


def test_workspace_combined_labels_aggregate_sections_per_order():
    html = _html()

    assert "const sectionBuckets = new Map();" in html
    assert "const getSectionBucket = (key, defaults = {}) =>" in html
    assert "bucket.lines.push(...lines);" in html
    assert "bucket.sourceGroups.push(group.display || group.raw || `Processing Group ${groupIndex + 1}`);" in html
    assert "existingKeys.has(sectionKey)" in html


def test_telegram_files_reuse_dashboard_label_print_flow():
    html = _html()

    assert 'data-telegram-action="labels"' in html
    assert "No linked order yet" in html
    assert 'aria-label="No linked order yet"' in html
    assert ".telegram-file-actions{display:grid;grid-template-columns:1fr}" in html
    assert "async function printTelegramLinkedLabels(file)" in html
    assert "const order = await fetchOrder(file.linked_order_id);" in html
    assert "await handlePrint(rows);" in html
    assert "generateLabelsPdf" in html


def test_telegram_files_default_to_active_with_touched_filters():
    html = _html()

    assert 'id="telegramFilesTouchedFilter"' in html
    assert '<option value="false" selected>Active</option>' in html
    assert '<option value="all">All</option>' in html
    assert '<option value="true">Touched</option>' in html
    assert 'touched: "false"' in html
    assert 'params.set("touched", telegramFilesState.touched || "false");' in html
    assert 'if (touchedFilter === "true" && !file.touched) return false;' in html
    assert 'if (touchedFilter === "false" && file.touched) return false;' in html
    assert 'telegramFilesState.items = (telegramFilesState.items || []).filter(item => String(item.id) !== String(fileId));' in html
    assert "telegramFilesTouchedFilter.addEventListener" in html


def test_telegram_files_auto_touch_workflow_hooks_present():
    html = _html()

    assert "function telegramHandlingProgressBadges(file)" in html
    assert "Labels printed" in html
    assert "Linked order opened" in html
    assert "async function markTelegramFileHandlingStep(fileId, step)" in html
    assert '"mark-labels-printed"' in html
    assert '"mark-linked-order-opened"' in html
    assert 'await markTelegramFileHandlingStep(file.id, "labels");' in html
    assert 'await markTelegramFileHandlingStep(file.id, "order");' in html
    assert 'data-telegram-action="order" data-id="${escapeHtml(file.id)}"' in html


def test_telegram_files_safe_delete_action_present():
    html = _html()

    assert 'data-telegram-action="delete"' in html
    assert "Delete this Telegram file?" in html
    assert "Original records are preserved." in html
    assert "Also delete linked draft order" in html
    assert "Linked order is not draft and will not be deleted." in html
    assert "async function deleteTelegramFile(file)" in html
    assert 'method: "DELETE"' in html
    assert 'params.set("also_delete_linked_order", "true");' in html
    assert "if (file.deleted) return false;" in html


def test_workspace_merge_across_orders_is_explicit_option():
    html = _html()

    assert "function inferWorkspaceMergeAcrossOrders(message)" in html
    assert "processValidatedOrdersViaExistingModules(orders, mode = \"combined\", options = {})" in html
    assert "const mergeAcrossOrders = mode === \"combined\" && !!options.mergeAcrossOrders;" in html
    assert "appState.processing.options.mergeAcrossOrders = mergeAcrossOrders;" in html
    assert "mergeAcrossOrders: !!frontendAction.mergeAcrossOrders" in html
    assert "Processed together, keeping orders separate." in html
    assert "Processed together with merge across orders enabled." in html
    assert "Merge across orders: ${result.mergeAcrossOrders ? \"Yes\" : \"No\"}" in html


def test_pdf_editor_page_operations_are_isolated_state():
    html = _html()

    assert "function rotateActivePdfPage" in html
    assert "function deleteActivePdfPage" in html
    assert "function movePdfPage" in html
    assert "page.rotation" in html
    assert "page.deleted = true" in html
    assert "pdfEditorState.pages.splice" in html


def test_pdf_editor_text_overlay_and_safe_text_edit_model_present():
    html = _html()

    assert 'data-pdf-tool="text"' in html
    assert 'data-pdf-tool="textEdit"' in html
    assert 'data-pdf-tool="nativeTextEdit"' in html
    assert "Safe Text Edit" in html
    assert "Native Text Edit" in html
    assert 'createPdfOverlay("text"' in html
    assert 'createPdfOverlay("textEdit"' in html
    assert "function selectPdfTextBlock" in html
    assert "pending: true" in html
    assert "originalText" in html
    assert "replacementText" in html
    assert "detectedFromTextLayer" in html
    assert "drawRectangle" in html


def test_pdf_editor_text_layer_clickability_and_coordinate_helpers_present():
    html = _html()

    assert "textItemToViewportRect" in html
    assert "screenToPdfCoords" in html
    assert "pdfToScreenCoords" in html
    assert "viewportRectToNormalized" in html
    assert "text-edit-mode" in html
    assert "pdfEditorState.textBlocks.push(block)" in html
    assert "span.dataset.textBlockId = block.id" in html
    assert "selectPdfTextBlock(block.id)" in html
    assert "selectNativePdfTextBlock(block.id)" in html


def test_pdf_editor_applied_text_edits_render_without_annotation_boxes():
    html = _html()

    assert ".pdf-overlay-textedit.applied" in html
    assert ".pdf-overlay-textedit.applied.selected" in html
    assert ".pdf-overlay-textedit.applied.select-outline" in html
    assert "border-color:transparent" in html
    assert 'pdfEditorState.selectedOverlayId = pdfEditorState.tool === "select" ? overlay.id : null' in html
    assert "pdfEditorState.hoveredTextBlockId = null" in html
    assert "pdfEditorState.activeTextBlockId = null" in html
    assert "overlay.applied = true" in html
    assert "getPdfTextDrawY" in html
    assert "overlay.pageIndex === pageState.sourceIndex && !overlay.pending" in html


def test_pdf_editor_text_style_and_cover_color_detection_present():
    html = _html()

    assert "samplePdfCanvasBackgroundColor" in html
    assert "normalizePdfHexColor" in html
    assert "fontFamily = style.fontFamily" in html
    assert "backgroundColor: samplePdfCanvasBackgroundColor(normalized)" in html
    assert "baselineY: rect.baseline / viewport.height" in html
    assert "pageScale: viewport.scale" in html
    assert "pageRotation: pageState ? pageState.rotation : 0" in html


def test_pdf_editor_text_edit_font_size_does_not_clip_replacement():
    html = _html()

    assert "calculateTextEditVisualBounds" in html
    assert "calculateTextEditPdfBounds" in html
    assert "estimateTextEditWidth" in html
    assert ".text-edit-cover" in html
    assert ".text-edit-replacement-text" in html
    assert ".text-edit-overlay" in html
    assert "pdf-editor-overlay-layer" in html
    assert 'if (overlay.type === "textEdit") el.classList.add("text-edit-overlay")' in html
    assert "overflow:visible" in html
    assert "white-space:nowrap" in html
    assert "cover.className = \"text-edit-cover\"" in html
    assert "textEl.className = \"text-edit-replacement-text\"" in html
    assert "copiedPage.drawRectangle" in html
    assert "copiedPage.drawText" in html


def test_pdf_editor_text_edit_replacement_can_move_independently():
    html = _html()

    assert "originalBounds" in html
    assert "replacementPosition" in html
    assert "replacementBounds" in html
    assert "setTextEditReplacementPosition" in html
    assert "startTextEditReplacementDrag" in html
    assert "handleTextEditReplacementDrag" in html
    assert "endTextEditReplacementDrag" in html
    assert "nudgeSelectedTextEdit" in html
    assert "pdfEditorResetTextPosition" in html
    assert "replacementX" in html
    assert "replacementY" in html


def test_pdf_editor_export_uses_copy_and_preserves_original_bytes():
    html = _html()

    assert "originalPdfBytes: null" in html
    assert "workingPdfBytes: null" in html
    assert "pdfEditorState.originalPdfBytes = new Uint8Array(bytes)" in html
    assert "pdfEditorState.workingPdfBytes = new Uint8Array(bytes)" in html
    assert "const sourceBytes = pdfEditorState.workingPdfBytes || pdfEditorState.originalPdfBytes" in html
    assert "PDFDocument.load(new Uint8Array(sourceBytes))" in html
    assert "PDFDocument.create()" in html
    assert "outputDoc.copyPages" in html
    assert "new Blob([bytes], { type: \"application/pdf\" })" in html
    assert "Original PDF bytes were not changed" in html


def test_pdf_editor_scanned_pdf_message_present():
    html = _html()

    assert "This PDF has no editable text layer. You can still add text boxes manually." in html


def test_pdf_editor_native_text_edit_pipeline_present():
    html = _html()

    assert "function selectNativePdfTextBlock" in html
    assert "function applyNativePdfTextEdit" in html
    assert "function fallbackNativeToSafeTextEdit" in html
    assert "Native edit available" in html
    assert "Trying native replacement" in html
    assert "Native edit applied" in html
    assert "Native edit failed. Use Safe Edit instead." in html
    assert "/api/pdf-editor/native-text-replace" in html
    assert "Apply Native Edit" in html
    assert "Use Safe Edit Instead" in html
    assert "Preserve background / lines" in html
    assert "preserveNearbyLines" in html
    assert "nativeEditMode" in html
    assert "content_stream_first" in html
    assert "Native edit used cleanup fallback and may affect background details." in html
    assert "reloadPdfEditorWorkingPdf" in html
    assert "editedPdfBase64" in html
