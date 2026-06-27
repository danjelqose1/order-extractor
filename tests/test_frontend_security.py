from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "docs" / "js" / "app.js"


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_frontend_does_not_read_or_send_openai_api_keys():
    source = _source()

    assert "api.openai.com" not in source
    assert 'localStorage.getItem("loe.openaiKey")' not in source
    assert 'localStorage.getItem("openai.apiKey")' not in source
    assert 'localStorage.removeItem("loe.openaiKey")' in source
    assert 'localStorage.removeItem("openai.apiKey")' in source
    assert "Authorization: `Bearer ${apiKey}`" not in source
    assert 'API_BASE + "/api/invoices/ai/glass-match"' in source
    assert 'API_BASE + "/api/invoices/ai/analyze-line"' in source


def test_extracted_order_group_label_is_html_escaped():
    source = _source()

    assert '<div style="font-weight:600">${escapeHtml(orderLabel)}</div>' in source
    assert '<div style="font-weight:600">${orderLabel}</div>' not in source


def test_status_messages_render_text_and_build_spinner_as_dom():
    source = _source()
    start = source.index("function setStatusMessage")
    end = source.index("function setScanStatus", start)
    implementation = source[start:end]

    assert "statusEl.textContent" in implementation
    assert 'document.createElement("span")' in implementation
    assert "statusEl.innerHTML" not in implementation
