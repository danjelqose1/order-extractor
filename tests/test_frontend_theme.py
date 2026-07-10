from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "docs" / "index.html"
STYLES_CSS = ROOT / "docs" / "css" / "styles.css"


def test_frontend_dark_mode_follows_system_preference():
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLES_CSS.read_text(encoding="utf-8")

    assert '<meta name="color-scheme" content="light dark" />' in html
    assert 'media="(prefers-color-scheme: dark)"' in html
    assert "@media (prefers-color-scheme:dark)" in css
    assert "color-scheme:dark;" in css


def test_processing_empty_state_has_a_dark_theme_surface():
    css = STYLES_CSS.read_text(encoding="utf-8")
    dark_theme = css.split("@media (prefers-color-scheme:dark)", 1)[1]

    assert ".processing-order-list," in dark_theme
    assert ".processing-order-list.processing-empty{" in dark_theme


def test_late_component_rules_keep_dark_theme_surfaces():
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLES_CSS.read_text(encoding="utf-8")
    late_dark_theme = css.rsplit("@media (prefers-color-scheme:dark)", 1)[1]

    assert 'class="analysis-question-input"' in html
    assert "background:#fbfcff" not in html.split('id="analysisQuestion"', 1)[1].split("/>", 1)[0]
    assert ".telegram-files-toolbar select," in late_dark_theme
    assert ".analysis-question-input," in late_dark_theme
    assert ".manual-photo-warning," in late_dark_theme
