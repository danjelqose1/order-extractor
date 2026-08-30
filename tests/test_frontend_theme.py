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


def test_diagnosis_panel_uses_theme_aware_surfaces():
    css = STYLES_CSS.read_text(encoding="utf-8")
    panel_rule = css.split(".diagnosis-panel{", 1)[1].split("}", 1)[0]
    evidence_rule = css.split(".diagnosis-evidence{", 1)[1].split("}", 1)[0]

    assert "background:var(--card-bg)" in panel_rule
    assert "background:#fff" not in panel_rule
    assert "background:var(--bg)" in evidence_rule
    assert "background:#fafafa" not in evidence_rule


def test_late_component_rules_keep_dark_theme_surfaces():
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLES_CSS.read_text(encoding="utf-8")
    late_dark_theme = css.rsplit("@media (prefers-color-scheme:dark)", 1)[1]

    assert 'class="analysis-question-input"' in html
    assert "background:#fbfcff" not in html.split('id="analysisQuestion"', 1)[1].split("/>", 1)[0]
    assert ".telegram-files-toolbar select," in late_dark_theme
    assert ".analysis-question-input," in late_dark_theme
    assert ".manual-photo-warning," in late_dark_theme


def test_beta_operator_uses_scoped_responsive_and_dark_theme_styles():
    css = STYLES_CSS.read_text(encoding="utf-8")

    assert "/* Beta operator: isolated Shadow Mode workspace */" in css
    assert ".beta-main-grid{" in css
    assert ".beta-journal-entry{" in css
    assert ".beta-plan-step.risk-high" in css
    assert "@media (max-width:760px)" in css
    beta_styles = css.split("/* Beta operator: isolated Shadow Mode workspace */", 1)[1]
    assert "@media (prefers-color-scheme:dark)" in beta_styles
    assert ".beta-safety-banner," in beta_styles
