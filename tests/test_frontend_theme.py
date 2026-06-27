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
