"""Check the actual label PDF geometry, including the reported wrapped type."""

import base64
import json
from pathlib import Path
import shutil
import subprocess

import fitz
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORTED_TYPE = "3 VETRI 33.1F +14CALDO+ 4F +16CALDO+ 33.1LOWE (44MM)"
NODE_RENDER = r"""
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {
  Uint8Array,
  pageSize: { w: 100 * 72 / 25.4, h: 40 * 72 / 25.4 },
  ensurePdfLib: async () => {},
  preloadLogos: async () => ({
    keliBytes: new Uint8Array(fs.readFileSync('docs/logokeli.png')),
    ceBytes: new Uint8Array(fs.readFileSync('docs/ce.png')),
  }),
};
context.window = context;
vm.createContext(context);
const library = require.resolve('pdf-lib/dist/pdf-lib.min.js', {
  paths: [process.cwd(), path.resolve('backend/workflow_runtime')],
});
vm.runInContext(fs.readFileSync(library, 'utf8'), context);
vm.runInContext(fs.readFileSync('docs/js/platform-workflows.js', 'utf8'), context);
context.generateLabelsPdf(request.rows, request.options).then(bytes => {
  process.stdout.write(Buffer.from(bytes).toString('base64'));
}).catch(error => { console.error(error.message); process.exitCode = 1; });
"""


def render_labels(glass_type, *, fit_description=True, source="processing"):
    if not shutil.which("node"):
        pytest.skip("Node is required to render shared label PDFs")
    row = {
        "order_number": "R-26-0877",
        "position": "3-1",
        "dimension": "455 × 2215",
        "type": glass_type,
        "quantity": 2,
        "source": source,
        "ms_index": 7,
    }
    result = subprocess.run(
        ["node", "-e", NODE_RENDER],
        input=json.dumps({"rows": [row], "options": {"fitDescription": fit_description}}),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if "Cannot find module 'pdf-lib" in result.stderr:
        pytest.skip("Install backend/workflow_runtime dependencies to render label PDFs")
    if result.returncode:
        raise RuntimeError(result.stderr)
    return base64.b64decode(result.stdout)


def text_spans(page):
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]


@pytest.mark.parametrize(
    "glass_type",
    [
        REPORTED_TYPE,
        "4F",
        "3 VETRI 44.2 ACUSTICO +18 CALDO NERO+ 6F TEMPERATO +18 CALDO NERO+ 44.2 LOWE (68MM)",
        "33.1F+14CALDO+4F+16CALDO+33.1LOWE(44MM)" * 2,
    ],
)
def test_labels_description_and_number_have_separate_space(glass_type):
    with fitz.open(stream=render_labels(glass_type), filetype="pdf") as pdf:
        assert len(pdf) == 2
        for page in pdf:
            assert page.rect.width == pytest.approx(100 * 72 / 25.4)
            assert page.rect.height == pytest.approx(40 * 72 / 25.4)
            spans = text_spans(page)
            assert "R-26-0877" in spans[1]["text"]
            assert "Pos: 3-1" in spans[1]["text"]
            assert "455 × 2215" in spans[1]["text"]
            description, number = spans[2:-1], spans[-1]
            assert number["text"] == "7"
            assert number["size"] == pytest.approx(18)
            assert (number["bbox"][0] + number["bbox"][2]) / 2 == pytest.approx(page.rect.width / 2)
            assert number["bbox"][3] <= page.rect.height
            assert "".join("".join(s["text"].split()) for s in description) == "".join(
                f"Glass Type: {glass_type}".split()
            )
            for span in description:
                assert 6 <= span["size"] <= 10
                assert span["bbox"][0] >= 16 - 0.1
                assert span["bbox"][2] <= page.rect.width - 16 + 0.1
                assert span["bbox"][3] <= number["bbox"][1] - 3
            if glass_type == REPORTED_TYPE:
                assert len(description) == 2
                assert all(span["size"] == pytest.approx(10) for span in description)


def test_other_label_entry_points_keep_their_existing_layout():
    with fitz.open(stream=render_labels(REPORTED_TYPE, fit_description=False), filetype="pdf") as pdf:
        spans = text_spans(pdf[0])
        assert spans[-1]["text"] == "7"
        assert spans[-1]["origin"][1] == pytest.approx(90)
        assert spans[2]["size"] == pytest.approx(10)
        assert spans[3]["origin"][1] - spans[2]["origin"][1] == pytest.approx(24)


def test_description_also_clears_the_brand_footer():
    with fitz.open(stream=render_labels(REPORTED_TYPE, source="extract"), filetype="pdf") as pdf:
        spans = text_spans(pdf[0])
        assert spans[-1]["text"] == "KELI ALBANIA PVC"
        assert max(span["bbox"][3] for span in spans[2:-1]) < spans[-1]["bbox"][1] - 3


def test_overlong_description_is_rejected_instead_of_truncated():
    with pytest.raises(RuntimeError, match="Glass type is too long to fit"):
        render_labels("LONG GLASS COMPOSITION " * 50)
