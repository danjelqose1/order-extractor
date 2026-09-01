"""Private stdin worker. Reuses the existing manual print generators."""
import base64
import json
import sys
from manual_documents import build_manual_processing_pdf, build_manual_labels_pdf


def main():
    payload = json.load(sys.stdin)
    build = build_manual_processing_pdf if payload["kind"] == "processing_pdf" else build_manual_labels_pdf
    sys.stdout.write(base64.b64encode(build(payload["order"], payload["settings"])).decode())


if __name__ == "__main__":
    main()
