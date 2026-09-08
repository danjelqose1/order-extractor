# Cutting dimensions and recovered values

The Dimension field means the maximum overall horizontal width × maximum overall
vertical height of the rectangular blank cut before shaping, in millimetres and in
the drawing's orientation. Read explicit overall dimensions from the correct
position's drawing; do not select the two largest numbers, use a shorter side or
sloping edge, add allowances, measure pixels, or infer missing dimensions from area.

PDF extraction and targeted page repair now carry this rule. The PDF prompt also
asks for a final check of every position across page breaks and repeated headers.
Targeted dimension repair rejects competing pairs or an explicit NO_VALUE answer.

During extraction, the automatic recovery pass fills missing dimensions that pass
the existing 0.8 repair threshold and dimension-format checks. The completed rows
are revalidated and saved in the new draft before extraction returns. The value is
marked **Recovered · review** and remains in the review issues. This happens without
pressing Fix. New OCR rechecks also fill the editable review field.
Existing entered dimensions and approved/production/completed orders are preserved.

For older drafts that already contain stored repair suggestions, opening an editable
draft/reviewed order prefills only its empty working field; those edits are saved
through the existing explicit save/approval action. Opening an order does not write
or approve anything. The original model reading, repair history and PDF remain
available for audit in both flows. Confidence is a screening heuristic, not
proof that a value was read correctly; the operator still compares it with the PDF.

## Terra trial (8 September 2026)

The extraction default is **gpt-5.6-terra**. **gpt-5.6-sol** remains a candidate for
difficult-page review. Both support image input and structured output. Terra is the documented
intelligence/cost balance; Sol is the flagship for complex professional work.
This is a candidate selection, not a measured accuracy improvement on factory PDFs.

Set `EXTRACTION_MODEL=gpt-5.6-terra` in the deployment environment for the trial;
an existing environment override takes precedence over the code default. Change
that setting to select another model later (previous setting: `gpt-5.4-nano`).
GPT-5.6 extraction explicitly uses reasoning `none`, preserving the previous Nano
baseline. PDF/image requests keep the Responses API; pasted-text and OCR-text
extraction keep Chat Completions and their strict JSON contract. The dedicated
`OCR_MODEL` fallback and unrelated AI features remain unchanged.

Compare models on operator-verified orders including
shapes, repeated sizes, small dimensions, page breaks and scans. Score exact ordered
dimension pairs, row/position coverage, all required fields, invented values,
latency and cost. Retain manual review and deterministic checks with every model.

Sources:
- https://developers.openai.com/api/docs/models/gpt-5.6-terra
- https://developers.openai.com/api/docs/models/gpt-5.6-sol
- https://developers.openai.com/api/docs/guides/images-vision#limitations

Local regression tests exercise the recovered R-26-0781 values and ambiguous OCR
responses using fixtures, without paid API calls. They verify the pipeline and UI
behavior; live model reading quality still needs the comparison described above.
