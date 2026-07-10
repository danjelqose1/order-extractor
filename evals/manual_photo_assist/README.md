# Manual Photo Assist Evaluation

Private evaluation photos and gold labels live in:

```text
data/private-evals/manual-photo-assist/
```

That directory is intentionally ignored by git because order photos can contain client information.

## Run

Run the development set against the deployed backend:

```bash
python3 scripts/evaluate_manual_photo_assist.py --run --split development
```

Score the saved predictions again without API calls:

```bash
python3 scripts/evaluate_manual_photo_assist.py --split development
```

Use the holdout only after a prompt or reconciliation change is finalized:

```bash
python3 scripts/evaluate_manual_photo_assist.py --run --split holdout
```

## Gold Review

Each case in the private `manifest.json` has a `gold_status`:

- `verified`: safe to use for all listed fields.
- `needs_user_confirmation`: confident fields are scored, while entries listed in `uncertain_fields` or `unscored_document_fields` are excluded.

Do not tune prompts against holdout predictions. Add difficult failures to the development set only after the holdout report is recorded.

## Primary Metrics

- Exact row count
- Exact order accuracy
- Client position and red-index accuracy
- Width, height, and quantity accuracy
- Material-group, section, client, and note accuracy
- Model fallback rate, latency, and manual corrections required
