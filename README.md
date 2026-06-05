---
title: LabPartner
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: ui.py
pinned: true
---

# LabPartner

LabPartner is a privacy-first medical report summarizer for the HuggingFace x Modal Hackathon.

The HuggingFace Space hosts the Gradio UI. Model inference and PDF processing run on Modal.

## Local Development

Run the UI with mock model output:

```bash
USE_MOCK=true uv run python ui.py
```

Deploy the Modal app:

```bash
uv run modal deploy modal_app.py
```

Run the UI against the deployed Modal class:

```bash
uv run python ui.py
```

Run tests:

```bash
uv run python -m unittest
```

## Secrets

Set these as HuggingFace Space secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Set this as a Modal secret named `huggingface`:

- `HF_TOKEN`

## Privacy Notes

Uploaded PDFs are processed for the active request and are not intentionally persisted by the app. The raw PDF passes through the HuggingFace Space process before being sent to Modal, but it is not sent to PubMed. Only minimal clinical search terms for non-normal findings are sent to PubMed.
