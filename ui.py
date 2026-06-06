from __future__ import annotations

import contextlib
import os

import gradio as gr


USE_MOCK = os.getenv("USE_MOCK", "").lower() in {"1", "true", "yes"}
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "labpartner")
MODAL_CLASS_NAME = os.getenv("MODAL_CLASS_NAME", "LabPartner")
MODAL_EXTRACT_FUNCTION_NAME = os.getenv("MODAL_EXTRACT_FUNCTION_NAME", "extract_pdf_findings")
SUMMARY_PENDING_TEXT = "Extracted findings. Generating summary..."


def analyze(pdf_file):
    if pdf_file is None:
        yield {}, "Please upload a PDF report.", ""
        return

    try:
        with open(pdf_file.name, "rb") as file:
            pdf_bytes = file.read()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(pdf_file.name)

    if USE_MOCK:
        from pipeline import extract_text_from_pdf, mock_extract_findings, mock_summarize, normalize_findings

        report_text = extract_text_from_pdf(pdf_bytes)
        findings = normalize_findings(mock_extract_findings(report_text))
        sources = "No external sources used."
        yield findings, SUMMARY_PENDING_TEXT, sources
        yield findings, mock_summarize(findings, {}), sources
        return

    import modal
    from pipeline import has_findings_for_summary, summarize_without_non_normal_findings

    extract_function = modal.Function.from_name(MODAL_APP_NAME, MODAL_EXTRACT_FUNCTION_NAME)
    extraction = extract_function.remote(pdf_bytes)
    findings = extraction["findings"]
    sources = extraction["sources"]
    yield findings, SUMMARY_PENDING_TEXT, sources

    if not findings.get("findings"):
        labpartner = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        result = labpartner().run_pipeline.remote(pdf_bytes)
        yield result["findings"], result["summary"], result["sources"]
        return

    if has_findings_for_summary(findings):
        labpartner = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        summary = labpartner().summarize.remote(findings, {})
    else:
        summary = summarize_without_non_normal_findings(findings)

    yield findings, summary, sources


with gr.Blocks() as demo:
    gr.Markdown("# LabPartner")
    gr.Markdown(
        "Upload your lab report and get a plain-language summary. "
        "**Reports are processed only for the active request. Only minimal clinical search terms are looked up online.**"
    )

    file_input = gr.File(label="Upload Lab Report (PDF)", file_types=[".pdf"])
    analyze_button = gr.Button("Analyze", variant="primary")

    with gr.Row():
        findings_panel = gr.JSON(label="Extracted Findings")
        summary_panel = gr.Textbox(label="Plain-Language Summary", lines=15)

    sources_panel = gr.Markdown(label="Sources")

    analyze_button.click(
        fn=analyze,
        inputs=file_input,
        outputs=[findings_panel, summary_panel, sources_panel],
    )

    gr.Markdown("_LabPartner is not a medical device. Always consult your doctor._")


if __name__ == "__main__":
    demo.launch()
