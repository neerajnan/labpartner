from __future__ import annotations

import modal


MODEL_ID = "google/medgemma-27b-text-it"

app = modal.App("labpartner")

parser_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("pdfplumber>=0.11.0")
    .add_local_python_source("pipeline")
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "accelerate>=1.8.0",
        "httpx>=0.28.0",
        "pdfplumber>=0.11.0",
        "torch",
        "transformers>=4.50.0",
    )
    .add_local_python_source("pipeline", "pubmed")
)


def extract_pdf_findings_impl(pdf_bytes: bytes) -> dict:
    from pipeline import (
        extract_findings_from_report_text,
        extract_text_from_pdf,
        timed_step,
    )

    with timed_step("modal.extract_pipeline.total", pdf_bytes=len(pdf_bytes)):
        with timed_step("modal.pdf.extract", pdf_bytes=len(pdf_bytes)):
            report_text = extract_text_from_pdf(pdf_bytes)
        with timed_step("modal.parser.extract", report_chars=len(report_text)):
            findings = extract_findings_from_report_text(report_text)
        finding_count = len(findings.get("findings", []))
        print(
            "timing step=modal.extract.result "
            f"status=ok finding_count={finding_count}",
            flush=True,
        )
    return {
        "findings": findings,
        "sources": "No external sources used.",
        "report_text": report_text,
    }


@app.function(image=parser_image, timeout=120)
def extract_pdf_findings(pdf_bytes: bytes) -> dict:
    result = extract_pdf_findings_impl(pdf_bytes)
    result.pop("report_text", None)
    return result


@app.cls(
    gpu="A100-80GB",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
    scaledown_window=300,
)
class LabPartner:
    @modal.enter()
    def load_model(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pipeline import timed_step

        with timed_step("modal.load_model"):
            token = os.environ["HF_TOKEN"]
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=token,
                device_map="auto",
                dtype=torch.bfloat16,
            )

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _extract_findings_impl(self, report_text: str) -> dict:
        from json import JSONDecodeError

        from pipeline import (
            build_extraction_prompt,
            build_json_repair_prompt,
            extraction_error_findings,
            normalize_findings,
            parse_json_object,
            timed_step,
        )

        prompt = build_extraction_prompt(report_text)
        with timed_step("modal.extract.generate", report_chars=len(report_text)):
            raw_output = self._generate(prompt, max_new_tokens=3000)
        try:
            with timed_step("modal.extract.parse", output_chars=len(raw_output)):
                return normalize_findings(parse_json_object(raw_output))
        except (JSONDecodeError, ValueError):
            repair_prompt = build_json_repair_prompt(raw_output)
            with timed_step("modal.extract.repair_generate", output_chars=len(raw_output)):
                repaired_output = self._generate(repair_prompt, max_new_tokens=3000)
            try:
                with timed_step("modal.extract.repair_parse", output_chars=len(repaired_output)):
                    return normalize_findings(parse_json_object(repaired_output))
            except (JSONDecodeError, ValueError):
                return extraction_error_findings(
                    "The model returned malformed extraction JSON after a repair attempt."
                )

    def _summarize_impl(self, findings: dict, pubmed_context: dict) -> str:
        from pipeline import build_summary_prompt, clean_summary_output, summary_token_budget, timed_step

        prompt = build_summary_prompt(findings, pubmed_context)
        max_new_tokens = summary_token_budget(findings)
        with timed_step(
            "modal.summary.generate",
            prompt_chars=len(prompt),
            max_new_tokens=max_new_tokens,
        ):
            raw_summary = self._generate(prompt, max_new_tokens=max_new_tokens)
        with timed_step("modal.summary.clean", output_chars=len(raw_summary)):
            return clean_summary_output(raw_summary)

    @modal.method()
    def extract_findings(self, report_text: str) -> dict:
        return self._extract_findings_impl(report_text)

    @modal.method()
    def summarize(self, findings: dict, pubmed_context: dict) -> str:
        return self._summarize_impl(findings, pubmed_context)

    @modal.method()
    def run_pipeline(self, pdf_bytes: bytes) -> dict:
        from pipeline import (
            has_findings_for_summary,
            summarize_without_non_normal_findings,
            timed_step,
        )

        with timed_step("modal.pipeline.total", pdf_bytes=len(pdf_bytes)):
            extraction = extract_pdf_findings_impl(pdf_bytes)
            findings = extraction["findings"]
            if not findings.get("findings"):
                with timed_step("modal.parser.fallback_model"):
                    findings = self._extract_findings_impl(extraction["report_text"])
            if has_findings_for_summary(findings):
                summary = self._summarize_impl(findings, {})
            else:
                with timed_step("modal.summary.deterministic"):
                    summary = summarize_without_non_normal_findings(findings)

        return {
            "findings": findings,
            "summary": summary,
            "sources": "No external sources used.",
        }
