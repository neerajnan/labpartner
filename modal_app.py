from __future__ import annotations

import modal


MODEL_ID = "google/medgemma-27b-text-it"

app = modal.App("labpartner")

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
        )

        prompt = build_extraction_prompt(report_text)
        raw_output = self._generate(prompt, max_new_tokens=3000)
        try:
            return normalize_findings(parse_json_object(raw_output))
        except (JSONDecodeError, ValueError):
            repair_prompt = build_json_repair_prompt(raw_output)
            repaired_output = self._generate(repair_prompt, max_new_tokens=3000)
            try:
                return normalize_findings(parse_json_object(repaired_output))
            except (JSONDecodeError, ValueError):
                return extraction_error_findings(
                    "The model returned malformed extraction JSON after a repair attempt."
                )

    def _summarize_impl(self, findings: dict, pubmed_context: dict) -> str:
        from pipeline import build_summary_prompt

        prompt = build_summary_prompt(findings, pubmed_context)
        return self._generate(prompt, max_new_tokens=1400)

    @modal.method()
    def extract_findings(self, report_text: str) -> dict:
        return self._extract_findings_impl(report_text)

    @modal.method()
    def summarize(self, findings: dict, pubmed_context: dict) -> str:
        return self._summarize_impl(findings, pubmed_context)

    @modal.method()
    def run_pipeline(self, pdf_bytes: bytes) -> dict:
        from pipeline import (
            anonymize_for_pubmed,
            extract_text_from_pdf,
            format_sources,
            has_findings_for_summary,
            run_async,
            summarize_without_non_normal_findings,
        )
        from pubmed import get_context_for_findings

        report_text = extract_text_from_pdf(pdf_bytes)
        findings = self._extract_findings_impl(report_text)
        search_terms = anonymize_for_pubmed(findings)
        pubmed_context = run_async(get_context_for_findings(search_terms))
        summary = (
            self._summarize_impl(findings, pubmed_context)
            if has_findings_for_summary(findings)
            else summarize_without_non_normal_findings(findings)
        )

        return {
            "findings": findings,
            "summary": summary,
            "sources": format_sources(pubmed_context),
        }
