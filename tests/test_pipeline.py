import unittest
from contextlib import redirect_stdout
from io import StringIO

from pipeline import (
    anonymize_for_pubmed,
    build_json_repair_prompt,
    build_summary_prompt,
    clean_summary_output,
    extraction_error_findings,
    extract_text_from_pdf,
    normalize_findings,
    pubmed_error_context,
    run_mock_pipeline,
    summarize_without_non_normal_findings,
    timed_step,
)


def make_text_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")

    xref_offset = sum(len(chunk) for chunk in chunks)
    xref = [b"xref\n0 6\n", b"0000000000 65535 f \n"]
    xref.extend(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    chunks.extend(
        [
            *xref,
            b"trailer\n<< /Size 6 /Root 1 0 R >>\n",
            b"startxref\n",
            str(xref_offset).encode(),
            b"\n%%EOF\n",
        ]
    )
    return b"".join(chunks)


class PipelineTest(unittest.TestCase):
    def test_extract_text_from_synthetic_pdf(self):
        pdf_bytes = make_text_pdf("HbA1c 7.2% LDL 130 mg/dL eGFR 82")

        extracted_text = extract_text_from_pdf(pdf_bytes)

        self.assertIn("HbA1c 7.2%", extracted_text)
        self.assertIn("LDL 130", extracted_text)
        self.assertIn("eGFR 82", extracted_text)

    def test_mock_pipeline_flags_synthetic_lab_values(self):
        pdf_bytes = make_text_pdf("Patient: Example Person HbA1c 7.2% LDL 130 mg/dL eGFR 82")

        result = run_mock_pipeline(pdf_bytes)

        finding_statuses = {
            finding["name"]: finding["status"] for finding in result["findings"]["findings"]
        }
        self.assertEqual(finding_statuses["HbA1c"], "abnormal")
        self.assertEqual(finding_statuses["LDL Cholesterol"], "abnormal")
        self.assertEqual(finding_statuses["eGFR"], "abnormal")
        self.assertIn("Please share this summary with your doctor.", result["summary"])

    def test_reference_range_normalization_clears_in_range_urine_ratio(self):
        extracted = {
            "findings": [
                {
                    "name": "Urine Protein/Creatinine Ratio",
                    "value": "0.04",
                    "reference_range": "<0.20",
                    "report_flag": None,
                    "status": "abnormal",
                    "search_term": "low urine protein creatinine ratio kidney function",
                },
                {
                    "name": "Urine Protein",
                    "value": "1.12 mg/dL",
                    "reference_range": "1-14 mg/dL",
                    "report_flag": None,
                    "status": "normal",
                    "search_term": "",
                },
            ]
        }

        normalized = normalize_findings(extracted)

        ratio = normalized["findings"][0]
        self.assertEqual(ratio["status"], "normal")
        self.assertEqual(ratio["search_term"], "")
        self.assertEqual(anonymize_for_pubmed(normalized), [])

    def test_summary_prompt_excludes_normal_findings(self):
        findings = normalize_findings(
            {
                "findings": [
                    {
                        "name": "Urine Protein/Creatinine Ratio",
                        "value": "0.04",
                        "reference_range": "<0.20",
                        "report_flag": None,
                        "status": "abnormal",
                        "search_term": "low urine protein creatinine ratio kidney function",
                    },
                    {
                        "name": "LDL",
                        "value": "130 mg/dL",
                        "reference_range": "<100 mg/dL",
                        "report_flag": None,
                        "status": "abnormal",
                        "search_term": "LDL elevated cardiovascular risk",
                    },
                ]
            }
        )

        prompt = build_summary_prompt(findings, {})

        self.assertNotIn("Urine Protein/Creatinine Ratio", prompt)
        self.assertIn("LDL", prompt)

    def test_missing_reference_range_does_not_infer_low_or_high(self):
        extracted = {
            "findings": [
                {
                    "name": "URINE PROTEIN/CREATININE RATIO",
                    "value": "0.04",
                    "reference_range": "-",
                    "report_flag": None,
                    "status": "abnormal",
                    "search_term": "urine protein creatinine ratio low",
                }
            ]
        }

        normalized = normalize_findings(extracted)

        finding = normalized["findings"][0]
        self.assertEqual(finding["status"], "unknown")
        self.assertEqual(finding["search_term"], "urine protein/creatinine ratio")
        self.assertEqual(anonymize_for_pubmed(normalized), [])

    def test_explicit_report_flag_preserves_non_normal_without_range(self):
        extracted = {
            "findings": [
                {
                    "name": "LDL",
                    "value": "130 mg/dL",
                    "reference_range": "-",
                    "report_flag": "H",
                    "status": "abnormal",
                    "search_term": "LDL cholesterol high",
                }
            ]
        }

        normalized = normalize_findings(extracted)

        finding = normalized["findings"][0]
        self.assertEqual(finding["status"], "abnormal")
        self.assertEqual(finding["search_term"], "LDL cholesterol high")
        self.assertEqual(anonymize_for_pubmed(normalized), ["LDL cholesterol high"])

    def test_no_non_normal_summary_is_deterministic_and_has_no_patient_hallucination(self):
        findings = normalize_findings(
            {
                "findings": [
                    {
                        "name": "URINE PROTEIN/CREATININE RATIO",
                        "value": "0.04",
                        "reference_range": None,
                        "report_flag": None,
                        "status": "abnormal",
                        "search_term": "urine protein creatinine ratio low",
                    },
                    {
                        "name": "URINE PROTEIN",
                        "value": "1.12 mg/dL",
                        "reference_range": "1-14 mg/dL",
                        "report_flag": None,
                        "status": "normal",
                        "search_term": "urine protein normal",
                    },
                    {
                        "name": "URINE CREATININE",
                        "value": "26 mg/dL",
                        "reference_range": "14.71-294.12 mg/dL",
                        "report_flag": None,
                        "status": "normal",
                        "search_term": "urine creatinine normal",
                    },
                ]
            }
        )

        summary = summarize_without_non_normal_findings(findings)

        self.assertIn("No abnormal or borderline findings", summary)
        self.assertIn("URINE PROTEIN/CREATININE RATIO", summary)
        self.assertIn("Please share this summary with your doctor.", summary)
        self.assertNotIn("John Smith", summary)
        self.assertNotIn("Patient ID", summary)
        self.assertNotIn("Date of Report", summary)

    def test_json_repair_prompt_requests_valid_schema(self):
        prompt = build_json_repair_prompt("{findings: [")

        self.assertIn("Return ONLY valid JSON", prompt)
        self.assertIn('"findings"', prompt)
        self.assertIn("Malformed output:", prompt)

    def test_extraction_error_summary_is_deterministic(self):
        findings = extraction_error_findings("bad json")

        summary = summarize_without_non_normal_findings(findings)

        self.assertIn("could not reliably extract structured findings", summary)
        self.assertIn("Please share this summary with your doctor.", summary)
        self.assertNotIn("John Smith", summary)

    def test_pubmed_error_context_renders_sources_warning(self):
        from pipeline import format_sources

        sources = format_sources(pubmed_error_context("PubMed request failed with HTTP 429."))

        self.assertIn("PubMed sources unavailable", sources)
        self.assertIn("HTTP 429", sources)

    def test_clean_summary_output_removes_leaked_prompt_lines(self):
        raw_summary = (
            '- A final "Please share this summary with your doctor." line\n'
            "- Use bullet points for the explanation of each finding.\n\n"
            "**Patient Summary:**\n\n"
            "**Neutrophils**\n"
            "* This test measures neutrophils.\n"
        )

        cleaned = clean_summary_output(raw_summary)

        self.assertNotIn("A final", cleaned)
        self.assertNotIn("Use bullet points", cleaned)
        self.assertTrue(cleaned.startswith("**Patient Summary:**"))

    def test_timed_step_logs_privacy_safe_metadata(self):
        output = StringIO()

        with redirect_stdout(output):
            with timed_step("unit.test", item_count=2):
                pass

        log_line = output.getvalue()
        self.assertIn("timing step=unit.test status=ok seconds=", log_line)
        self.assertIn("item_count=2", log_line)


if __name__ == "__main__":
    unittest.main()
