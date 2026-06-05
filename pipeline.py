from __future__ import annotations

import asyncio
import io
import json
import re
from copy import deepcopy
from typing import Any

import pdfplumber


ALLOWED_FINDING_STATUSES = {"abnormal", "borderline"}
EXPLICIT_NON_NORMAL_FLAGS = {"h", "high", "l", "low", "a", "abnormal", "borderline"}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a digital PDF."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_text = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(text for text in page_text if text).strip()


def build_extraction_prompt(report_text: str) -> str:
    return f"""You are a clinical information extraction system.

Extract all findings from the following medical lab report and return ONLY a JSON object.

JSON schema:
{{
  "findings": [
    {{
      "name": "finding name (e.g. HbA1c)",
      "value": "reported value with unit (e.g. 9.2%)",
      "reference_range": "normal range if mentioned",
      "report_flag": "explicit flag shown in the report, e.g. H, L, High, Low, Abnormal, or null",
      "status": "normal | abnormal | borderline | unknown",
      "search_term": "short clinical term for PubMed search, using high/low/elevated/reduced only when supported by the report"
    }}
  ]
}}

Return only the JSON. No explanation. No markdown.

Rules:
- Use the report's stated reference range when deciding status.
- If a numeric value is within the stated reference range, set status to "normal".
- If the report does not provide a usable reference range or explicit abnormal flag, set status to "unknown".
- Do not infer high, low, elevated, or reduced from the numeric value alone.
- For unknown status, use a neutral search_term based on the test name only, with no high/low/elevated/reduced direction.
- If status is "normal", leave search_term as an empty string.

Report:
{report_text}
"""


def build_summary_prompt(findings: dict[str, Any], pubmed_context: dict[str, Any]) -> str:
    findings_json = json.dumps(filter_findings_for_summary(findings), indent=2)
    context_json = json.dumps(pubmed_context, indent=2)
    return f"""You are a medical report interpreter helping a patient understand their lab results.

You have the following non-normal extracted findings. These are the ONLY findings to summarize:
{findings_json}

You have the following reference context from medical literature:
{context_json}

Write a plain-language summary for the patient. For each abnormal or borderline finding:
1. Explain what it measures
2. Explain what the abnormal value means
3. Mention what a doctor might want to investigate further

Use simple language. Avoid jargon. Do not diagnose. Do not recommend treatment.
Do not invent concern for values marked normal or values that are within the stated reference range.
Do not say that a low urine protein/creatinine ratio suggests kidney dysfunction unless the report explicitly flags it as abnormal or it is outside the stated reference range.
End with: "Please share this summary with your doctor."

Format:
- One paragraph per finding
- A final "Overall Summary" paragraph
"""


def parse_json_object(raw_output: str) -> dict[str, Any]:
    """Parse a model response that should contain a single JSON object."""
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object.")
    if not isinstance(parsed.get("findings"), list):
        raise ValueError("Model output must include a findings array.")
    return parsed


def normalize_findings(findings: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic status corrections from numeric reference ranges."""
    normalized = deepcopy(findings)
    for finding in normalized.get("findings", []):
        if not isinstance(finding, dict):
            continue
        finding.setdefault("report_flag", None)
        range_status = status_from_reference_range(
            str(finding.get("value", "")),
            finding.get("reference_range"),
        )
        if range_status:
            finding["status"] = range_status
            if range_status == "normal":
                finding["search_term"] = ""
            continue

        if has_explicit_non_normal_flag(finding.get("report_flag")):
            if finding.get("status") not in ALLOWED_FINDING_STATUSES:
                finding["status"] = "abnormal"
            continue

        if finding.get("status") in ALLOWED_FINDING_STATUSES:
            finding["status"] = "unknown"
            finding["search_term"] = neutral_search_term(finding)
        elif finding.get("status") == "normal":
            finding["search_term"] = ""
    return normalized


def filter_findings_for_summary(findings: dict[str, Any]) -> dict[str, Any]:
    """Return only findings the summary model is allowed to discuss as non-normal."""
    return {
        "findings": [
            finding
            for finding in findings.get("findings", [])
            if isinstance(finding, dict) and finding.get("status") in ALLOWED_FINDING_STATUSES
        ]
    }


def has_findings_for_summary(findings: dict[str, Any]) -> bool:
    return bool(filter_findings_for_summary(findings)["findings"])


def summarize_without_non_normal_findings(findings: dict[str, Any]) -> str:
    """Return a deterministic summary when there is nothing safe to send to the model."""
    unknown_names = [
        str(finding.get("name", "a finding")).strip()
        for finding in findings.get("findings", [])
        if isinstance(finding, dict) and finding.get("status") == "unknown"
    ]
    unknown_names = [name for name in unknown_names if name]

    paragraphs = [
        "No abnormal or borderline findings were identified from the values that included a usable reference range or explicit report flag."
    ]
    if unknown_names:
        if len(unknown_names) == 1:
            unknown_text = unknown_names[0]
        else:
            unknown_text = ", ".join(unknown_names[:-1]) + f", and {unknown_names[-1]}"
        paragraphs.append(
            f"The report also includes {unknown_text}, but it did not provide enough reference-range or flag information for LabPartner to classify it as normal or abnormal."
        )

    paragraphs.append(
        "Overall Summary: Please review the full report with your doctor, especially any items your lab or clinician has flagged."
    )
    paragraphs.append("Please share this summary with your doctor.")
    return "\n\n".join(paragraphs)


def status_from_reference_range(value: str, reference_range: Any) -> str | None:
    """Infer normal/abnormal status for simple numeric lab ranges."""
    numeric_value = first_number(value)
    if numeric_value is None or reference_range in {None, ""}:
        return None

    range_text = str(reference_range)
    bounds = parse_reference_range(range_text)
    if bounds is None:
        return None

    low, high, low_inclusive, high_inclusive = bounds
    below_low = low is not None and (
        numeric_value < low if low_inclusive else numeric_value <= low
    )
    above_high = high is not None and (
        numeric_value > high if high_inclusive else numeric_value >= high
    )
    return "abnormal" if below_low or above_high else "normal"


def has_explicit_non_normal_flag(report_flag: Any) -> bool:
    if report_flag in {None, ""}:
        return False
    return str(report_flag).strip().lower() in EXPLICIT_NON_NORMAL_FLAGS


def neutral_search_term(finding: dict[str, Any]) -> str:
    name = str(finding.get("name", "")).strip()
    if name:
        return name.lower()
    return ""


def parse_reference_range(reference_range: str) -> tuple[float | None, float | None, bool, bool] | None:
    """Parse common lab ranges such as 1-14, <100, <=5.6, >90, and >=90."""
    text = reference_range.replace("–", "-").replace("—", "-").strip()

    between_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if between_match:
        low = float(between_match.group(1))
        high = float(between_match.group(2))
        if low > high:
            low, high = high, low
        return low, high, True, True

    upper_match = re.search(r"(<=|<|less than|under)\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if upper_match:
        operator = upper_match.group(1).lower()
        return None, float(upper_match.group(2)), True, operator in {"<=", "less than", "under"}

    lower_match = re.search(
        r"(>=|>|greater than|over|at least)\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if lower_match:
        operator = lower_match.group(1).lower()
        return float(lower_match.group(2)), None, operator in {">=", "greater than", "over", "at least"}, True

    return None


def first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def anonymize_for_pubmed(findings: dict[str, Any]) -> list[str]:
    """Return only minimal clinical search terms for non-normal findings."""
    terms = []
    for finding in findings.get("findings", []):
        if not isinstance(finding, dict):
            continue
        if finding.get("status") not in ALLOWED_FINDING_STATUSES:
            continue
        term = str(finding.get("search_term", "")).strip()
        if term:
            terms.append(term)
    return terms


def run_async(coro):
    """Run async PubMed code from a synchronous pipeline method."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("run_async cannot be called from an active event loop")


def format_sources(pubmed_context: dict[str, Any]) -> str:
    """Render PubMed source links for the Gradio Markdown panel."""
    lines = []
    for term, context in pubmed_context.items():
        urls = context.get("urls", []) if isinstance(context, dict) else []
        if not urls:
            continue
        links = ", ".join(f"[PMID {url.rstrip('/').split('/')[-1]}]({url})" for url in urls)
        lines.append(f"- **{term}**: {links}")
    return "\n".join(lines) if lines else "No PubMed sources found."


def mock_extract_findings(report_text: str) -> dict[str, Any]:
    """Deterministic extraction for local UI development."""
    text = report_text or ""
    findings = []

    patterns = [
        ("HbA1c", r"\b(?:HbA1c|A1c)\b[^\d]*(\d+(?:\.\d+)?)\s*%?", "%", "4.0-5.6%", 5.7, "HbA1c elevated diabetes"),
        ("Glucose", r"\bGlucose\b[^\d]*(\d+(?:\.\d+)?)\s*(?:mg/dL)?", "mg/dL", "70-99 mg/dL", 100.0, "fasting glucose elevated"),
        ("LDL Cholesterol", r"\bLDL\b[^\d]*(\d+(?:\.\d+)?)\s*(?:mg/dL)?", "mg/dL", "<100 mg/dL", 100.0, "LDL cholesterol elevated"),
        ("eGFR", r"\beGFR\b[^\d]*(\d+(?:\.\d+)?)", "mL/min/1.73m2", ">=90 mL/min/1.73m2", 90.0, "eGFR reduced kidney function"),
    ]

    for name, pattern, unit, reference_range, threshold, search_term in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        is_low_marker = name == "eGFR"
        abnormal = value < threshold if is_low_marker else value >= threshold
        findings.append(
            {
                "name": name,
                "value": f"{value:g} {unit}",
                "reference_range": reference_range,
                "status": "abnormal" if abnormal else "normal",
                "search_term": search_term if abnormal else "",
            }
        )

    if not findings:
        findings.append(
            {
                "name": "Example finding",
                "value": "Not detected by mock parser",
                "reference_range": "",
                "status": "borderline",
                "search_term": "lab result interpretation",
            }
        )

    return {"findings": findings}


def mock_summarize(findings: dict[str, Any], pubmed_context: dict[str, Any]) -> str:
    if not has_findings_for_summary(findings):
        return summarize_without_non_normal_findings(findings)

    paragraphs = []
    for finding in filter_findings_for_summary(findings).get("findings", []):
        name = finding.get("name", "This finding")
        value = finding.get("value", "the reported value")
        paragraphs.append(
            f"{name} was reported as {value}. This result was marked {finding.get('status', 'non-normal')}. "
            "A doctor may want to interpret it alongside your symptoms, medical history, medications, and other lab results."
        )

    if not paragraphs:
        paragraphs.append("No abnormal findings were detected by the current parser.")

    paragraphs.append(
        "Overall Summary: This summary is intended to make the report easier to discuss with a clinician, not to diagnose or treat a condition. "
        "Please share this summary with your doctor."
    )
    return "\n\n".join(paragraphs)


def run_mock_pipeline(pdf_bytes: bytes) -> dict[str, Any]:
    report_text = extract_text_from_pdf(pdf_bytes)
    findings = normalize_findings(mock_extract_findings(report_text))
    search_terms = anonymize_for_pubmed(findings)
    pubmed_context = {
        term: {
            "pmids": [],
            "abstracts": "Mock PubMed context for local development.",
            "urls": [],
        }
        for term in search_terms
    }
    return {
        "findings": findings,
        "summary": mock_summarize(findings, pubmed_context),
        "sources": format_sources(pubmed_context),
    }
