from __future__ import annotations

import asyncio
import io
import json
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import pdfplumber


ALLOWED_FINDING_STATUSES = {"abnormal", "borderline"}
EXPLICIT_NON_NORMAL_FLAGS = {"h", "high", "l", "low", "a", "abnormal", "borderline"}
SKIP_LINE_PREFIXES = (
    "patient",
    "name",
    "age",
    "sex",
    "gender",
    "date",
    "doctor",
    "hospital",
    "mrn",
    "printed",
    "printed by",
    "printed on",
    "sample",
    "specimen",
    "report",
    "page",
    "pin",
    "pin no",
)
METADATA_MARKERS = (
    "mrn",
    "patient id",
    "patient name",
    "pin no",
    "printed by",
    "printed on",
    "page ",
    "age :",
    " age ",
    " years",
    " months",
)
METHOD_OR_NARRATIVE_MARKERS = (
    "flowcytometry",
    "flow cytometry",
    "fluorescent",
    "fluroscent",
    "fluro",
    "hydrodynamic",
    "electrical",
    "impedance",
    "focusing",
    "performed on",
    "complete blood count is performed",
)
TRAILING_INTERPRETATION_MARKERS = (
    "diagnosis of diabetes",
    "diagnostic of diabetes",
)
NUMBER_PATTERN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
DOCTOR_SHARE_SENTENCE = "Please share this summary with your doctor."


@contextmanager
def timed_step(step: str, **metadata: Any):
    """Log privacy-safe step timing metadata."""
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        metadata_text = " ".join(f"{key}={value}" for key, value in metadata.items())
        suffix = f" {metadata_text}" if metadata_text else ""
        print(f"timing step={step} status={status} seconds={elapsed:.3f}{suffix}", flush=True)


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
    return f"""You are a medical report interpreter helping a patient understand their lab results.

You have the following non-normal extracted findings. These are the ONLY findings to summarize:
{findings_json}

Return only the final patient-facing summary. Do not include examples, drafts, reasoning, analysis, placeholders, alternate versions, or markdown separators.

Write a plain-language summary for the patient. For each abnormal or borderline finding, write exactly one compact bullet that includes:
1. What the test measures
2. The patient's value and the report's reference range
3. Whether the value is high, low, abnormal, or borderline based on that reference range

Keep each finding bullet under 30 words.

Use simple language. Avoid jargon. Do not diagnose. Do not recommend treatment.
Do not use bracketed placeholder text like "[what it means]".
Do not repeat "your doctor may want to investigate" in each bullet; put follow-up guidance only in the Overall Summary.
Do not invent concern for values marked normal or values that are within the stated reference range.
Do not say that a low urine protein/creatinine ratio suggests kidney dysfunction unless the report explicitly flags it as abnormal or it is outside the stated reference range.
The final line must be exactly: Please share this summary with your doctor.
Do not repeat these instructions, the requested format, or any checklist text in your answer.

Format:
- Use one bullet point for each finding
- A final "Overall Summary" paragraph
"""


def build_json_repair_prompt(raw_output: str) -> str:
    return f"""Convert the following malformed model output into one valid JSON object.

Return ONLY valid JSON. No markdown. No explanation.

Required schema:
{{
  "findings": [
    {{
      "name": "finding name",
      "value": "reported value with unit",
      "reference_range": "normal range if mentioned, otherwise null",
      "report_flag": "explicit flag shown in the report, otherwise null",
      "status": "normal | abnormal | borderline | unknown",
      "search_term": "short clinical term for PubMed search"
    }}
  ]
}}

Malformed output:
{raw_output}
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


def extraction_error_findings(message: str) -> dict[str, Any]:
    return {
        "findings": [],
        "extraction_error": message,
    }


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


def extract_findings_from_report_text(report_text: str) -> dict[str, Any]:
    """Extract lab-like rows from report text without model inference."""
    findings = []
    seen = set()
    for raw_line in report_text.splitlines():
        finding = parse_lab_line(raw_line)
        if not finding:
            continue
        key = (finding["name"].lower(), finding["value"].lower(), str(finding.get("reference_range")))
        if key in seen:
            continue
        seen.add(key)
        findings.append(finding)
    return normalize_findings({"findings": findings})


def parse_lab_line(raw_line: str) -> dict[str, Any] | None:
    line = preprocess_lab_line(raw_line)
    if not should_parse_lab_line(line):
        return None

    reference_range, range_start, range_end = find_reference_range(line)
    value_region = line[:range_start] if reference_range else line
    value_region = strip_value_region_annotations(value_region)
    if looks_like_split_test_name(value_region):
        return None
    report_flag = extract_report_flag(line)
    value_match = lab_value_number_match(value_region)
    if not value_match:
        return None

    name = clean_finding_name(value_region[: value_match.start()])
    if not is_plausible_finding_name(name):
        return None

    unit = clean_unit_text(value_region[value_match.end() :])
    value = value_match.group(0)
    if unit:
        value = f"{value} {unit}"

    if reference_range:
        tail_after_range = line[range_end:].strip()
        if not report_flag:
            report_flag = extract_report_flag(tail_after_range)

    initial_status = "abnormal" if has_explicit_non_normal_flag(report_flag) else "unknown"
    search_term = neutral_search_term({"name": name}) if initial_status != "normal" else ""
    return {
        "name": name,
        "value": value,
        "reference_range": reference_range,
        "report_flag": report_flag,
        "status": initial_status,
        "search_term": search_term,
    }


def should_parse_lab_line(line: str) -> bool:
    if not line or not re.search(r"[A-Za-z]", line) or not re.search(NUMBER_PATTERN, line):
        return False
    lowered = line.lower().strip()
    if lowered.startswith(SKIP_LINE_PREFIXES):
        return False
    if looks_like_method_or_narrative(lowered):
        return False
    return not looks_like_report_metadata(lowered)


def find_reference_range(line: str) -> tuple[str | None, int, int]:
    patterns = [
        rf"(?P<range>{NUMBER_PATTERN}\s*(?:-|–|—|to)\s*{NUMBER_PATTERN}(?:\s*[A-Za-zµμ/%][A-Za-z0-9µμ/%.^-]*)?)",
        rf"(?P<range>(?:<=|>=|<|>|≤|≥)\s*{NUMBER_PATTERN}(?:\s*[A-Za-zµμ/%][A-Za-z0-9µμ/%.^-]*)?)",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, line, flags=re.IGNORECASE))
    if not matches:
        return None, len(line), len(line)
    match = max(matches, key=lambda item: item.start())
    return match.group("range").strip(), match.start(), match.end()


def extract_report_flag(text: str) -> str | None:
    flag_pattern = r"(?<![/A-Za-zµμ])(?:H|L|HIGH|LOW|ABNORMAL|BORDERLINE)(?![A-Za-zµμ])"
    for token in re.findall(flag_pattern, text, flags=re.IGNORECASE):
        return token
    return None


def last_number_match(text: str) -> re.Match[str] | None:
    matches = list(re.finditer(NUMBER_PATTERN, text))
    return matches[-1] if matches else None


def lab_value_number_match(text: str) -> re.Match[str] | None:
    matches = list(re.finditer(NUMBER_PATTERN, text))
    if not matches:
        return None
    for match in reversed(matches):
        tail = text[match.end() :].strip()
        if re.match(r"^(?:x\s*)?10\s*\^\s*\d+\b", tail, flags=re.IGNORECASE):
            return match
    return matches[-1]


def clean_finding_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" :-|")


def clean_unit_text(text: str) -> str:
    text = re.sub(r"\b(?:H|L|HIGH|LOW|ABNORMAL|BORDERLINE)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" :-|")
    if not text:
        return ""
    scientific_unit_match = re.fullmatch(
        r"((?:x\s*)?10\s*\^\s*\d+\s*/?\s*[A-Za-zµμ/]+)",
        text,
        flags=re.IGNORECASE,
    )
    if scientific_unit_match:
        unit = re.sub(r"\s+", " ", scientific_unit_match.group(1)).strip()
        return re.sub(r"\^\s+", "^", unit)
    tokens = text.split()
    if len(tokens) > 3:
        return ""
    if any(re.search(r"\d", token) for token in tokens):
        return ""
    if not re.search(r"[A-Za-zµμ/%]", text):
        return ""
    return " ".join(tokens)


def is_plausible_finding_name(name: str) -> bool:
    if len(name) < 2 or len(name) > 80:
        return False
    lowered = name.lower()
    if lowered.startswith("("):
        return False
    if lowered.startswith(SKIP_LINE_PREFIXES):
        return False
    if ":" in name or looks_like_report_metadata(lowered) or looks_like_method_or_narrative(lowered):
        return False
    return bool(re.search(r"[A-Za-z]", name))


def looks_like_report_metadata(text: str) -> bool:
    if any(marker in text for marker in METADATA_MARKERS):
        return True
    if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", text):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
        return True
    if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", text):
        return True
    return False


def preprocess_lab_line(raw_line: str) -> str:
    line = re.sub(r"\(cid:?\s*\d+\s*\)?", " ", raw_line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip()
    lowered = line.lower()
    for marker in TRAILING_INTERPRETATION_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            line = line[:index].strip()
            break
    return re.sub(r"\s+", " ", line).strip(" :-|")


def strip_value_region_annotations(value_region: str) -> str:
    return re.sub(r"\b(?:male|female)\b.*$", "", value_region, flags=re.IGNORECASE).strip()


def looks_like_split_test_name(value_region: str) -> bool:
    return bool(re.fullmatch(r"\s*hba\s*1\s*c\s*", value_region, flags=re.IGNORECASE))


def looks_like_method_or_narrative(text: str) -> bool:
    return any(marker in text for marker in METHOD_OR_NARRATIVE_MARKERS)


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


def summary_token_budget(findings: dict[str, Any]) -> int:
    """Scale summary generation budget to non-normal finding count."""
    finding_count = len(filter_findings_for_summary(findings)["findings"])
    return min(900, max(400, 220 + finding_count * 45))


def summarize_without_non_normal_findings(findings: dict[str, Any]) -> str:
    """Return a deterministic summary when there is nothing safe to send to the model."""
    if findings.get("extraction_error"):
        return (
            "LabPartner could not reliably extract structured findings from this PDF. "
            "Please try a clearer digital PDF or review the report directly with your doctor.\n\n"
            "Please share this summary with your doctor."
        )

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


def clean_summary_output(summary: str) -> str:
    """Remove leaked prompt/checklist/draft fragments from a model summary."""
    text = summary.replace("\r\n", "\n").strip()
    if not text:
        return ""

    response_markers = list(re.finditer(r"\*\*Your Response:\*\*|Your Response:", text, flags=re.IGNORECASE))
    if response_markers:
        text = text[response_markers[-1].end() :].lstrip()

    lab_headings = list(re.finditer(r"\*\*Lab Results Summary\*\*", text, flags=re.IGNORECASE))
    if lab_headings:
        text = text[lab_headings[-1].start() :].lstrip()

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"---", "***", "___"}:
            continue
        if is_leaked_instruction_line(stripped):
            continue
        lines.append(line)

    cleaned = "\n".join(lines).lstrip()
    sentence_index = cleaned.find(DOCTOR_SHARE_SENTENCE)
    if sentence_index != -1:
        cleaned = cleaned[: sentence_index + len(DOCTOR_SHARE_SENTENCE)]
    while cleaned.startswith("\n"):
        cleaned = cleaned[1:]
    cleaned = cleaned.strip()
    if not cleaned:
        return DOCTOR_SHARE_SENTENCE
    if DOCTOR_SHARE_SENTENCE not in cleaned:
        cleaned = f"{cleaned}\n\n{DOCTOR_SHARE_SENTENCE}"
    return cleaned


def is_leaked_instruction_line(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return False
    normalized = normalized.lstrip("-*• ").strip()
    if "please share this summary with your doctor" in normalized and (
        normalized.startswith("end with") or normalized.startswith("the final line")
    ):
        return True
    leaked_prefixes = (
        "a final ",
        'a final "',
        "use bullet points",
        "one paragraph per finding",
        "format:",
        "end with:",
        "end with ",
        "the final line",
        "here is the example format",
        "example format",
        "okay,",
        "ok,",
        "i need",
        "i'll",
        "i will",
        "now i'll",
        "now i will",
        "let's",
        "your response:",
    )
    return normalized.startswith(leaked_prefixes)


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
        rf"({NUMBER_PATTERN})\s*(?:-|to)\s*({NUMBER_PATTERN})",
        text,
        flags=re.IGNORECASE,
    )
    if between_match:
        low = parse_numeric_token(between_match.group(1))
        high = parse_numeric_token(between_match.group(2))
        if low > high:
            low, high = high, low
        return low, high, True, True

    upper_match = re.search(rf"(<=|<|≤|less than|under)\s*({NUMBER_PATTERN})", text, flags=re.IGNORECASE)
    if upper_match:
        operator = upper_match.group(1).lower()
        return None, parse_numeric_token(upper_match.group(2)), True, operator in {"<=", "≤", "less than", "under"}

    lower_match = re.search(
        rf"(>=|>|≥|greater than|over|at least)\s*({NUMBER_PATTERN})",
        text,
        flags=re.IGNORECASE,
    )
    if lower_match:
        operator = lower_match.group(1).lower()
        return parse_numeric_token(lower_match.group(2)), None, operator in {">=", "≥", "greater than", "over", "at least"}, True

    return None


def first_number(text: str) -> float | None:
    match = re.search(NUMBER_PATTERN, text)
    return parse_numeric_token(match.group(0)) if match else None


def parse_numeric_token(token: str) -> float:
    return float(token.replace(",", ""))


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
    if pubmed_context.get("_error"):
        return f"PubMed sources unavailable: {pubmed_context['_error']}"

    lines = []
    for term, context in pubmed_context.items():
        urls = context.get("urls", []) if isinstance(context, dict) else []
        if not urls:
            continue
        links = ", ".join(f"[PMID {url.rstrip('/').split('/')[-1]}]({url})" for url in urls)
        lines.append(f"- **{term}**: {links}")
    return "\n".join(lines) if lines else "No PubMed sources found."


def pubmed_error_context(message: str) -> dict[str, Any]:
    return {"_error": message}


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
