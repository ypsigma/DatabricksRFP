"""
rfp_lib — shared ingestion + chat logic for the RFP Assistant.

Pure Python, no Spark dependency. Every function that touches the database takes
an `execute` callable so the SAME code runs in two environments:

    execute(sql: str) -> list[dict]

  • Notebook / Job:  execute = lambda q: [r.asDict() for r in spark.sql(q).collect()]
  • Streamlit App:   execute = _run_sql   (databricks-sql-connector)

The AI functions (ai_parse_document, ai_query) and the MERGE all run on the SQL
warehouse, so the identical SQL works through either executor.
"""
from __future__ import annotations

import json
import re
from typing import Callable

Execute = Callable[[str], list]

# ── Field set (single source of truth) ────────────────────────────────────────
EXTRACTION_FIELDS = [
    "project_name", "issuing_org",
    "submission_deadline", "qa_deadline", "project_start_end", "milestones",
    "budget", "pricing_model", "payment_terms", "penalties",
    "technical_score_weighting", "price_weighting", "team_experience_weighting",
    "functional_requirements", "technical_requirements", "compliance_requirements",
    "required_certifications", "minimum_experience", "financial_thresholds",
    "liability", "governing_law", "contract_type", "termination_clauses",
    "deliverables", "project_scale", "scope_summary",
    "proposal_structure", "mandatory_forms", "page_limits",
    "provided_templates", "submission_instructions",
]

EXTRACTION_PROMPT = """
You are extracting structured data from an RFP (Request for Proposal) document.
Return ONLY a valid JSON object with exactly these keys:
{fields}

Rules:
- Use "not stated" (exact string) if a field is not present in the document.
- For list-type fields (requirements, milestones, deliverables, etc.) join items with newlines.
- Do not add commentary, markdown fences, or any text outside the JSON.
- Do not infer or guess values not explicitly stated in the document.

RFP document:
{text}
"""

RAW_TEXT_LIMIT = 90_000  # guard the 128K context window


def _row(rows: list, idx: int = 0) -> dict:
    """Return row `idx` as a dict, tolerating dict rows or objects with asDict()."""
    r = rows[idx]
    if isinstance(r, dict):
        return r
    if hasattr(r, "asDict"):
        return r.asDict()
    return dict(r)


def _esc(v) -> str:
    """SQL single-quote escape, with a 'not stated' default."""
    return str(v if v not in (None, "") else "not stated").replace("'", "''")


# ── Infrastructure bootstrap ──────────────────────────────────────────────────
def _table_ddl(full_table: str) -> str:
    """Table DDL generated from EXTRACTION_FIELDS so code and table never drift."""
    field_cols = ",\n        ".join(f"{f} STRING" for f in EXTRACTION_FIELDS)
    return f"""
        CREATE TABLE IF NOT EXISTS {full_table} (
        rfp_id STRING,
        original_filename STRING,
        volume_path STRING,
        {field_cols},
        raw_text STRING,
        page_count INT,
        status STRING,
        error_message STRING,
        ingested_at TIMESTAMP
        )
    """


def ensure_infra(
    execute: Execute,
    catalog: str,
    schema: str,
    volume: str,
    full_table: str,
) -> list[str]:
    """
    Ensure the schema, volume, and table exist; create whatever is missing.
    Returns a list of non-fatal warnings.

    A failed CREATE is tolerated when the object turns out to exist anyway —
    e.g. the deployed app's service principal has no CREATE privilege, but the
    objects were provisioned earlier (by a local run as a real user). Missing
    AND uncreatable is fatal: we raise, the app cannot work without them.
    """
    warnings: list[str] = []
    steps = [
        (f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}",
         f"DESCRIBE SCHEMA {catalog}.{schema}"),
        (f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}",
         f"DESCRIBE VOLUME {catalog}.{schema}.{volume}"),
        (_table_ddl(full_table),
         f"DESCRIBE TABLE {full_table}"),
    ]
    for create_sql, probe_sql in steps:
        try:
            execute(create_sql)
        except Exception as create_exc:
            target = probe_sql.split(" ", 2)[2]
            try:
                execute(probe_sql)
                warnings.append(
                    f"No permission to create {target}, but it already exists — continuing."
                )
            except Exception:
                raise RuntimeError(
                    f"{target} is missing and could not be created: {create_exc}"
                ) from create_exc
    return warnings


# ── Parse ──────────────────────────────────────────────────────────────────────
def parse_document(execute: Execute, volume_path: str, rfp_id: str) -> tuple[str, int]:
    """
    Parse the PDF in {volume_path}/{rfp_id}/ via ai_parse_document (runs on the warehouse).
    Returns (raw_text, page_count).
    """
    subfolder = f"{volume_path}/{rfp_id}/"
    # NOTE: no 'descriptionElementTypes' — that runs a vision model to describe every
    # figure/element and is very slow on long docs. We only need the text content.
    rows = execute(f"""
        WITH parsed_documents AS (
            SELECT ai_parse_document(
                content,
                map('version', '2.0')
            ) AS parsed
            FROM READ_FILES('{subfolder}', format => 'binaryFile')
        )
        SELECT to_json(parsed) AS parsed_json
        FROM parsed_documents
    """)

    parsed = json.loads(_row(rows)["parsed_json"])
    elements = parsed["document"]["elements"]

    raw_text = "\n".join(el["content"] for el in elements if el.get("content"))
    page_count = max(
        (el["bbox"][0]["page_id"] for el in elements if el.get("bbox")),
        default=0,
    ) + 1
    return raw_text, page_count


# ── Extract ─────────────────────────────────────────────────────────────────────
def extract_fields(execute: Execute, fm_endpoint: str, raw_text: str) -> dict:
    """Extract all fields in one ai_query call (runs on the warehouse). Returns a dict."""
    text_guard = raw_text[:RAW_TEXT_LIMIT]
    fields_list = ", ".join(EXTRACTION_FIELDS)
    prompt = EXTRACTION_PROMPT.format(fields=fields_list, text=text_guard)
    safe_prompt = prompt.replace("'", "''")  # SQL single-quote escape

    rows = execute(f"""
        SELECT ai_query(
            '{fm_endpoint}',
            '{safe_prompt}',
            modelParameters => named_struct('max_tokens', 2000, 'temperature', 0.0)
        ) AS fields_json
    """)

    raw_json = _row(rows)["fields_json"]
    cleaned = re.sub(r"(`*\s*json)|(\s*\`)", "", raw_json).strip()
    return json.loads(cleaned)


# ── Persist ──────────────────────────────────────────────────────────────────────
def write_to_delta(
    execute: Execute,
    full_table: str,
    rfp_id: str,
    original_filename: str,
    file_path: str,
    extracted: dict,
    raw_text: str,
    page_count: int,
) -> None:
    """MERGE extraction results into the Delta table, setting status='READY'."""
    fields_sql = ", ".join(
        f"'{_esc(extracted.get(f, 'not stated'))}' AS {f}" for f in EXTRACTION_FIELDS
    )

    execute(f"""
        MERGE INTO {full_table} AS t
        USING (
            SELECT
                '{rfp_id}' AS rfp_id,
                '{_esc(original_filename)}' AS original_filename,
                '{_esc(file_path)}' AS volume_path,
                {fields_sql},
                '{_esc(raw_text[:RAW_TEXT_LIMIT])}' AS raw_text,
                {int(page_count)} AS page_count,
                'READY' AS status,
                '' AS error_message,
                current_timestamp() AS ingested_at
        ) AS s
        ON t.rfp_id = s.rfp_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


# ── Orchestration ────────────────────────────────────────────────────────────────
def run_ingestion(
    execute: Execute,
    full_table: str,
    volume_path: str,
    fm_endpoint: str,
    rfp_id: str,
) -> str:
    """
    Full pipeline for one RFP. Returns 'ALREADY_READY', 'READY', or 'FAILED'.
    Reads filename/path from the existing UPLOADED row (no filesystem access needed).
    """
    existing = execute(f"""
        SELECT status, original_filename, volume_path
        FROM {full_table} WHERE rfp_id = '{rfp_id}'
    """)
    if existing and _row(existing).get("status") == "READY":
        return "ALREADY_READY"

    try:
        execute(f"UPDATE {full_table} SET status='PARSING' WHERE rfp_id='{rfp_id}'")

        meta = _row(existing) if existing else {}
        filename = meta.get("original_filename") or "unknown.pdf"
        file_path = meta.get("volume_path") or f"{volume_path}/{rfp_id}/"

        raw_text, page_count = parse_document(execute, volume_path, rfp_id)
        if not raw_text.strip():
            raise ValueError("Parsed text is empty — the PDF may be corrupted or image-only.")

        extracted = extract_fields(execute, fm_endpoint, raw_text)
        write_to_delta(execute, full_table, rfp_id, filename, file_path,
                       extracted, raw_text, page_count)
        return "READY"

    except Exception as exc:
        err = str(exc)[:2000].replace("'", "''")
        execute(f"""
            UPDATE {full_table}
            SET status='FAILED', error_message='{err}'
            WHERE rfp_id='{rfp_id}'
        """)
        return "FAILED"


# ── Chat context (prompt-stuffing) ────────────────────────────────────────────────
def build_system_prompt(row: dict) -> str:
    """Build the chat system prompt from a Delta row (structured fields + raw text)."""
    raw = (row.get("raw_text") or "")[:RAW_TEXT_LIMIT]

    def f(key):
        return row.get(key) or "not stated"

    return f"""You are an assistant answering questions about ONE specific RFP document.
Use ONLY the information provided below. If information is absent, say "not stated in this RFP".
Do not infer, extrapolate, or fabricate details.

== STRUCTURED FIELDS ==
Project name: {f('project_name')}
Issuing organisation: {f('issuing_org')}
Submission deadline: {f('submission_deadline')}
Q&A deadline: {f('qa_deadline')}
Project start/end: {f('project_start_end')}
Milestones: {f('milestones')}
Budget: {f('budget')}
Pricing model: {f('pricing_model')}
Payment terms: {f('payment_terms')}
Penalties: {f('penalties')}
Evaluation weighting — technical / price / team: {f('technical_score_weighting')} / {f('price_weighting')} / {f('team_experience_weighting')}
Functional requirements:
{f('functional_requirements')}
Technical requirements:
{f('technical_requirements')}
Compliance requirements:
{f('compliance_requirements')}
Required certifications: {f('required_certifications')}
Minimum experience: {f('minimum_experience')}
Financial thresholds: {f('financial_thresholds')}
Liability: {f('liability')}
Governing law: {f('governing_law')}
Contract type: {f('contract_type')}
Termination clauses: {f('termination_clauses')}
Deliverables:
{f('deliverables')}
Project scale: {f('project_scale')}
Scope summary: {f('scope_summary')}
Proposal structure: {f('proposal_structure')}
Mandatory forms: {f('mandatory_forms')}
Page limits: {f('page_limits')}
Provided templates: {f('provided_templates')}
Submission instructions:
{f('submission_instructions')}

== FULL DOCUMENT TEXT (first {RAW_TEXT_LIMIT:,} characters) ==
{raw}
"""
