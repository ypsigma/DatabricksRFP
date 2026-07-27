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
import logging
import re
from typing import Callable

Execute = Callable[[str], list]

logger = logging.getLogger("rfp.lib")

# ── Field set (single source of truth) ────────────────────────────────────────
EXTRACTION_FIELDS = [
    "client_name",
    "domain",
    "summary",
    "submission_deadline",
    "budget",
    "roles_required",
    "technologies",
    "evaluation_criteria",
    "compliance_requirements",
    "response_format",
]

CONFIDENCE_LEVELS = ["High", "Medium", "Low", "Not found"]

EXTRACTION_PROMPT = """
You are extracting structured data from an RFP (Request for Proposal) document.
Return ONLY a valid JSON object with exactly these keys:
{fields}

Each key must map to an object with exactly two keys:
  "value": string — the extracted content
  "confidence": one of "High", "Medium", "Low", "Not found"

Rules:
- confidence "High" = explicitly and unambiguously stated in the document.
- confidence "Medium" = stated but ambiguous, scattered, or requires light interpretation.
- confidence "Low" = only partially stated or inferred from context.
- confidence "Not found" = absent; then value must be the exact string "not stated".
- roles_required: needed roles/team with count and seniority when stated, one per line.
- technologies / evaluation_criteria / compliance_requirements: one item per line.
- Do not add commentary, markdown fences, or any text outside the JSON.
- Do not invent values that are not in the document.

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


def _fields_table_ddl(fields_table: str) -> str:
    """Long-format field store: one row per field per extraction or correction."""
    return f"""
        CREATE TABLE IF NOT EXISTS {fields_table} (
        rfp_id STRING,
        field_name STRING,
        extracted_value STRING,
        confidence STRING,
        source STRING,
        reviewer_id STRING,
        created_ts TIMESTAMP
        )
    """


def ensure_infra(
    execute: Execute,
    catalog: str,
    schema: str,
    volume: str,
    full_table: str,
    fields_table: str,
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
        (_fields_table_ddl(fields_table),
         f"DESCRIBE TABLE {fields_table}"),
    ]
    for create_sql, probe_sql in steps:
        target = probe_sql.split(" ", 2)[2]
        logger.info("bootstrap: ensuring %s …", target)
        try:
            execute(create_sql)
            logger.info("bootstrap: %s OK", target)
        except Exception as create_exc:
            logger.warning("bootstrap: CREATE failed for %s (%s); probing existence", target, create_exc)
            try:
                execute(probe_sql)
                logger.info("bootstrap: %s exists — continuing without create rights", target)
                warnings.append(
                    f"No permission to create {target}, but it already exists — continuing."
                )
            except Exception:
                logger.error("bootstrap: %s missing and uncreatable", target)
                raise RuntimeError(
                    f"{target} is missing and could not be created: {create_exc}"
                ) from create_exc
    logger.info("bootstrap: all infrastructure verified")
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
    """
    Extract all fields in one ai_query call (runs on the warehouse).
    Returns {field: {"value": str, "confidence": str}}, normalized so every
    EXTRACTION_FIELDS key is present even if the model omitted it.
    """
    text_guard = raw_text[:RAW_TEXT_LIMIT]
    fields_list = ", ".join(EXTRACTION_FIELDS)
    prompt = EXTRACTION_PROMPT.format(fields=fields_list, text=text_guard)
    safe_prompt = prompt.replace("'", "''")  # SQL single-quote escape

    rows = execute(f"""
        SELECT ai_query(
            '{fm_endpoint}',
            '{safe_prompt}',
            modelParameters => named_struct('max_tokens', 4000, 'temperature', 0.0)
        ) AS fields_json
    """)

    raw_json = _row(rows)["fields_json"]
    cleaned = re.sub(r"(`*\s*json)|(\s*\`)", "", raw_json).strip()
    parsed = json.loads(cleaned)

    normalized = {}
    for f in EXTRACTION_FIELDS:
        entry = parsed.get(f) or {}
        if not isinstance(entry, dict):  # model returned a bare string
            entry = {"value": str(entry), "confidence": "Low"}
        conf = entry.get("confidence", "Low")
        normalized[f] = {
            "value": str(entry.get("value", "not stated")),
            "confidence": conf if conf in CONFIDENCE_LEVELS else "Low",
        }
    return normalized


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


# ── Field rows: extraction history + reviewer overrides ─────────────────────────
def write_field_rows(
    execute: Execute,
    fields_table: str,
    rfp_id: str,
    extracted: dict,
    source: str = "ai_extracted",
    reviewer_id: str | None = None,
) -> None:
    """Append one row per field. Corrections are new rows, never updates."""
    values = ",\n".join(
        f"('{rfp_id}', '{f}', '{_esc(d['value'])}', '{_esc(d['confidence'])}', "
        f"'{source}', {('NULL' if reviewer_id is None else repr(reviewer_id))}, current_timestamp())"
        for f, d in extracted.items()
    )
    execute(f"""
        INSERT INTO {fields_table}
        (rfp_id, field_name, extracted_value, confidence, source, reviewer_id, created_ts)
        VALUES {values}
    """)


def get_current_fields(execute: Execute, fields_table: str, rfp_id: str) -> dict:
    """
    Current state per field = the latest row. Returns
    {field: {"value", "confidence", "source"}} in EXTRACTION_FIELDS order.
    """
    rows = execute(f"""
        SELECT field_name, extracted_value, confidence, source
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY field_name ORDER BY created_ts DESC
            ) AS rn
            FROM {fields_table}
            WHERE rfp_id = '{rfp_id}'
        )
        WHERE rn = 1
    """)
    by_name = {r["field_name"]: r for r in map(dict, (_row(rows, i) for i in range(len(rows))))}
    return {
        f: {
            "value": by_name.get(f, {}).get("extracted_value", "not stated"),
            "confidence": by_name.get(f, {}).get("confidence", "Not found"),
            "source": by_name.get(f, {}).get("source", "ai_extracted"),
        }
        for f in EXTRACTION_FIELDS
    }


def save_overrides(
    execute: Execute,
    fields_table: str,
    full_table: str,
    rfp_id: str,
    changes: dict,
    reviewer_id: str,
) -> None:
    """
    Persist reviewer corrections: append override rows to the fields table
    (audit trail), then sync the wide table so chat/banner see current values.
    `changes` = {field: new_value} for edited fields only.
    """
    if not changes:
        return
    entries = {
        f: {"value": v, "confidence": "High"}  # human-confirmed = High
        for f, v in changes.items()
    }
    write_field_rows(
        execute, fields_table, rfp_id, entries,
        source="reviewer_override", reviewer_id=reviewer_id,
    )
    set_clause = ", ".join(f"{f} = '{_esc(v)}'" for f, v in changes.items())
    execute(f"UPDATE {full_table} SET {set_clause} WHERE rfp_id = '{rfp_id}'")
    logger.info("overrides %s: %d fields saved by %s", rfp_id, len(changes), reviewer_id)


# ── Orchestration ────────────────────────────────────────────────────────────────
def run_ingestion(
    execute: Execute,
    full_table: str,
    volume_path: str,
    fm_endpoint: str,
    rfp_id: str,
    fields_table: str,
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

        logger.info("ingestion %s: parsing document …", rfp_id)
        raw_text, page_count = parse_document(execute, volume_path, rfp_id)
        logger.info("ingestion %s: parsed %d pages, %d chars", rfp_id, page_count, len(raw_text))
        if not raw_text.strip():
            raise ValueError("Parsed text is empty — the PDF may be corrupted or image-only.")

        logger.info("ingestion %s: extracting fields …", rfp_id)
        extracted = extract_fields(execute, fm_endpoint, raw_text)
        logger.info("ingestion %s: extracted %d fields, writing to Delta", rfp_id, len(extracted))
        flat_values = {f: d["value"] for f, d in extracted.items()}
        write_to_delta(execute, full_table, rfp_id, filename, file_path,
                       flat_values, raw_text, page_count)
        write_field_rows(execute, fields_table, rfp_id, extracted)
        logger.info("ingestion %s: READY", rfp_id)
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

    field_lines = "\n".join(
        f"{f.replace('_', ' ').capitalize()}:\n{row.get(f) or 'not stated'}"
        for f in EXTRACTION_FIELDS
    )

    return f"""You are an assistant answering questions about ONE specific RFP document.
Use ONLY the information provided below. If information is absent, say "not stated in this RFP".
Do not infer, extrapolate, or fabricate details.
The structured fields have been reviewed by a human and take precedence over the
document text if they disagree.

== STRUCTURED FIELDS (human-reviewed) ==
{field_lines}

== FULL DOCUMENT TEXT (first {RAW_TEXT_LIMIT:,} characters) ==
{raw}
"""
