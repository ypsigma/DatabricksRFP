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
    """
    SQL string-literal escape, with a 'not stated' default.
    Backslashes must be doubled BEFORE quote-doubling: Spark interprets \\n, \\" etc.
    inside string literals, which silently corrupts JSON payloads (json.dumps
    escapes become real control characters on write).
    """
    s = str(v if v not in (None, "") else "not stated")
    return s.replace("\\", "\\\\").replace("'", "''")


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
        confidence_score DOUBLE,
        citations STRING,
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
    # Migration for fields tables created before confidence_score/citations existed.
    # ALTER fails if the columns are already there — that's the expected no-op case.
    try:
        execute(f"ALTER TABLE {fields_table} ADD COLUMNS (confidence_score DOUBLE, citations STRING)")
        logger.info("bootstrap: added confidence_score/citations columns to %s", fields_table)
    except Exception:
        pass
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
def _coerce_confidence(conf) -> str:
    """Map whatever confidence representation comes back to our four labels."""
    if conf is None:
        return "Low"
    if isinstance(conf, (int, float)):  # numeric score 0..1
        return "High" if conf >= 0.8 else "Medium" if conf >= 0.5 else "Low"
    conf = str(conf).strip().capitalize()
    return conf if conf in CONFIDENCE_LEVELS else "Low"


def _unwrap_entry(entry) -> dict:
    """
    Normalize one field's raw entry to
    {"value": str, "confidence": str, "confidence_score": float|None, "citation_ids": list}.
    Handles: bare scalar, list of scalars, and the ai_extract wrapper shapes
    ({"value": ..., "confidence_score": ..., "citation_ids": [...]}).
    """
    confidence = "Low"
    score = None
    citation_ids: list = []
    if isinstance(entry, dict):
        raw_conf = entry.get("confidence", entry.get("confidence_score"))
        if isinstance(raw_conf, (int, float)):
            score = float(raw_conf)
        confidence = _coerce_confidence(raw_conf)
        citation_ids = list(entry.get("citation_ids") or [])
        entry = entry.get("value", entry)
    if isinstance(entry, list):
        item_confs, item_scores = [], []
        for e in entry:
            if isinstance(e, dict):
                rc = e.get("confidence", e.get("confidence_score"))
                if rc is not None:
                    item_confs.append(_coerce_confidence(rc))
                    if isinstance(rc, (int, float)):
                        item_scores.append(float(rc))
                citation_ids.extend(e.get("citation_ids") or [])
        if item_confs:  # conservative: the weakest item speaks for the list
            confidence = max(item_confs, key=CONFIDENCE_LEVELS.index)
        if item_scores:
            score = min(item_scores)
        entry = "\n".join(
            str(e.get("value", e)) if isinstance(e, dict) else str(e)
            for e in entry if e is not None
        )
    value = "not stated" if entry in (None, "", {}, "null") else str(entry)
    if value == "not stated":
        confidence = "Not found"
        score = None
    return {
        "value": value,
        "confidence": confidence,
        "confidence_score": score,
        "citation_ids": citation_ids,
    }


def _resolve_snippets(citation_ids, citation_spans: dict, text: str) -> list:
    """Map citation ids to deduped quoted snippets cut from the source text."""
    out = []
    for cid in dict.fromkeys(citation_ids or []):
        span = citation_spans.get(cid)
        if span:
            snippet = text[span[0]:span[1]].strip()
            if snippet:
                out.append(snippet)
    return out


def _unwrap_roles(entry, citation_spans: dict, text: str) -> dict:
    """
    roles_required as array of role objects, keeping per-role confidence and
    citations. Stored value is a JSON array of
    {title, seniority, requirements, confidence, confidence_score, citations}.
    """
    def _leaf(v):
        """A property may itself be wrapped: {value, confidence_score, citation_ids}."""
        if isinstance(v, dict) and "value" in v:
            return (
                v.get("value"),
                v.get("confidence", v.get("confidence_score")),
                v.get("citation_ids") or [],
            )
        return v, None, []

    if isinstance(entry, dict):  # whole-array wrapper
        entry = entry.get("value", entry)
    roles, confs, scores = [], [], []
    for item in (entry if isinstance(entry, list) else []):
        conf_raw, cit_ids, obj = None, [], item
        if isinstance(item, dict):
            conf_raw = item.get("confidence", item.get("confidence_score"))
            cit_ids = list(item.get("citation_ids") or [])
            if "value" in item and "title" not in item:
                obj = item["value"]
        if isinstance(obj, str):  # legacy/flat shape: bare role string
            obj = {"title": obj}
        if not isinstance(obj, dict):
            continue
        # unwrap leaf-wrapped properties; the title's confidence speaks for the role
        cols = {}
        for col in ("title", "seniority", "requirements"):
            val, col_conf, col_cits = _leaf(obj.get(col))
            cols[col] = "not stated" if val in (None, "", {}, "null") else str(val)
            cit_ids.extend(col_cits)
            if col == "title" and col_conf is not None:
                conf_raw = col_conf
        label = _coerce_confidence(conf_raw)
        if isinstance(conf_raw, (int, float)):
            scores.append(float(conf_raw))
        confs.append(label)
        roles.append({
            **cols,
            "confidence": label,
            "confidence_score": float(conf_raw) if isinstance(conf_raw, (int, float)) else None,
            "citations": _resolve_snippets(cit_ids, citation_spans, text),
        })
    if not roles:
        return {"value": "not stated", "confidence": "Not found",
                "confidence_score": None, "citations": []}
    field_conf = max(confs, key=CONFIDENCE_LEVELS.index) if confs else "Low"
    return {
        "value": json.dumps(roles, ensure_ascii=False),
        "confidence": field_conf,
        "confidence_score": min(scores) if scores else None,
        "citations": [],  # per-role citations live inside the value JSON
    }


def extract_fields(
    execute: Execute,
    fm_endpoint: str,
    raw_text: str,
    extract_schema: str = "",
) -> dict:
    """
    Extract all fields in one call (runs on the warehouse).

    Two backends, chosen by config:
      • extract_schema set   → ai_extract with the schema JSON iterated in the
        portal IE agent (copied into extraction_schema.json), with native
        confidence scores and citations enabled.
      • extract_schema empty → legacy in-code prompt via ai_query.

    Returns {field: {"value": str, "confidence": str}}, normalized so every
    EXTRACTION_FIELDS key is present even if the model omitted it.
    """
    text_guard = raw_text[:RAW_TEXT_LIMIT]

    if extract_schema:
        logger.info("extract: using ai_extract with portal schema")
        safe_text = text_guard.replace("'", "''")    # SQL single-quote escape
        safe_schema = extract_schema.replace("'", "\\'")
        rows = execute(f"""
            SELECT to_json(ai_extract(
                '{safe_text}',
                '{safe_schema}',
                options => map(
                    'version', '2.1',
                    'enableCitations', 'true',
                    'enableConfidenceScores', 'true'
                )
            )) AS fields_json
        """)
    else:
        logger.info("extract: using in-code prompt on %s", fm_endpoint)
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
    if not isinstance(parsed, dict):
        raise ValueError(f"Extraction returned non-object JSON: {cleaned[:200]}")
    # ai_extract v2.1 envelope: {error_message, metadata: {citations…}, response: {fields}}
    citation_spans: dict = {}
    if isinstance(parsed.get("response"), dict):
        if parsed.get("error_message"):
            raise ValueError(f"ai_extract reported an error: {parsed['error_message']}")
        meta = parsed.get("metadata") or {}
        for c in meta.get("citations") or []:
            if c.get("start") is not None and c.get("stop") is not None:
                citation_spans[c.get("id")] = (int(c["start"]), int(c["stop"]))
        parsed = parsed["response"]

    result = {}
    for f in EXTRACTION_FIELDS:
        if f == "roles_required":
            result[f] = _unwrap_roles(parsed.get(f), citation_spans, text_guard)
            continue
        entry = _unwrap_entry(parsed.get(f))
        entry["citations"] = _resolve_snippets(
            entry.pop("citation_ids", []), citation_spans, text_guard,
        )
        result[f] = entry
    return result


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
    def _score_sql(d):
        s = d.get("confidence_score")
        return "NULL" if s is None else str(float(s))

    def _citations_sql(d):
        c = d.get("citations")
        return "NULL" if c is None else f"'{_esc(json.dumps(c))}'"

    values = ",\n".join(
        f"('{rfp_id}', '{f}', '{_esc(d['value'])}', '{_esc(d['confidence'])}', "
        f"{_score_sql(d)}, {_citations_sql(d)}, "
        f"'{source}', {('NULL' if reviewer_id is None else repr(reviewer_id))}, current_timestamp())"
        for f, d in extracted.items()
    )
    execute(f"""
        INSERT INTO {fields_table}
        (rfp_id, field_name, extracted_value, confidence, confidence_score, citations,
         source, reviewer_id, created_ts)
        VALUES {values}
    """)


def get_current_fields(execute: Execute, fields_table: str, rfp_id: str) -> dict:
    """
    Current state per field = the latest row. Returns
    {field: {"value", "confidence", "source"}} in EXTRACTION_FIELDS order.
    """
    rows = execute(f"""
        SELECT field_name, extracted_value, confidence, confidence_score, citations, source
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

    def _parse_citations(raw):
        try:
            # strict=False tolerates control chars in rows written before the
            # backslash-escaping fix in _esc()
            parsed = json.loads(raw, strict=False) if raw else []
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []

    return {
        f: {
            "value": by_name.get(f, {}).get("extracted_value", "not stated"),
            "confidence": by_name.get(f, {}).get("confidence", "Not found"),
            "confidence_score": by_name.get(f, {}).get("confidence_score"),
            "citations": _parse_citations(by_name.get(f, {}).get("citations")),
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


# ── Delete ───────────────────────────────────────────────────────────────────────
def delete_rfp(execute: Execute, full_table: str, fields_table: str, rfp_id: str) -> None:
    """Remove one RFP's rows from both tables. File cleanup is the caller's job."""
    execute(f"DELETE FROM {fields_table} WHERE rfp_id = '{rfp_id}'")
    execute(f"DELETE FROM {full_table} WHERE rfp_id = '{rfp_id}'")
    logger.info("delete %s: table rows removed", rfp_id)


# ── Orchestration ────────────────────────────────────────────────────────────────
def run_ingestion(
    execute: Execute,
    full_table: str,
    volume_path: str,
    fm_endpoint: str,
    rfp_id: str,
    fields_table: str,
    extract_schema: str = "",
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
        extracted = extract_fields(execute, fm_endpoint, raw_text, extract_schema)
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
