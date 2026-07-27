# """RFP Assistant — Databricks App (Phase 1)."""
from __future__ import annotations

import os
import time
import uuid
from typing import Generator

import streamlit as st
from databricks import sql as dbsql
from databricks.sdk.core import Config
from openai import OpenAI

import rfp_lib
from rfp_lib import EXTRACTION_FIELDS

# ── Environment config (injected by app.yaml) ─────────────────────────────────
CATALOG             = os.environ["CATALOG"]
SCHEMA              = os.environ["SCHEMA"]
VOLUME              = os.environ["VOLUME"]
TABLE               = os.environ["TABLE"]
FM_ENDPOINT         = os.environ["FM_ENDPOINT"]
WAREHOUSE_HTTP_PATH = os.environ["WAREHOUSE_HTTP_PATH"]
WORKSPACE_HOST      = os.environ["WORKSPACE_HOST"]
KA_ENDPOINT         = os.environ.get("KA_ENDPOINT", "")  # empty = KA disabled

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
FULL_TABLE  = f"{CATALOG}.{SCHEMA}.{TABLE}"


# ── Auth (Databricks Apps service principal) ──────────────────────────────────
# Inside a Databricks App, the SDK auto-discovers the app's service-principal
# credentials from injected env vars (DATABRICKS_CLIENT_ID / _SECRET / _HOST).
# No manual tokens, no per-request header juggling.
@st.cache_resource
def _config() -> Config:
    return Config()


# ── SQL connector ─────────────────────────────────────────────────────────────
def _sql_conn():
    cfg = _config()
    return dbsql.connect(
        server_hostname=cfg.host.replace("https://", ""),
        http_path=WAREHOUSE_HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate,
    )


def _run_sql(query: str, params: dict | None = None):
    """Executes a SQL query and returns all rows as a list of Row-like objects."""
    with _sql_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or {})
        return cursor.fetchall()


def _execute(query: str) -> list[dict]:
    """
    Executor passed to rfp_lib. Runs SQL on the warehouse and returns rows as dicts.
    Returns [] for statements with no result set (UPDATE / MERGE / INSERT).
    """
    with _sql_conn() as conn:
        cur = conn.cursor()
        cur.execute(query)
        if cur.description:
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        return []


# ── FM API client (OpenAI-compatible) ─────────────────────────────────────────
def _fm_client() -> OpenAI:
    """
    OpenAI-compatible client authenticated as the app's service principal.
    Builds the client directly from the SP token so it works on any SDK version
    (avoids serving_endpoints.get_open_ai_client(), which needs a newer SDK).
    """
    cfg = _config()
    headers = cfg.authenticate()                       # {"Authorization": "Bearer <token>"}
    token = headers["Authorization"].split(" ", 1)[1]
    return OpenAI(api_key=token, base_url=f"{cfg.host}/serving-endpoints")


# ── Ingestion (inline, Phase 1) ───────────────────────────────────────────────
# Parsing + extraction + MERGE all live in rfp_lib and run the SAME tested SQL the
# 01_ingestion notebook uses — executed here through the dbsql connector (_execute).

def run_ingestion(rfp_id: str, uploaded_file) -> str:
    """
    Write the uploaded file to the Volume, record an UPLOADED row, then run the
    shared rfp_lib pipeline (ai_parse_document → ai_query → MERGE) on the warehouse.
    Returns 'READY', 'ALREADY_READY', or 'FAILED'.
    """
    filename  = uploaded_file.name
    dest_path = f"{VOLUME_PATH}/{rfp_id}/{filename}"

    try:
        # Write to the Volume via the Files API. The app container has NO Volume
        # FUSE mount, so open("/Volumes/...") would fail — Files API is the way.
        import io
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(config=_config())
        w.files.upload(dest_path, io.BytesIO(uploaded_file.getbuffer()), overwrite=True)
    except Exception as exc:
        return f"FAILED: could not write file to Volume — {exc}"

    # Record the UPLOADED row (rfp_lib reads filename/path back from this row)
    _execute(f"""
        INSERT INTO {FULL_TABLE} (rfp_id, original_filename, volume_path, status)
        VALUES ('{rfp_id}', '{filename.replace("'", "''")}',
                '{dest_path.replace("'", "''")}', 'UPLOADED')
    """)

    # Shared pipeline — identical SQL to the notebook
    return rfp_lib.run_ingestion(
        execute=_execute,
        full_table=FULL_TABLE,
        volume_path=VOLUME_PATH,
        fm_endpoint=FM_ENDPOINT,
        rfp_id=rfp_id,
    )


# ── Chat: Knowledge Assistant path ────────────────────────────────────────────

def _chat_knowledge_assistant(
    rfp_id: str,
    messages: list[dict],
) -> Generator[str, None, None]:
    """
    Streams a response from the Agent Bricks Knowledge Assistant endpoint.
    Passes rfp_id as a retrieval metadata filter so the assistant scopes
    its retrieval to the selected document only.

    NOTE: The exact filter parameter name depends on how the Knowledge Assistant
    was configured (metadata field name set in the KA UI). Adjust 'rfp_id' below
    to match the field name you used when indexing the documents.
    """
    response = _fm_client().chat.completions.create(
        model=KA_ENDPOINT,
        messages=messages,
        max_tokens=2048,
        stream=True,
        extra_body={
            # Databricks Knowledge Assistant retrieval filter.
            # Key name must match the metadata field configured in the KA UI.
            "databricks_options": {
                "retrieval_filter": {"rfp_id": rfp_id}
            }
        },
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Chat: Prompt-stuffing path ────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _load_rfp_row(rfp_id: str) -> dict:
    """Fetches the full Delta row for one RFP. Cached for 2 minutes."""
    rows = _run_sql(f"SELECT * FROM {FULL_TABLE} WHERE rfp_id = '{rfp_id}'")
    if not rows:
        raise ValueError(f"rfp_id {rfp_id} not found in Delta table.")
    # databricks-sql-connector returns Row objects; convert to dict
    row = rows[0]
    return {col: getattr(row, col, None) for col in EXTRACTION_FIELDS + ["raw_text"]}


def _chat_prompt_stuffing(
    rfp_id: str,
    messages: list[dict],
    ) -> Generator[str, None, None]:
    """Streams a response using prompt-stuffing (single-doc-safe)."""
    row = _load_rfp_row(rfp_id)
    system_prompt = rfp_lib.build_system_prompt(row)

    api_messages = [{"role": "system", "content": system_prompt}] + messages

    response = _fm_client().chat.completions.create(
        model=FM_ENDPOINT,
        messages=api_messages,
        max_tokens=2048,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_stream(
    rfp_id: str,
    messages: list[dict],
    ) -> Generator[str, None, None]:
    """
    Tries the Knowledge Assistant endpoint first; falls back to prompt-stuffing.
    If KA_ENDPOINT is not configured, goes straight to prompt-stuffing.
    """
    if KA_ENDPOINT:
        try:
            yield from _chat_knowledge_assistant(rfp_id, messages)
            return
        except Exception as exc:
            st.warning(
                f"Knowledge Assistant unavailable ({exc}). Falling back to prompt-stuffing."
            )
    yield from _chat_prompt_stuffing(rfp_id, messages)


# ── Delta helpers for the UI ───────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def list_ready_rfps() -> list[dict]:
    """Returns READY RFPs ordered by most recently ingested."""
    rows = _run_sql(f"""
        SELECT rfp_id, project_name, original_filename, ingested_at
        FROM {FULL_TABLE}
        WHERE status = 'READY'
        ORDER BY ingested_at DESC
    """)
    return [
        {
            "rfp_id": r.rfp_id,
            "label": r.project_name or r.original_filename,
        }
        for r in rows
    ]


def poll_status(rfp_id: str, timeout_s: int = 300) -> str:
    """Polls the Delta row status until READY, FAILED, or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rows = _run_sql(
            f"SELECT status, error_message FROM {FULL_TABLE} WHERE rfp_id = '{rfp_id}'"
        )
        if rows:
            status = rows[0].status
            if status in ("READY", "FAILED"):
                return status, getattr(rows[0], "error_message", None)
        time.sleep(3)
    return "FAILED", "Ingestion timed out."


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="RFP Assistant", page_icon="📄", layout="wide")
st.title("📄 RFP Assistant")

# ── Infrastructure bootstrap (once per app process) ────────────────────────────
@st.cache_resource(show_spinner="Checking schema / volume / table… (first run may cold-start the warehouse)")
def _ensure_infra() -> list[str]:
    return rfp_lib.ensure_infra(
        execute=_execute,
        catalog=CATALOG,
        schema=SCHEMA,
        volume=VOLUME,
        full_table=FULL_TABLE,
    )

try:
    for _w in _ensure_infra():
        st.warning(_w)
except Exception as _exc:
    st.error(f"Infrastructure check failed — the app cannot run: {_exc}")
    st.stop()

# ── Session state initialisation ──────────────────────────────────────────────
if "rfp_id" not in st.session_state:
    st.session_state.rfp_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar: select or upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("RFP Selection")

    try:
        with st.spinner("Connecting to SQL warehouse… (first query cold-starts it)"):
            ready_rfps = list_ready_rfps()
    except Exception as e:
        st.error(f"Could not load RFPs: {e}")
        st.caption(
            "If this persists, check the SQL warehouse is **Running** and that the "
            "app's service principal has **CAN USE** on it and **SELECT** on the table."
        )
        ready_rfps = []

    if ready_rfps:
        options = [r["rfp_id"] for r in ready_rfps]
        labels  = {r["rfp_id"]: r["label"] for r in ready_rfps}

        selected = st.selectbox(
            "Select an RFP",
            options,
            format_func=lambda rid: labels.get(rid, rid),
            index=options.index(st.session_state.rfp_id)
                  if st.session_state.rfp_id in options else 0,
        )

        # Reset chat history when the user switches documents
        if selected != st.session_state.rfp_id:
            st.session_state.rfp_id = selected
            st.session_state.messages = []
            _load_rfp_row.clear()  # invalidate cache for the old rfp

    else:
        st.info("No RFPs ingested yet. Upload one below.")

    st.divider()
    st.subheader("Upload a new RFP")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded and st.button("Ingest", type="primary"):
        new_rfp_id = uuid.uuid4().hex
        with st.status("Parsing & extracting…", expanded=True) as status_widget:
            result = run_ingestion(new_rfp_id, uploaded)
            if result in ("READY", "ALREADY_READY"):
                status_widget.update(label="✅ Ready!", state="complete")
                st.session_state.rfp_id = new_rfp_id
                st.session_state.messages = []
                list_ready_rfps.clear()  # force dropdown refresh
                st.rerun()
            else:
                status_widget.update(label="❌ Ingestion failed", state="error")

        # Surface the error OUTSIDE the status block so it's always visible.
        if result not in ("READY", "ALREADY_READY"):
            # `result` itself carries the message for pre-Delta failures (e.g. file write).
            st.error(f"Ingestion result: {result}")
            try:
                rows = _run_sql(
                    f"SELECT error_message FROM {FULL_TABLE} WHERE rfp_id='{new_rfp_id}'"
                )
                if rows and rows[0].error_message:
                    st.error(f"Pipeline error: {rows[0].error_message}")
            except Exception:
                pass

# ── Main area: status banner + chat ───────────────────────────────────────────
if not st.session_state.rfp_id:
    st.info("Select or upload an RFP from the sidebar to start chatting.")
    st.stop()

current_rfp_id = st.session_state.rfp_id

# Show basic RFP info as a banner
rows = _run_sql(f"""
    SELECT project_name, original_filename, submission_deadline, budget, status
    FROM {FULL_TABLE} WHERE rfp_id = '{current_rfp_id}'
""")
if rows:
    r = rows[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Project", r.project_name or r.original_filename)
    col2.metric("Submission deadline", r.submission_deadline or "not stated")
    col3.metric("Budget", r.budget or "not stated")

    if r.status != "READY":
        st.warning(f"Status: {r.status}")
        st.stop()

st.divider()

# Render conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about this RFP…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream = chat_stream(
            current_rfp_id,
            # Pass only the conversation turns (not the system prompt) to the API
            [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        )
        answer = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": answer})
