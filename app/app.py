"""Minimal Databricks App — deployment smoke test."""
import os

import streamlit as st

st.set_page_config(page_title="DatabricksRFP Test App", page_icon="✅")

st.title("✅ DatabricksRFP — Deployment Test")
st.write(
    "If you can see this page, the app was deployed successfully "
    "from the GitHub-synced repo."
)

# Databricks Apps passes the logged-in user's email in this header.
headers = st.context.headers
user = headers.get("X-Forwarded-Email", "unknown (running locally?)")
st.info(f"Logged in as: **{user}**")

st.subheader("Quick interactivity check")
name = st.text_input("Your name", value="")
if st.button("Say hello"):
    st.success(f"Hello, {name or 'stranger'}! Streamlit is working.")

with st.expander("App environment"):
    st.write({
        "DATABRICKS_APP_NAME": os.environ.get("DATABRICKS_APP_NAME", "-"),
        "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST", "-"),
        "PORT": os.environ.get("DATABRICKS_APP_PORT", "8080"),
    })
