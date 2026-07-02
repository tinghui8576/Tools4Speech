import io
import wave
import os
import time
import pandas as pd
import streamlit as st
import numpy as np
from page.setup import normalize_schema, _next_available_name, _switch_tab, reset_state, merge_datasets, build_mapping_ui


# -----------------------------
# DATA ENGINE RESOLUTION HELPERS
# -----------------------------

def _choose_output_dir():
    project = st.session_state["project"]

    default_dir = st.session_state.get("_output_dir_suggestion", "outputs")

    output_dir = st.text_input(
        "Output directory",
        value=default_dir,
        key="update_output_dir"
    )

    project["files"]["output_dir"] = output_dir

    return output_dir

def run_upload():

    st.write("Choose a folder that contains transcription and metadata")

    left_col, right_col = st.columns(2)

    df_ts, df_md = None, None

    with left_col:
        ts_file = st.file_uploader(
            "Transcription Labels file input",
            type=["txt", "csv", "tsv"],
            key="tf_file",
        )
        if ts_file:
            df_ts = pd.read_csv(ts_file, sep="\t")
            tf_cols = list(df_ts.columns) if df_ts is not None else []
            st.session_state["project"]["tf_cols"] = tf_cols
            st.session_state.uploaded = True

    with right_col:
        md_file = st.file_uploader(
            "Metadata Labels file input",
            type=["txt", "csv", "tsv"],
            key="md_file",
        )
        if md_file:
            df_md = pd.read_csv(md_file, sep="\t")
            md_cols = list(df_md.columns) if df_md is not None else []
            st.session_state["project"]["md_cols"] = md_cols
            st.session_state.uploaded = True

    output_dir = _choose_output_dir()
    if st.session_state.get("uploaded"):
        if md_file is None:
            st.session_state["project"]["md_cols"] = []  
        if ts_file is None:
            st.session_state["project"]["tf_cols"] = []
        tf_mapping, md_mapping = build_mapping_ui()
    if "merged" not in st.session_state:
        st.session_state.merged = False
    if not st.session_state.merged:
        if st.button("Merge datasets"):
            reset_state()
            
            df_ts = normalize_schema(df_ts, tf_mapping)
            df_md = normalize_schema(df_md, md_mapping)
            df = merge_datasets(df_ts, df_md)
            st.session_state["project"]["df"] = df
            st.session_state["project"]["mapping"] ={
                "tf": tf_mapping,
                "md": md_mapping
            }

            os.makedirs(output_dir, exist_ok=True)

            st.session_state["project"]["files"] = {
                "output_dir": output_dir,
                "update_transcript": _next_available_name(output_dir, "update_transcript"),
                "update_meta": _next_available_name(output_dir, "update_meta"),
            }

            st.session_state.merged = True
            st.rerun()
    else:
        st.success("Dataset loaded into project state.")
        st.session_state.pop("merged", None)
        st.button("Go Annotation?",type="primary", on_click=_switch_tab, args=("✂️ Annotation",))



