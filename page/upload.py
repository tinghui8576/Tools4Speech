import io
import wave
import os
import time
import pandas as pd
import streamlit as st
import numpy as np
from page.setup import normalize_schema, _next_available_name, _switch_tab
# -----------------------------
# CONFIG & PATH CONSTANTS
# -----------------------------

REQUIRED_FIELDS = [
    "audio_filename",
    "transcription",
    "start_sec",
    "end_sec",
]

PIPELINE_FIELDS = [
    "audio_filename",
    "transcription",
    "filename",
    'start_sec',
    'end_sec',
    "speaker",
    "type",
    "sex",
    "age",
    "emoCat",
    "arousal",
    "valence",
    "dominance",
]

# -----------------------------
# DATA ENGINE RESOLUTION HELPERS
# -----------------------------
def _upload_files():
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

    with right_col:
        md_file = st.file_uploader(
            "Metadata Labels file input",
            type=["txt", "csv", "tsv"],
            key="md_file",
        )
        if md_file:
            df_md = pd.read_csv(md_file, sep="\t")

    return df_ts, df_md

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
def _build_mapping_ui(df_ts, df_md):
    project = st.session_state["project"]

    tf_cols = list(df_ts.columns) if df_ts is not None else []
    md_cols = list(df_md.columns) if df_md is not None else []

    project["tf_cols"] = tf_cols
    project["md_cols"] = md_cols

    all_cols = sorted(set(tf_cols + md_cols))

    if not all_cols:
        st.info("Upload at least one file to configure mapping")
        return None

    st.subheader("Column Mapping")

    def guess(field, cols):
        suggestions = {
            "audio_filename": ["origin_filename", "audio_filename"],
            "transcription": ["transcription", "text", "utterance"],
            "start_sec": ["start", "start_sec"],
            "end_sec": ["end", "end_sec"],
            "filename": ["filename"],
            "speaker": ["speaker", "spk", "speaker_id"],
            "type": ["type", "channel"],
            "emoCat": ["emoCat", "emotion", "label"],
            "age": ["age"],
            "sex": ["sex", "gender"],
            "arousal": ["arousal"],
            "valence": ["valence"],
            "dominance": ["dominance"],
        }

        for s in suggestions.get(field, []):
            if s in cols:
                return s
        return "(none)"

    mapping = {}

    for field in PIPELINE_FIELDS:
        options = ["(none)"] + all_cols

        default = guess(field, all_cols)
        default_index = options.index(default) if default in options else 0

        mapping[field] = st.selectbox(
            f"{field} → column",
            options=options,
            index=default_index
        )

    missing = [
        f for f in REQUIRED_FIELDS
        if mapping[f] == "(none)"
    ]

    if missing:
        st.error(
            "Please map the following required fields:\n" +
            ", ".join(missing)
        )
        st.stop()

    rename_dict = {}
    for pipeline_field, csv_column in mapping.items(): 
        if csv_column != "(none)": 
            rename_dict[csv_column] = pipeline_field
    return rename_dict

def _merge_datasets(df_ts, df_md):
    if df_ts is not None and df_md is not None:
        shared_cols = sorted(set(df_ts.columns) & set(df_md.columns))
        st.session_state["project"]["files"]["shared_cols"] = shared_cols
        return df_ts.merge(df_md, on=shared_cols, how="inner")

    return df_ts if df_ts is not None else df_md

def run_upload():

    df_ts, df_md = _upload_files()
    output_dir = _choose_output_dir()
    mapping = _build_mapping_ui(df_ts, df_md)
    if "merged" not in st.session_state:
        st.session_state.merged = False
    if not st.session_state.merged:
        if st.button("Merge datasets"):
            df = _merge_datasets(df_ts, df_md)

            df = normalize_schema(df, mapping)
            st.session_state["project"]["df"] = df
            st.session_state["project"]["mapping"] = mapping

            os.makedirs(output_dir, exist_ok=True)

            st.session_state["project"]["files"] = {
                "output_dir": output_dir,
                "update_transcript": _next_available_name(output_dir, "update_transcript"),
                "update_meta": _next_available_name(output_dir, "update_meta"),
            }

            # reset editor when new dataset loads
            st.session_state["editor"] = {
                "selected_idx": None,
                "dirty": False,
                "hidden_fields": set(),
                "current_state": {},
                "init_hidden_fields": set(),
                "initialized_fields": set(),
                "last_selected_idx": None,
                "pending_idx": None,
                "show_discard": False,
            }
            st.session_state.merged = True
            st.rerun()
    else:
        st.success("Dataset loaded into project state.")
        st.session_state.pop("merged", None)
        st.button("Go Annotation?",type="primary", on_click=_switch_tab, args=("✂️ Annotation",))



