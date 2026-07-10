import streamlit as st
import os
import pandas as pd
import numpy as np
# -----------------------------
# CONFIG & PATH CONSTANTS
# -----------------------------

def _switch_tab(tab):
    st.session_state.mode = tab


def _next_available_name(output_dir, prefix):
    path = os.path.join(output_dir, f"{prefix}.txt")

    i = 1
    while os.path.exists(path):
        path = os.path.join(output_dir, f"{prefix}_{i}.txt")
        i += 1
    return path

def normalize_schema(df: pd.DataFrame, mapping: dict = None):
    # if mapping exists → rename
    if mapping:
        df = df.rename(columns=mapping)

    return df

PROJECT_DEFAULT = {
    "mode": None,
    "files": {},
    "df": None,
    "mapping": {},
    "md_cols": [],
    "tf_cols": [],
}

EDITOR_DEFAULT = {
    "selected_idx": None,
    "dirty": False,
    "hidden_fields": set(),
    "init_hidden_fields": set(),
    "initialized_fields": set(),
    "current_state": {},
    "last_selected_idx": None,
    "pending_idx": None,
    "show_discard": False,
}

# ── Session-state initialisation ──────────────────────────────────────────
def init_state():
    for key, default in {
        "running": False,
        "done": False,
        "uploaded": False,
        "log_lines": [],
        "result": None,
        "error": None,
        "_thread": None,
        "_shared": None,
        "tmp_audio": None,
        "ls_url": None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    st.session_state.setdefault("project", PROJECT_DEFAULT.copy())
    st.session_state.setdefault("editor", EDITOR_DEFAULT.copy())

# ── Session-state initialisation ──────────────────────────────────────────
def reset_state() -> None:
    for key, default in {
        "running": False,
        "uploaded": False,
        "done": False,
        "log_lines": [],
        "result": None,
        "error": None,
        "_thread": None,
        "_shared": None,
        "tmp_audio": None,
        "ls_url": None,
    }.items():
        st.session_state[key] = default
    st.session_state["project"] = PROJECT_DEFAULT.copy()
    st.session_state["editor"] = EDITOR_DEFAULT.copy()

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
    "seg_filename",
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
def merge_datasets(df_ts, df_md):
    
    if df_ts is not None and df_md is not None:
        shared_cols = sorted(set(df_ts.columns) & set(df_md.columns))
        st.session_state["project"]["files"]["shared_cols"] = shared_cols
        df = df_ts.merge(df_md, on=shared_cols, how="inner")
        # ensure all expected columns exist
        for col in PIPELINE_FIELDS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    return df_ts if df_ts is not None else df_md

def _guess(field, cols):
    suggestions = {
        "audio_filename": ["origin_filename", "audio_filename"],
        "transcription": ["transcription", "text", "utterance"],
        "start_sec": ["start", "start_sec"],
        "end_sec": ["end", "end_sec"],
        "seg_filename": ["seg_filename", "filename"],
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

def build_mapping_ui():
    project = st.session_state["project"]

    tf_cols = project.get("tf_cols", [])
    md_cols = project.get("md_cols", [])

    if not tf_cols and not md_cols:
        st.info("Upload at least one file to configure mapping")
        return None, None

    st.subheader("Column Mapping")

    tf_mapping = {}
    md_mapping = {}

    mapping = {}

    for field in PIPELINE_FIELDS:

        c1, c2 = st.columns(2)

        tf_choice = "(none)"
        md_choice = "(none)"

        # ---------------- TF ----------------
        if tf_cols:
            with c1:
                default = _guess(field, tf_cols)
                options = ["(none)"] + tf_cols

                tf_choice = st.selectbox(
                    f"{field} → TF column",
                    options=options,
                    key=f"tf_{field}",
                    index=options.index(default) if default in options else 0
                )

        # ---------------- MD ----------------
        if md_cols:
            with c2:
                default = _guess(field, md_cols)
                options = ["(none)"] + md_cols

                md_choice = st.selectbox(
                    f"{field} → MD column",
                    options=options,
                    key=f"md_{field}",
                    index=options.index(default) if default in options else 0
                )


        mapping[field] = {"tf": tf_choice, "md": md_choice}

    # ---------------- VALIDATION ----------------
    missing = [
        f for f in REQUIRED_FIELDS
        if mapping[f]["tf"] == "(none)" and mapping[f]["md"] == "(none)"
    ]

    if missing:
        st.error("Missing required fields: " + ", ".join(missing))
        return None, None

    # ---------------- BUILD FINAL MAPS ----------------
    for field, m in mapping.items():
        if m["tf"] != "(none)":
            tf_mapping[m["tf"]] = field

        if m["md"] != "(none)":
            md_mapping[m["md"]] = field

    return tf_mapping, md_mapping