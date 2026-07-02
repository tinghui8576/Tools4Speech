import streamlit as st
import os
import pandas as pd
import numpy as np
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

def _switch_tab(tab):
    print("switch", tab)
    st.session_state.mode = tab


def _next_available_name(output_dir, prefix):
    path = os.path.join(output_dir, f"{prefix}.txt")

    i = 1
    while os.path.exists(path):
        path = os.path.join(output_dir, f"{prefix}_{i}.txt")
        i += 1
    return path

def normalize_schema(df: pd.DataFrame, mapping: dict):
    # if mapping exists → rename
    if mapping:
        df = df.rename(columns=mapping)

    # ensure all expected columns exist
    for col in PIPELINE_FIELDS:
        if col not in df.columns:
            df[col] = np.nan

    return df




