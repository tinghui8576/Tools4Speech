import io
import wave
import os
import time
import pandas as pd
import streamlit as st
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
# -----------------------------
# FILE & AUDIO UTILITIES
# -----------------------------
def slice_wav_bytes(file_path: str, start_sec: float, end_sec: float) -> bytes:
    """Extracts segment slices from a wav file source structure safely."""
    if not os.path.exists(file_path):
        return b""

    with wave.open(file_path, 'rb') as wav:
        params = wav.getparams()
        sr = wav.getframerate()
        start_frame = int(start_sec * sr)
        end_frame = int(end_sec * sr)

        wav.setpos(start_frame)
        frames = wav.readframes(max(1, end_frame - start_frame))

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as out:
        out.setparams(params)
        out.writeframes(frames)
    return buf.getvalue()

def apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = st.session_state.get("col_mapping", {})
    return df.rename(columns=rename_dict)


def _next_available_name(output_dir, prefix):
    path = os.path.join(output_dir, f"{prefix}.txt")

    i = 1
    while os.path.exists(path):
        path = os.path.join(output_dir, f"{prefix}_{i}.txt")
        i += 1
    return path

def load_data() -> pd.DataFrame:
    """Loads previously processed transcript and metadata into the project."""

    result = st.session_state.get("result")
    if not result:
        return None

    project = st.session_state["project"]

    output_dir = result.get("output_dir", "")

    project["files"] = {
        "output_dir": output_dir,
        "transcript_file": result["final_labels"],
        "meta_file": result["metadata_labels"],
        "update_transcript": _next_available_name(
            output_dir, "update_transcript"
        ),
        "update_meta": _next_available_name(
            output_dir, "update_meta"
        ),
    }

    files = project["files"]

    if not all(os.path.exists(f) for f in [files["transcript_file"], files["meta_file"]]):
        st.error("Required source data files missing.")
        return None

    df_ts = pd.read_csv(files["transcript_file"], sep="\t")
    df_md = pd.read_csv(files["meta_file"], sep="\t")

    project["tf_cols"] = list(df_ts.columns)
    project["md_cols"] = list(df_md.columns)

    if "seg_filename" not in df_ts.columns or "seg_filename" not in df_md.columns:
        st.error("Missing 'seg_filename' column.")
        return None

    df = df_ts.merge(df_md, on=["seg_filename", "speaker"], how="inner")
    df = normalize_schema(df, None)

    project["df"] = df
    project["mode"] = "annotation"

    # Reset editor state
    st.session_state["editor"] = {
        "selected_idx": None,
        "dirty": False,
        "hidden_fields": set(),
        "init_hidden_fields": set(),
        "initialized_fields": set(),
        "current_state": {},
        "pending_idx": None,
        "show_discard": False,
    }

    return df
def save_data(df: pd.DataFrame):
    """Saves updated schemas back to user target destination footprints safely."""
    project = st.session_state.get("project", {})
    editor = st.session_state.get("editor", {})

    tf_cols = project.get("tf_cols")
    md_cols = project.get("md_cols")

    files: dict = project.get("files", {})
    reverse_mapping = project.get("reverse_mapping")

    # -----------------------------
    # APPLY REVERSE MAPPING (UI → RAW schema)
    # -----------------------------
    if reverse_mapping:
        df = df.rename(columns=reverse_mapping)

    # -----------------------------
    # OUTPUT PATHS
    # -----------------------------
    transcript_path = files.get("update_transcript")
    metadata_path = files.get("update_meta")

    if not transcript_path or not metadata_path:
        raise ValueError("Missing output file paths in project['files']")
    
    # -----------------------------
    # WRITE TRANSCRIPT FILE
    # -----------------------------
    if tf_cols:
        if not transcript_path:
            raise ValueError("Missing output file paths in project['files']")
        tf_cols = [c for c in tf_cols if c in df.columns]
        df[tf_cols].to_csv(transcript_path, sep="\t", index=False)
    
    # -----------------------------
    # WRITE METADATA FILE
    # -----------------------------
    if md_cols:
        if not metadata_path:
            raise ValueError("Missing output file paths in project['files']")
        md_cols = [c for c in md_cols if c in df.columns]
        df[md_cols].to_csv(metadata_path, sep="\t", index=False)

    # -----------------------------
    # RESET DIRTY STATE AFTER SAVE
    # -----------------------------
    editor["dirty"] = False
    st.session_state["editor"] = editor


# -----------------------------
# DATA ENGINE RESOLUTION HELPERS
# -----------------------------

def normalize_schema(df: pd.DataFrame, mapping: dict):
    # if mapping exists → rename
    if mapping:
        df = df.rename(columns=mapping)

    # ensure all expected columns exist
    for col in PIPELINE_FIELDS:
        if col not in df.columns:
            df[col] = np.nan

    return df

def build_optional_fields(md_cols: list) -> dict:
    """Builds visibility lookup matrix maps using source file constraints."""
    optional = {}
    if "age" in md_cols: optional["age"] = ["age"]
    if "emoCat" in md_cols: optional["emoCat"] = ["emoCat"]
    if "sex" in md_cols: optional["sex"] = ["sex"]
    avd = [c for c in ["arousal", "valence", "dominance"] if c in md_cols]
    if avd: optional["arousal__valence__dominance"] = avd
    return optional


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
def _switch_tab(tab):
    print("switch", tab)
    st.session_state.mode = tab
def _init_state():
    st.session_state.setdefault("project", {
        "mode": None,
        "files": {},
        "df": None,
        "mapping": {},
    })

    st.session_state.setdefault("editor", {
        "selected_idx": None,
        "dirty": False,
        "hidden_fields": set(),
        "init_hidden_fields": set(),
        "initialized_fields": set(),
        "current_state": {},
        "last_selected_idx": None,
        "pending_idx": None,
        "show_discard": False,
    })
def run_upload():
    _init_state()
    st.session_state["project"]["mode"] = "upload"

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



