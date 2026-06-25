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
TRANSCRIPT_FILE = "outputs/vad/F1F2_quiet_food_1m_01_ch/final_labels.txt"
UPDATED_TRANSCRIPT_FILE = "outputs/vad/F1F2_quiet_food_1m_01_ch/updated_labels.txt"
METADATA_FILE = "outputs/vad/F1F2_quiet_food_1m_01_ch/raw_agesex.txt"
UPDATED_METADATA_FILE = "outputs/vad/F1F2_quiet_food_1m_01_ch/updated_meta.txt"

st.set_page_config(page_title="Visual Timeline Audio Trimmer", initial_sidebar_state="expanded", layout="wide")
st.title("✂️ Visual Audio Range Trimmer & Editor")

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


def load_data() -> pd.DataFrame:
    """Loads and merges target transcription and speaker metadata assets."""
    ts_file = UPDATED_TRANSCRIPT_FILE if os.path.exists(UPDATED_TRANSCRIPT_FILE) else TRANSCRIPT_FILE
    md_file = UPDATED_METADATA_FILE if os.path.exists(UPDATED_METADATA_FILE) else METADATA_FILE

    if not all(os.path.exists(f) for f in [ts_file, md_file]):
        st.error("Required source data files missing from filesystem boundaries.")
        return None

    df_ts = pd.read_csv(ts_file, sep="\t")
    df_md = pd.read_csv(md_file, sep="\t")

    st.session_state["tf_cols"] = list(df_ts.columns)
    st.session_state["md_cols"] = list(df_md.columns)

    if "seg_filename" not in df_ts.columns or "seg_filename" not in df_md.columns:
        st.error("Missing critical structural keys ('seg_filename') across files.")
        return None

    return df_ts.merge(df_md, on=["seg_filename", "speaker"], how="inner")


def save_data(df: pd.DataFrame):
    """Saves updated schemas back to user target destination footprints safely."""
    tf_cols = st.session_state.get("tf_cols")
    md_cols = st.session_state.get("md_cols")
    
    if tf_cols and md_cols:
        df[tf_cols].to_csv(UPDATED_TRANSCRIPT_FILE, sep="\t", index=False)
        df[md_cols].to_csv(UPDATED_METADATA_FILE, sep="\t", index=False)

# -----------------------------
# CORE APP ENGINE STATE INITIALIZATION
# -----------------------------
for key, default in [
    ("df_segments", None), ("selected_idx", None), ("last_selected_idx", None),
    ("pending_idx", None), ("show_discard", False), ("hidden_fields", set()),
    ("initialized_fields", set())
]:
    if key not in st.session_state:
        st.session_state[key] = load_data() if key == "df_segments" else default

df = st.session_state.df_segments

# -----------------------------
# DATA ENGINE RESOLUTION HELPERS
# -----------------------------
def resolve_value(field: str, row_data: pd.Series, default=None):
    """Unified fallback resolution pipeline for active component data blocks."""
    if field not in st.session_state.get("md_cols", []):
        return None
    val = row_data.get(field, None)
    return default if (val is None or pd.isna(val)) else val


def build_optional_fields(md_cols: list) -> dict:
    """Builds visibility lookup matrix maps using source file constraints."""
    optional = {}
    if "age" in md_cols: optional["age"] = ["age"]
    if "emoCat" in md_cols: optional["emoCat"] = ["emoCat"]
    
    avd = [c for c in ["arousal", "valence", "dominance"] if c in md_cols]
    if avd: optional["arousal__valence__dominance"] = avd
    return optional

# -----------------------------
# COMPONENT BUILDERS (UI ELEMENTS)
# -----------------------------
def field_header(field_name, title: str, show_button: bool = True):
    """Renders contextual structural layouts with custom interactive close actions."""
    col1, col2 = st.columns([6, 1])
    field_key = "__".join(field_name) if isinstance(field_name, list) else str(field_name)
    
    col1.markdown(f"##### {title}")
    if show_button:
        # Fixed: Running safely as standard components outside a restrictive form block context
        if col2.button("✖", key=f"hide_{field_key}", type="tertiary"):
            st.session_state.hidden_fields.add(field_key)
            st.rerun()
    else:
        col2.html("<div style='height: 40px;'></div>")


def switch_segment(new_idx: int):
    """Processes pipeline steps for index changes while verifying modification statuses."""
    current = st.session_state.selected_idx
    if current == new_idx:
        return
        
    if st.session_state.get(f"dirty_{current}", False):
        st.session_state.pending_idx = new_idx
        st.session_state.show_discard = True
    else:
        st.session_state.selected_idx = new_idx
        st.session_state.hidden_fields = set()
        st.session_state.initialized_fields = set()
        st.rerun()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.header("⏳ Audio Slices Queue")
if df is not None:
    for idx, row in df.iterrows():
        label = f"[{row.get('start_sec', 0.0):.2f}s - {row.get('end_sec', 0.0):.2f}s] {row.get('speaker', '??')}"
        if st.sidebar.button(label, key=f"seg_{idx}", use_container_width=True):
            switch_segment(idx)

# -----------------------------
# DIALOGS (UNSAVED CHANGES POPUP)
# -----------------------------
if st.session_state.show_discard:
    @st.dialog("⚠️ Unsaved Changes")
    def discard_dialog():
        st.write("You have unsaved workspace adjustments in progress. Discard your modifications?")
        col1, col2 = st.columns(2)
        
        if col1.button("Discard Changes", use_container_width=True):
            old_idx = st.session_state.selected_idx
            st.session_state[f"dirty_{old_idx}"] = False
            st.session_state.selected_idx = st.session_state.pending_idx
            st.session_state.show_discard = False
            st.session_state.pending_idx = None
            st.session_state.hidden_fields.clear()
            st.session_state.initialized_fields.clear()
            st.rerun()
            
        if col2.button("Cancel", use_container_width=True):
            st.session_state.show_discard = False
            st.session_state.pending_idx = None
            st.rerun()
    discard_dialog()
    st.stop()

# -----------------------------
# MAIN EDITOR WORKSPACE
# -----------------------------
if df is not None and st.session_state.selected_idx is not None:
    idx = st.session_state.selected_idx
    row_data = df.iloc[idx]
    
    st.markdown(
        f"<p style='color: gray; font-size: 14px; margin-bottom: 20px;'>"
        # f"📁 <b>Source File Footprint:</b> <code>{row_data['origin_filename']}</code> | "
        f"💾 <b>Target:</b> <code>{row_data['seg_filename']}</code></p>", 
        unsafe_allow_html=True
    )

    if not os.path.exists(str(row_data["origin_filename"])):
        st.error("Audio media object file not found at the configured location.")
        st.stop()

    with wave.open(str(row_data["origin_filename"]), "rb") as w:
        total_file_seconds = w.getnframes() / w.getframerate()

    st.write("---")
    st.markdown("### 🎚️ Drag & Cut Audio Boundaries")

    # Range Slider Configuration Block
    slider_key, dirty_key = f"slider_{idx}", f"dirty_{idx}"
    default_range = (float(row_data["start_sec"]), float(row_data["end_sec"]))

    # reset hook (IMPORTANT: must happen BEFORE widget)
    if st.session_state.get("_reset_slider"):
        st.session_state[slider_key] = default_range
        st.session_state["_reset_slider"] = False

    if slider_key not in st.session_state or st.session_state.last_selected_idx != idx:
        st.session_state[slider_key] = default_range
        st.session_state[dirty_key] = False
        st.session_state.last_selected_idx = idx

    col_slider, col_refresh = st.columns([5, 1])

    with col_slider:
        time_range = st.slider(
            "Isolate Audio Playback Area",
            0.0,
            float(total_file_seconds),
            step=0.05,
            format="%.2f seconds",
            key=slider_key
        )

    with col_refresh:
        if st.button("🔄 Reset Trim", use_container_width=True):
            st.session_state["_reset_slider"] = True
            st.rerun()

    # Audio Fragment Context Playback
    audio_bytes = slice_wav_bytes(str(row_data["origin_filename"]), time_range[0], time_range[1])
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.caption(f"🎵 Active selection run: {(time_range[1] - time_range[0]):.2f} seconds total duration.")

    # -----------------------------
    # EDITOR WORKSPACE BLOCK
    # -----------------------------
    md_cols = st.session_state.get("md_cols", [])
    optional_matrix = build_optional_fields(md_cols)
    
    # Run structural automatic hider logic calculations
    for hide_key, cols in optional_matrix.items():
        if hide_key not in st.session_state.initialized_fields:
            if all(row_data.get(c) is None or pd.isna(row_data.get(c)) for c in cols):
                st.session_state.hidden_fields.add(hide_key)
            st.session_state.initialized_fields.add(hide_key)

    # Replaced st.form with an st.container to fully support the nested header hide buttons
    with st.container(border=True):
        st.markdown("### 📝 Edit Metadata Attributes")
        
        new_text = st.text_area(
            "Transcription text content:", 
            value=str(row_data["transcription"]), 
            key=f"text_input_{idx}",
            height=120
        )
        
        c_att = st.columns(2)
        with c_att[0]:
            st.markdown("##### 👥 Speaker Attribution")
            speakers = sorted(list(df["speaker"].dropna().unique()))
            new_speaker = st.selectbox(
                "Select Identity:", speakers + ["➕ Add New Speaker..."],
                key=f"speaker_input_{idx}",
                index=speakers.index(str(row_data.get("speaker"))) if str(row_data.get("speaker")) in speakers else 0
            )
        with c_att[1]:
            st.markdown("##### 🏷️ Interaction Type")
            types = sorted(list(df["type"].dropna().unique()))
            new_type = st.selectbox(
                "Select Interaction Type Classification:", types + ["➕ Add New Type..."],
                key=f"type_input_{idx}",
                index=types.index(str(row_data.get("type"))) if str(row_data.get("type")) in types else 0
            )

        c_dem = st.columns(2)
        new_sex = np.nan
        if "sex" not in st.session_state.hidden_fields:
            with c_dem[0]:
                field_header("sex", "🧑 Sex Identification", show_button=False)
                sex_opts = ["Female", "Male", "Other"]
                new_sex = st.selectbox("Select Gender Designation:", sex_opts, key=f"sex_input_{idx}", index=sex_opts.index(str(row_data.get("sex", "Female"))) if str(row_data.get("sex")) in sex_opts else 0)

        new_age = np.nan
        if "age" in md_cols and "age" not in st.session_state.hidden_fields:
            with c_dem[1]:
                field_header("age", "⏳ Evaluated Age Component")
                new_age = st.number_input("Enter Numeric Value:", 0, 120, key=f"age_input_{idx}", value=int(resolve_value("age", row_data, 25)))

        c_emo = st.columns(4)
        new_emotion = np.nan
        if "emoCat" not in st.session_state.hidden_fields:
            with c_emo[0]:
                field_header("emoCat", "🎭 Emotion Category")
                emo_opts = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
                new_emotion = st.selectbox("Categorization Strategy:", emo_opts, key=f"emo_input_{idx}", index=emo_opts.index(str(row_data.get("emoCat", "Neutral"))) if str(row_data.get("emoCat")) in emo_opts else 5)

        new_arousal, new_valence, new_dominance = np.nan, np.nan, np.nan
        if any(x in md_cols for x in ["arousal", "valence", "dominance"]) and "arousal__valence__dominance" not in st.session_state.hidden_fields:
            with c_emo[1]:
                field_header(["arousal", "valence", "dominance"], "📈 Arousal", show_button=False)
                new_arousal = st.slider("Arousal Scaling", 0.0, 1.0, float(resolve_value("arousal", row_data, 0.5)), key=f"arousal_input_{idx}")
            with c_emo[2]:
                field_header(["arousal", "valence", "dominance"], "📉 Valence", show_button=False)
                new_valence = st.slider("Valence Scaling", 0.0, 1.0, float(resolve_value("valence", row_data, 0.5)), key=f"valence_input_{idx}")
            with c_emo[3]:
                field_header(["arousal", "valence", "dominance"], "👑 Dominance")
                new_dominance = st.slider("Dominance Scaling", 0.0, 1.0, float(resolve_value("dominance", row_data, 0.5)), key=f"dom_input_{idx}")

        st.write("")
        submit_save = st.button("💾 Save", key=f"save_btn_{idx}", use_container_width=True)

    # Evaluate dirty flag state via deterministic checking logic
    is_changed = (
        (abs(time_range[0] - float(row_data["start_sec"])) > 0.01) or
        (abs(time_range[1] - float(row_data["end_sec"])) > 0.01) or
        (new_text != str(row_data["transcription"]).strip()) or
        (new_speaker != str(row_data.get("speaker", ""))) or
        (new_type != str(row_data.get("type", "")))
    )
    st.session_state[dirty_key] = is_changed

    if submit_save:
        df.at[idx, "start_sec"] = round(time_range[0], 2)
        df.at[idx, "end_sec"] = round(time_range[1], 2)
        df.at[idx, "transcription"] = new_text
        df.at[idx, "speaker"] = new_speaker
        df.at[idx, "type"] = new_type
        df.at[idx, "sex"] = new_sex if "sex" not in st.session_state.hidden_fields else np.nan
        df.at[idx, "age"] = new_age if "age" not in st.session_state.hidden_fields else np.nan
        df.at[idx, "emoCat"] = new_emotion if "emoCat" not in st.session_state.hidden_fields else np.nan
        
        if "arousal__valence__dominance" not in st.session_state.hidden_fields:
            df.at[idx, "arousal"] = round(new_arousal, 2)
            df.at[idx, "valence"] = round(new_valence, 2)
            df.at[idx, "dominance"] = round(new_dominance, 2)
        else:
            for field in ["arousal", "valence", "dominance"]:
                df.at[idx, field] = np.nan

        try:
            save_data(df)
            st.session_state[dirty_key] = False
            st.session_state.df_segments = load_data()
            st.success("Successfully committed workspace changes!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Write operation failure encountered: {e}")

    # -----------------------------
    # SIDEBAR: RESTORE FIELDS UTILITY
    # -----------------------------
    if st.session_state.hidden_fields:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Hidden Interface Blocks")
        for field in sorted(list(st.session_state.hidden_fields)):
            if st.sidebar.button(f"↩️ Restore {field.split('__')[0]}", key=f"restore_{field}"):
                st.session_state.hidden_fields.remove(field)
                st.rerun()
else:
    st.info("👈 Please select an audio segment tracking slice from the sidebar queue to start.")