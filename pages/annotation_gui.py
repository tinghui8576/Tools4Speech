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
print(st.session_state.get("result"))
PIPELINE_FIELDS = [
    "audio_path",
    "filename", 
    "start_sec",
    "end_sec",
    "transcription",
    "speaker",
    "type",
    "sex",
    "age",
    "emoCat",
    "arousal",
    "valence",
    "dominance",
]

REQUIRED_FIELDS = [
    "audio_filename",
    "transcription",
    "start_sec",
    "end_sec",
]
st.session_state["tf_cols"] = []
st.session_state["md_cols"] = []

def write_back(df, idx, sample, mapping):
    for field, col in mapping.items():
        if col != "(none)" and field in sample:
            df.at[idx, col] = sample[field]
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
    df_to_save = df.rename(columns=st.session_state["reverse_mapping"])
    if tf_cols and md_cols:
        df_to_save[tf_cols].to_csv(UPDATED_TRANSCRIPT_FILE, sep="\t", index=False)
        df_to_save[md_cols].to_csv(UPDATED_METADATA_FILE, sep="\t", index=False)

# -----------------------------
# DATA ENGINE RESOLUTION HELPERS
# -----------------------------
def _resolve_value(field, row_data, default=None):
    if field in st.session_state.get("current_state", {}):
        return st.session_state.current_state[field]

    value = row_data.get(field)

    if pd.isna(value):
        return default

    return value

# def _resolve_value(field: str, row_data: pd.Series, default=None): 
#     """Unified fallback resolution pipeline for active component data blocks.""" 
#     if field not in st.session_state.get("md_cols", []) and field not in st.session_state.get("tf_cols", []): 
#         return None 
#     if st.session_state.get("current_state", {}).get(field) is not None: 
#         return st.session_state.current_state[field] 
    
#     val = row_data.get(field, None) 

#     return default if (val is None or pd.isna(val)) else val

def build_optional_fields(md_cols: list) -> dict:
    """Builds visibility lookup matrix maps using source file constraints."""
    optional = {}
    if "age" in md_cols: optional["age"] = ["age"]
    if "emoCat" in md_cols: optional["emoCat"] = ["emoCat"]
    
    avd = [c for c in ["arousal", "valence", "dominance"] if c in md_cols]
    if avd: optional["arousal__valence__dominance"] = avd
    return optional

# -----------------------------
# INTEGRATED DIRTY CHECK CALLBACK
# -----------------------------
def _changed(field, current, original):
    return (
        field in st.session_state.initialized_fields
        and field not in st.session_state.hidden_fields
        and current != original
    )

def _changed_float(field, current, original, tol=0.01):
    return (
        field in st.session_state.initialized_fields
        and field not in st.session_state.hidden_fields
        and abs(float(current) - float(original)) > tol
    )

def _hidden_changed():
    return st.session_state.get("init_hidden_fields", set()) != st.session_state.get("hidden_fields", set())

def _csv_column(field):
    return st.session_state["col_mapping"].get(field, "(none)")

def check_dirty_callback():
    """
    Evaluates changes instantaneously whenever a widget value changes,
    comparing current state values safely against baseline row data.
    """
    idx = st.session_state.selected_idx
    df = st.session_state.get("df_segments")
    if idx is None or df is None:
        return
    row_data = df.iloc[idx]
    # Extract data securely from state or fallbacks
    w_range = st.session_state.get(f"slider_{idx}", (float(row_data["start_sec"]), float(row_data["end_sec"])))
    w_text = st.session_state.get(f"transc_{idx}", str(row_data["transcription"])).strip()
    w_spk = st.session_state.get(f"spk_select_{idx}", str(row_data.get("speaker", ""))).strip()
    w_type = st.session_state.get(f"type_select_{idx}", str(row_data.get("type", ""))).strip()
    w_sex = st.session_state.get(f"sex_select_{idx}", str(row_data.get("sex", "")))
    w_age = st.session_state.get(f"age_input_{idx}",_resolve_value("age",row_data, 25))
    w_emo = st.session_state.get(f"emo_select_{idx}",_resolve_value("emoCat",row_data, "Neutral"))
    w_arousal = st.session_state.get(f"arousal_input_{idx}", _resolve_value("arousal", row_data, 0.5))
    w_valence = st.session_state.get(f"valence_input_{idx}",_resolve_value("valence", row_data, 0.5))
    w_dom = st.session_state.get(f"dom_input_{idx}",_resolve_value("dominance", row_data, 0.5))
    
    is_dirty = (
        abs(w_range[0] - float(row_data["start_sec"])) > 0.01
        or abs(w_range[1] - float(row_data["end_sec"])) > 0.01
        or w_text != str(row_data["transcription"]).strip()
        or w_spk != str(row_data.get("speaker", "")).strip()
        or w_type != str(row_data.get("type", "")).strip()
        or _changed("sex", w_sex, str(row_data.get("sex", "")))
        or _changed("age", int(w_age), int(_resolve_value("age", row_data, 25)))
        or _changed("emoCat", str(w_emo), str(_resolve_value("emoCat", row_data, "Neutral")))
        or _changed_float("arousal", w_arousal, _resolve_value("arousal", row_data, 0.5))
        or _changed_float("valence", w_valence, _resolve_value("valence", row_data, 0.5))
        or _changed_float("dominance", w_dom, _resolve_value("dominance", row_data, 0.5))
        or _hidden_changed()
    )
    st.session_state[f"dirty_{idx}"] = is_dirty

# -----------------------------
# COMPONENT BUILDERS (UI ELEMENTS)
# -----------------------------
def _field_header(field_name, title: str, show_button: bool = True):
    """Renders contextual structural layouts with custom interactive close actions."""
    col1, col2 = st.columns([6, 1])
    field_key = "__".join(field_name) if isinstance(field_name, list) else str(field_name)
    
    col1.markdown(f"##### {title}")
    if show_button:
        if col2.button("✖", key=f"hide_{field_key}", type="tertiary"):
            st.session_state.hidden_fields.add(field_key)
            check_dirty_callback()
            st.rerun()
    else:
        col2.html("<div style='height: 40px;'></div>")

def _switch_segment(new_idx: int):
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
        st.session_state.current_state = {}
        st.rerun()



def run_upload():
    st.write("Choose a folder that contain transcption and metadata")
    left_col, right_col = st.columns(2)
    if "shared_cols" not in st.session_state:
        st.session_state["shared_cols"] = []
    mapping = {}

    all_cols = []
    with left_col:
        ts_file = st.file_uploader(
            f"Transcrption Labels file input",
            type=["txt", "csv", "tsv"],
            key=f"tf_file",
        )
        if ts_file:
            df_ts = pd.read_csv(ts_file, sep="\t")
            st.session_state["tf_cols"] = list(df_ts.columns)

    with right_col:
        md_file = st.file_uploader(
            f"Metadata Labels file input",
            type=["txt", "csv", "tsv"],
            key=f"md_file",
        )
        if md_file:
            df_md = pd.read_csv(md_file, sep="\t")
            st.session_state["md_cols"] = list(df_md.columns)
    all_cols = sorted(set(st.session_state["tf_cols"] + st.session_state["md_cols"]))
  
    if not all_cols:
        st.info("Upload at least one file to configure mapping")
        return
    st.subheader("Column Mapping")

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

    def guess(field, cols):
        suggestions = {
            "audio_filename": ["origin_filename", "audio_filename"],
            "transcription": ["transcription", "text", "utterance"],
            'start_sec': ["start", "start_sec"],
            'end_sec': ["end", 'end_sec'],
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

        mapping[field] = st.selectbox(
            f"{field} → column",
            options=options,
            index=options.index(guess(field, all_cols))
        )

    st.session_state["col_mapping"] = mapping
    st.session_state["reverse_mapping"] = {
        v: k
        for k, v in mapping.items()
        if v != "(none)"
    }   
    missing = [
        f for f in REQUIRED_FIELDS
        if mapping[f] == "(none)"
    ]

    if missing:
        st.error(
            "Please map the following required fields:\n"
            + ", ".join(missing)
        )
        st.stop()
    if st.button("Merge datasets"):
        if ts_file and md_file:  
            shared_cols = sorted(set(df_ts.columns) & set(df_md.columns))
            st.session_state["shared_cols"] = shared_cols
            df = df_ts.merge(df_md, on=shared_cols, how="inner")
        elif ts_file:
            df = df_ts
        elif md_file:
            df = df_md
        else:
            st.error("Please upload at least one transcription or metadata file.")
            return
        
        rename_dict = {}

        for pipeline_field, csv_column in mapping.items():
            if csv_column != "(none)":
                rename_dict[csv_column] = pipeline_field

        df = df.rename(columns=rename_dict)
        st.session_state["df_segments"] = df
        print(df)

        st.success("Merged successfully!")
        st.rerun()
def run_annotation():
    
    df = st.session_state.df_segments
    print(df)
        # -----------------------------
    # SIDEBAR NAVIGATION
    # -----------------------------
    st.sidebar.header("⏳ Audio Slices Queue")
    if df is not None:
        for idx, row in df.iterrows():
            label = f"[{row.get('start_sec', 0.0):.2f}s - {row.get('end_sec', 0.0):.2f}s] {row.get('speaker', '??')}"
            if st.sidebar.button(label, key=f"seg_{idx}", use_container_width=True):
                _switch_segment(idx)

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
                st.session_state.current = True
                st.session_state.hidden_fields.clear()
                st.session_state.current_state.clear()
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
            # f"📁 <b>Source File Footprint:</b> <code>{row_data['audio_filename']}</code> | "
            f"💾 <b>Target:</b> <code>{row_data['seg_filename']}</code></p>", 
            unsafe_allow_html=True
        )

        if not os.path.exists(str(row_data["audio_filename"])):
            st.error("Audio media object file not found at the configured location.")
            st.stop()

        with wave.open(str(row_data["audio_filename"]), "rb") as w:
            total_file_seconds = w.getnframes() / w.getframerate()

        st.write("---")
        st.markdown("### 🎚️ Drag & Cut Audio Boundaries")

        # Range Slider Configuration Block
        slider_key, dirty_key = f"slider_{idx}", f"dirty_{idx}"
        default_range = (float(row_data["start_sec"]), float(row_data["end_sec"]))

        # reset hook (IMPORTANT: must happen BEFORE widget)
        if st.session_state.get("_reset_slider"):
            st.session_state[slider_key] = default_range
            time_range = default_range
            st.session_state["_reset_slider"] = False

        if slider_key not in st.session_state or st.session_state.last_selected_idx != idx or dirty_key not in st.session_state:
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
                key=slider_key,
                value=default_range,
                on_change=check_dirty_callback
            )
        with col_refresh:
            if st.button("🔄 Reset Trim", use_container_width=True):
                st.session_state["_reset_slider"] = True
                st.rerun()
        selected_start, selected_end = time_range
        

        # Audio Fragment Context Playback
        audio_bytes = slice_wav_bytes(str(row_data["audio_filename"]), selected_start, selected_end )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.caption(f"🎵 Active selection run: {(selected_end - selected_start):.2f} seconds total duration.")

        # -----------------------------
        # EDITOR WORKSPACE BLOCK
        # -----------------------------
        md_cols = st.session_state.get("md_cols", [])
        optional_matrix = build_optional_fields(md_cols)
        
        # Run structural automatic hider logic calculations
        for hide_key, cols in optional_matrix.items():
            if hide_key not in st.session_state.initialized_fields:
                if all(row_data.get(c) is None or pd.isna(row_data.get(c)) for c in cols):
                    st.session_state.init_hidden_fields.add(hide_key)
                    st.session_state.hidden_fields.add(hide_key)
                st.session_state.initialized_fields.add(hide_key)
        # Replaced st.form with an st.container to fully support the nested header hide buttons
        with st.container(border=True):
            with st.expander("📝 Edit Transcription Notes", expanded=True):
                current_text = _resolve_value("transcription", row_data, "")
                new_text = st.text_area(
                    "Transcription text content:", 
                    value=str(current_text ), 
                    key=f"text_input_{idx}",
                    height=max(100, 50 * int(len(current_text) / 100))
                )
            with st.expander("Characteristics", expanded=True):
                c_att = st.columns(2)
                with c_att[0]:
                    st.markdown("##### 👥 Speaker Attribution")
                    # speakers = sorted(list(df["speaker"].dropna().unique()))
                    # new_speaker = st.selectbox(
                    #     "Select Identity:", speakers + ["➕ Add New Speaker..."],
                    #     key=f"speaker_input_{idx}",
                    #     index=speakers.index(str(row_data.get("speaker"))) if str(row_data.get("speaker")) in speakers else 0
                    # )

                    existing_speakers = sorted(list(df["speaker"].dropna().unique()))
                    current_speaker = _resolve_value("speaker", row_data, "P1")
                    speaker_options = existing_speakers + ["➕ Add New Speaker..."]
                    default_spk_idx = existing_speakers.index(current_speaker) if current_speaker in existing_speakers else 0
                    final_speaker = st.selectbox(
                            "Select Speaker identity:",
                            speaker_options,
                            index=default_spk_idx,
                            key=f"spk_select_{idx}",
                            on_change=check_dirty_callback

                        )

                with c_att[1]:
                    st.markdown("##### 🏷️ Interaction Type")
                    existing_types = sorted(list(df["type"].dropna().unique()))
                    type_options = existing_types + ["➕ Add New Type..."]
                    current_type = _resolve_value("type", row_data, "")
                    default_type_idx = existing_types.index(current_type) if current_type in existing_types else 0
                    final_type = st.selectbox(
                        "Select Interaction Type:",
                        type_options,
                        index=default_type_idx,
                        key=f"type_select_{idx}",
                        on_change=check_dirty_callback
                    )

                c_dem = st.columns(2)
                if "sex" not in st.session_state.hidden_fields and "sex" in md_cols:
                    current_sex = _resolve_value("sex", row_data, "")
                    sex_options = ["Female", "Male", "Other"]
                    with c_dem[0]:
                        _field_header("sex", "🧑 Sex Identification", show_button=False)
                        default_sex_idx = sex_options.index(current_sex) if current_sex in sex_options else 0
                        final_sex = st.selectbox(
                            "Select Sex:",
                            sex_options,
                            index=default_sex_idx,
                            key=f"sex_select_{idx}",
                            on_change=check_dirty_callback)
                        # sex_opts = ["Female", "Male", "Other"]
                        # new_sex = st.selectbox("Select Gender Designation:", sex_opts, key=f"sex_input_{idx}", index=sex_opts.index(str(row_data.get("sex", "Female"))) if str(row_data.get("sex")) in sex_opts else 0)
                else:
                    final_sex = None
                if "age" not in st.session_state.hidden_fields and "age" in md_cols:
                    current_age = _resolve_value("age", row_data, 25)
                    with c_dem[1]:
                        _field_header("age", "⏳ Age")
                        final_age = st.number_input(
                            "Age",
                            0, 120,
                            value=int(current_age),
                            key=f"age_input_{idx}",
                            on_change=check_dirty_callback
                        )
                else:
                    final_age = None

                c_emo = st.columns(4)
                emotion_options = [ "Anger", "Contempt", "Disgust", "Fear","Happiness", "Neutral", "Sadness", "Surprise", "Other"]
                if "emoCat" not in st.session_state.hidden_fields and "emoCat" in md_cols:
                    current_emotion = _resolve_value("emoCat", row_data, "Neutral")
                    with c_emo[0]:
                        _field_header("emoCat", "🎭 Emotion Category")
                        emo_opts = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
                        final_emotion = st.selectbox(
                            "Select Emotion:",
                            options=emotion_options,
                            index=emotion_options.index(current_emotion) if current_emotion in emotion_options else emotion_options.index("Neutral"),
                            key=f"emo_select_{idx}",
                            on_change=check_dirty_callback
                        )
                else:
                    final_emotion = None
                # new_arousal, new_valence, new_dominance = np.nan, np.nan, np.nan
                if any(x in md_cols for x in ["arousal", "valence", "dominance"]) and "arousal__valence__dominance" not in st.session_state.hidden_fields:
                    with c_emo[1]:
                        _field_header(["arousal", "valence", "dominance"], "📈 Arousal", show_button=False)
                        final_arousal = st.slider("Arousal Scaling", 0.0, 1.0, float(_resolve_value("arousal", row_data, 0.5)), key=f"arousal_input_{idx}")
                    with c_emo[2]:
                        _field_header(["arousal", "valence", "dominance"], "📉 Valence", show_button=False)
                        final_valence = st.slider("Valence Scaling", 0.0, 1.0, float(_resolve_value("valence", row_data,  0.5)), key=f"valence_input_{idx}")
                    with c_emo[3]:
                        _field_header(["arousal", "valence", "dominance"], "👑 Dominance")
                        final_dominance = st.slider("Dominance Scaling", 0.0, 1.0, float(_resolve_value("dominance", row_data,  0.5)), key=f"dom_input_{idx}")
                else:
                    final_arousal, final_valence, final_dominance = None, None, None
                st.write("")
                submit_save = st.button("💾 Save", key=f"save_btn_{idx}", use_container_width=True)

                st.session_state.current_state = {
                    "start_sec": selected_start,
                    "end_sec": selected_end,
                    "transcription": new_text,
                    "speaker": final_speaker,
                    "type": final_type,
                    "sex": final_sex,
                    "age": final_age,
                    "emoCat": final_emotion,
                    "arousal": final_arousal,
                    "valence": final_valence,
                    "dominance": final_dominance,
                }

        if submit_save:
            df.at[idx, "start_sec"] = round(selected_start, 2)
            df.at[idx, "end_sec"] = round(selected_end, 2)
            df.at[idx, "transcription"] = new_text
            df.at[idx, "speaker"] = final_speaker
            df.at[idx, "type"] = final_type
            df.at[idx, "sex"] = final_sex if "sex" not in st.session_state.hidden_fields and "sex" in st.session_state.initialized_fields else np.nan
            df.at[idx, "age"] = final_age if "age" not in st.session_state.hidden_fields and "age" in st.session_state.initialized_fields else np.nan
            df.at[idx, "emoCat"] = final_emotion if "emoCat" not in st.session_state.hidden_fields and "emoCat" in st.session_state.initialized_fields else np.nan
            if "arousal__valence__dominance" not in st.session_state.hidden_fields and "arousal__valence__dominance" in st.session_state.initialized_fields:
                df.at[idx, "arousal"] = round(final_arousal, 2)
                df.at[idx, "valence"] = round(final_valence, 2)
                df.at[idx, "dominance"] = round(final_dominance, 2)
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
                    md_cols = st.session_state.get("md_cols", [])

                    if st.session_state.current_state is None:
                        st.session_state.current_state = {}
                    if field == "arousal__valence__dominance":
                        for k in ["arousal", "valence", "dominance"]:
                            if k not in md_cols:
                                continue
                            st.session_state.current_state.pop(k, None)
                    else:
                        if field not in md_cols:
                            continue
                        st.session_state.current_state.pop(field, None)
                    st.session_state.hidden_fields.remove(field)
                    check_dirty_callback()
                    st.rerun()
    else:
        st.info("👈 Please select an audio segment tracking slice from the sidebar queue to start.")


# -----------------------------
# CORE APP ENGINE STATE INITIALIZATION
# -----------------------------

st.set_page_config(page_title="Visual Timeline Audio Trimmer", initial_sidebar_state="expanded", layout="wide")
st.title("✂️ Visual Audio Range Trimmer & Editor")
defaults = {
    "selected_idx": None, "last_selected_idx": None, "pending_idx": None, "show_discard": False, "current_state": {},
    "hidden_fields": set(), "initialized_fields": set(), "init_hidden_fields": set(), "df_segments": None}
for key, default in defaults.items():
    st.session_state.setdefault(key, default)
if st.session_state["df_segments"] is None:
    if st.session_state.get("result"):
        st.session_state["df_segments"] = load_data()
    else:
        run_upload() 
        st.stop()

run_annotation()

    
    # st.write("Choose a folder that contain transcption and metadata")
    
