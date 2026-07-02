import io
import wave
import os
import time
import pandas as pd
import streamlit as st
import numpy as np
from page.setup import merge_datasets, normalize_schema, _next_available_name

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
    rename_dict = st.session_state["project"].get("mapping", {})
    return df.rename(columns=rename_dict)
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
    project["tf_cols"] = list(df_ts.columns) if df_ts is not None else []
    project["md_cols"] = list(df_md.columns) if df_md is not None else []
    if "seg_filename" not in df_ts.columns or "seg_filename" not in df_md.columns:
        st.error("Missing 'seg_filename' column.")
        return None
    
    df_ts = normalize_schema(df_ts)
    df_md = normalize_schema(df_md)
    df = merge_datasets(df_ts, df_md)
    project["df"] = df
    project["mode"] = "annotation"

def save_data(df: pd.DataFrame):
    project = st.session_state.get("project", {})
    editor = st.session_state.get("editor", {})

    mapping = project.get("mapping", {})
    tf_mapping = mapping.get("tf", {})
    md_mapping = mapping.get("md", {})

    tf_cols = project.get("tf_cols", [])
    md_cols = project.get("md_cols", [])

    files = project.get("files", {})
    transcript_path = files.get("update_transcript")
    metadata_path = files.get("update_meta")

    if not transcript_path or not metadata_path:
        raise ValueError("Missing output file paths")

    # -------------------------
    # TRANSCRIPT EXPORT
    # -------------------------
    df_tf = df.copy()

    if tf_mapping:
        df_tf = df_tf.rename(columns={v: k for k, v in tf_mapping.items()})

    ordered_tf_cols = [c for c in tf_cols if c in df_tf.columns]

    # fallback: if nothing matches, keep full df order
    if not ordered_tf_cols:
        ordered_tf_cols = df_tf.columns.tolist()

    df_tf[ordered_tf_cols].to_csv(transcript_path, sep="\t", index=False)

    # -------------------------
    # METADATA EXPORT
    # -------------------------
    df_md = df.copy()

    if md_mapping:
        df_md = df_md.rename(columns={v: k for k, v in md_mapping.items()})

    ordered_md_cols = [c for c in md_cols if c in df_md.columns]

    if not ordered_md_cols:
        ordered_md_cols = df_md.columns.tolist()

    df_md[ordered_md_cols].to_csv(metadata_path, sep="\t", index=False)

    # -------------------------
    # RESET STATE
    # -------------------------
    editor["dirty"] = False
    st.session_state["editor"] = editor

# -----------------------------
# DATA ENGINE RESOLUTION HELPERS
# -----------------------------
def _resolve_value(field, row_data, default=None):
    
    if field in st.session_state.editor.get("current_state", {}):
        return st.session_state.editor["current_state"][field]

    value = row_data.get(field)

    if pd.isna(value):
        return default

    return value

def build_optional_fields(md_cols: list) -> dict:
    """Builds visibility lookup matrix maps using source file constraints."""
    optional = {}
    if "age" in md_cols: optional["age"] = ["age"]
    if "emoCat" in md_cols: optional["emoCat"] = ["emoCat"]
    if "sex" in md_cols: optional["sex"] = ["sex"]
    avd = [c for c in ["arousal", "valence", "dominance"] if c in md_cols]
    if avd: optional["arousal__valence__dominance"] = avd
    return optional

# -----------------------------
# INTEGRATED DIRTY CHECK CALLBACK
# -----------------------------
def _changed(field, current, original):
    return (
        
        field in st.session_state.editor["initialized_fields"]
        and field not in st.session_state.editor["hidden_fields"]
        and current != original
    )

def _changed_float(field, current, original, tol=0.01):
    return (
        field in st.session_state.editor["initialized_fields"]
        and field not in st.session_state.editor["hidden_fields"]
        and abs(float(current) - float(original)) > tol
    )
    
def check_dirty_callback():
    project = st.session_state.get("project", {})
    editor = st.session_state.get("editor", {})

    idx = editor.get("selected_idx")
    df = project.get("df", None)

    if idx is None or df is None:
        return
    row_data = df.iloc[idx]

    # -----------------------------
    # SAFE widget reads
    # -----------------------------
    w_range = st.session_state.get(f"slider_{idx}", (float(row_data["start_sec"]), float(row_data["end_sec"])))
    w_text = st.session_state.get(f"text_input_{idx}", str(row_data["transcription"])).strip()
    w_spk = st.session_state.get(f"spk_select_{idx}", str(row_data.get("speaker", ""))).strip()
    w_type = st.session_state.get(f"type_select_{idx}", str(row_data.get("type", ""))).strip()
    w_sex = st.session_state.get(f"sex_select_{idx}", _resolve_value("sex",row_data, ""))
    w_age = st.session_state.get(f"age_input_{idx}",_resolve_value("age",row_data, 25))
    w_emo = st.session_state.get(f"emo_select_{idx}",_resolve_value("emoCat",row_data, "Neutral"))
    w_arousal = st.session_state.get(f"arousal_input_{idx}", _resolve_value("arousal", row_data, 0.5))
    w_valence = st.session_state.get(f"valence_input_{idx}",_resolve_value("valence", row_data, 0.5))
    w_dom = st.session_state.get(f"dom_input_{idx}",_resolve_value("dominance", row_data, 0.5))
    # -----------------------------
    # DIRTY CHECK
    # -----------------------------
    is_dirty = (
        abs(w_range[0] - float(row_data["start_sec"])) > 0.01
        or abs(w_range[1] - float(row_data["end_sec"])) > 0.01
        or w_text != str(row_data.get("transcription", "")).strip()
        or w_spk != str(row_data.get("speaker", "")).strip()
        or w_type != str(row_data.get("type", "")).strip()
        or _changed("sex", w_sex, _resolve_value("sex", row_data, ""))
        or _changed("age", w_age, _resolve_value("age", row_data, 25))
        or _changed("emoCat", w_emo, _resolve_value("emoCat", row_data, "Neutral"))
        or _changed_float("arousal__valence__dominance", w_arousal, _resolve_value("arousal", row_data, 0.5))
        or _changed_float("arousal__valence__dominance", w_valence, _resolve_value("valence", row_data, 0.5))
        or _changed_float("arousal__valence__dominance", w_dom, _resolve_value("dominance", row_data, 0.5))
        or editor.get("init_hidden_fields", set()) != editor.get("hidden_fields", set())
        # or editor.get("hidden_fields", set()) != set()
    )
    editor["dirty"] = is_dirty
    st.session_state["editor"] = editor

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
            st.session_state["editor"]["hidden_fields"].add(field_key)
            check_dirty_callback()
            st.rerun()
    else:
        col2.html("<div style='height: 40px;'></div>")

def _switch_segment(new_idx: int):
    editor = st.session_state["editor"]

    current = editor.get("selected_idx")

    if current == new_idx:
        return

    if editor.get("dirty", False):
        editor["pending_idx"] = new_idx
        editor["show_discard"] = True
    else:
        editor["selected_idx"] = new_idx
        editor["hidden_fields"] = set()
        editor["initialized_fields"] = set()
        editor["current_state"] = {}

    st.session_state["editor"] = editor
    st.rerun()

def run_annotation_gui():

    project = st.session_state.get("project", {})
    editor = st.session_state.get("editor", {})
    df = project.get("df")
    if df is None:
        st.info("No dataset loaded.")
        return

    # -----------------------------
    # SIDEBAR NAVIGATION
    # -----------------------------
    from pathlib import Path
    with st.sidebar.expander("📁 Saved Annotation Path"):
        files = st.session_state["project"]["files"]
        files["output_dir"] = st.text_input(
            "Output directory",
            value=files["output_dir"],
        )

        if st.session_state["project"].get("tf_cols"):
            files["update_transcript"] =  str(Path(files["output_dir"]) /st.text_input(
                "Transcript filename",
                value=Path(files["update_transcript"]).name,
            ))
        if st.session_state["project"].get("md_cols"):
            files["update_meta"] = str(Path(files["output_dir"]) / st.text_input(
                "Metadata filename",
                value=Path(files["update_meta"]).name,
            ))
        
        
    
    st.sidebar.header("⏳ Audio Slices Queue")

    for idx, row in df.iterrows():
        label = f"[{row.get('start_sec',0):.2f}s - {row.get('end_sec',0):.2f}s] {row.get('speaker','??')}"

        if st.sidebar.button(label, key=f"seg_{idx}"):
            _switch_segment(idx)

    # -----------------------------
    # UNSAVED CHANGES DIALOG
    # -----------------------------
    if editor.get("show_discard"):
        
        @st.dialog("⚠️ Unsaved Changes")
        def discard_dialog():

            col1, col2 = st.columns(2)

            if col1.button("Discard Changes"):
                st.session_state["editor"]["selected_idx"] = editor.get("pending_idx")
                st.session_state["editor"]["pending_idx"] = None
                st.session_state["editor"]["show_discard"] = False
                st.session_state["editor"]["hidden_fields"].clear()
                st.session_state['editor']["dirty"] = False
                st.session_state["editor"]["init_hidden_fields"].clear()
                st.session_state["editor"]["initialized_fields"].clear()
                st.session_state["editor"]["current_state"].clear()

                st.rerun()

            if col2.button("Cancel"):
                st.session_state["editor"]["show_discard"] = False
                st.session_state["editor"]["pending_idx"] = None
                st.rerun()
        discard_dialog()
        st.stop()

    # -----------------------------
    # MAIN WORKSPACE
    # -----------------------------
    idx = editor.get("selected_idx")

    if df is None or idx is None:
        st.info("👈 Select an audio slice from the sidebar")
        return

    row = df.iloc[idx]
    # -----------------------------
    # AUDIO CHECK
    # -----------------------------
    tmp_audio = st.session_state.get("tmp_audio", {})
    if isinstance(tmp_audio, dict):
        audio_path = tmp_audio.get(str(row.get("audio_filename", ""))) or tmp_audio.get(row["speaker"])
    elif isinstance(tmp_audio, str):
        audio_path = tmp_audio
    elif os.path.exists(str(row.get("audio_filename", ""))):
        audio_path = str(row.get("audio_filename", ""))
    else:
        audio_path = None
    if audio_path is None:
        st.info(f"{str(row.get('audio_filename', ''))} not found. Do you want to upload it manually?")

        uploaded_file = st.file_uploader(
            "Upload WAV file",
            type=["wav"],
            key=f"upload_audio_{idx}",
        )

        if uploaded_file is not None:
            import tempfile

            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if "tmp_audio" not in st.session_state or not isinstance(st.session_state["tmp_audio"], dict):
                st.session_state["tmp_audio"] = {}

            if len(df["audio_filename"].dropna().unique()) > 1:
                st.session_state["tmp_audio"][row["audio_filename"]] = temp_path
            else:
                st.session_state["tmp_audio"] = temp_path
            
            st.success("Audio uploaded successfully!")
            audio_path = temp_path
            st.rerun()

        st.stop()
        


    with wave.open(audio_path, "rb") as w:
        total_file_seconds = w.getnframes() / w.getframerate()
    # st.write("---")
    col_audio, col_transcript = st.columns(2)
    with col_audio:
        st.markdown("### 🎚️ Audio Editing")
    with col_transcript:
        final_seg_filename =  str(Path(row["seg_filename"]) / st.text_input(
            "Transcript filename",
            value=Path(row["seg_filename"]).name,
        ))
    # -----------------------------
    # RANGE SLIDER
    # -----------------------------
    slider_key = f"slider_{idx}"
    
    # reset hook 
    if st.session_state.get("_reset_slider"):
        st.session_state[slider_key] = (float(row["start_sec"]), float(row["end_sec"]))
        check_dirty_callback()
        st.session_state["_reset_slider"] = False

    col_slider, col_refresh = st.columns([5, 1])
    if slider_key not in st.session_state:
        st.session_state[slider_key] = (
            float(_resolve_value("start_sec", row, 0.0)),
            float(_resolve_value("end_sec", row, 0.0)),
        )
    with col_slider:
        time_range = st.slider(
            "Isolate Audio Playback Area",
            0.0,
            float(total_file_seconds),
            step=0.05,
            format="%.2f seconds",
            key=slider_key,
            # value=(float(_resolve_value("start_sec", row, 0.0)), float(_resolve_value("end_sec", row, 0.0))),
            on_change=check_dirty_callback
        )
    with col_refresh:
        if st.button("🔄 Reset Trim", width='stretch'):
            st.session_state["_reset_slider"] = True
            st.rerun()
    selected_start, selected_end = time_range
    # Audio Fragment Context Playback
    audio_bytes = slice_wav_bytes(audio_path, selected_start, selected_end )
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.caption(f"🎵 Active selection run: {(selected_end - selected_start):.2f} seconds total duration.")
    # -----------------------------
    # EDITOR WORKSPACE BLOCK
    # -----------------------------
    md_cols = st.session_state["project"]["md_cols"]
    optional_matrix = build_optional_fields(md_cols)
    # Run structural automatic hider logic calculations
    for hide_key, cols in optional_matrix.items():
        if hide_key not in editor['initialized_fields']:
            if all(row.get(c) is None or pd.isna(row.get(c)) for c in cols):
                st.session_state["editor"]['hidden_fields'].add(hide_key)
                st.session_state["editor"]['init_hidden_fields'].add(hide_key)
            st.session_state["editor"]['initialized_fields'].add(hide_key)
    # Replaced st.form with an st.container to fully support the nested header hide buttons
    with st.container(border=True):
        with st.expander("📝 Transcription Notes", expanded=True):
            current_text = _resolve_value("transcription", row, "")
            new_text = st.text_area(
                "Transcription text content:", 
                value=str(current_text ), 
                key=f"text_input_{idx}",
                on_change=check_dirty_callback,
                height=max(100, 50 * int(len(current_text) / 100))
            )

    
    
    # -----------------------------
    # SPEAKER / TYPE
    # -----------------------------
    with st.expander("Characteristics", expanded=True):
        c_att = st.columns(2)
        with c_att[0]:
            st.markdown("##### 👥 Speaker Attribution")

            existing_speakers = sorted(list(df["speaker"].dropna().unique()))
            current_speaker = _resolve_value("speaker", row, "P1")
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
            current_type = _resolve_value("type", row, "")
            default_type_idx = existing_types.index(current_type) if current_type in existing_types else 0
            final_type = st.selectbox(
                "Select Interaction Type:",
                type_options,
                index=default_type_idx,
                key=f"type_select_{idx}",
                on_change=check_dirty_callback
            )
        # -----------------------------
        # DEMOGRAPHICS
        # -----------------------------

        c_dem = st.columns(2)
        if "sex" not in editor['hidden_fields'] and "sex" in md_cols:
            current_sex = _resolve_value("sex", row, "")
            sex_options = ["Female", "Male", "Other"]
            with c_dem[0]:
                _field_header("sex", "🧑 Sex Identification", show_button=False)
                default_sex_idx = sex_options.index(current_sex) if current_sex in sex_options else 0
                final_sex = st.selectbox(
                    "Select Sex:",
                    options=sex_options,
                    index=default_sex_idx,
                    key=f"sex_select_{idx}",
                    on_change=check_dirty_callback)
        else:
            final_sex = None
        if "age" not in editor['hidden_fields'] and "age" in md_cols:
            current_age = _resolve_value("age", row, 25)
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
        # -----------------------------
        # EMOTION
        # -----------------------------

        c_emo = st.columns(4)
        emotion_options = [ "Anger", "Contempt", "Disgust", "Fear","Happiness", "Neutral", "Sadness", "Surprise", "Other"]
        if "emoCat" not in editor['hidden_fields'] and "emoCat" in md_cols:
            current_emotion = _resolve_value("emoCat", row, "Neutral")
            with c_emo[0]:
                _field_header("emoCat", "🎭 Emotion Category")
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
        if any(x in md_cols for x in ["arousal", "valence", "dominance"]) and "arousal__valence__dominance" not in editor['hidden_fields']:
            with c_emo[1]:
                _field_header(["arousal", "valence", "dominance"], "📈 Arousal", show_button=False)
                final_arousal = st.slider("Arousal Scaling", 0.0, 1.0, float(_resolve_value("arousal", row, 0.5)), key=f"arousal_input_{idx}", on_change=check_dirty_callback)
            with c_emo[2]:
                _field_header(["arousal", "valence", "dominance"], "📉 Valence", show_button=False)
                final_valence = st.slider("Valence Scaling", 0.0, 1.0, float(_resolve_value("valence", row,  0.5)), key=f"valence_input_{idx}", on_change=check_dirty_callback)
            with c_emo[3]:
                _field_header(["arousal", "valence", "dominance"], "👑 Dominance")
                final_dominance = st.slider("Dominance Scaling", 0.0, 1.0, float(_resolve_value("dominance", row,  0.5)), key=f"dom_input_{idx}", on_change=check_dirty_callback)
        else:
            final_arousal, final_valence, final_dominance = None, None, None
        st.write("")


    # -----------------------------
    # SAVE STATE SNAPSHOT
    # -----------------------------
    editor["current_state"] = {
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

    # -----------------------------
    # SAVE BUTTON
    # -----------------------------
    if st.button("💾 Save"):
        df.at[idx, "seg_filename"] = final_seg_filename
        df.at[idx, "start_sec"] = round(selected_start, 2)
        df.at[idx, "end_sec"] = round(selected_end, 2)
        df.at[idx, "transcription"] = new_text
        df.at[idx, "speaker"] = final_speaker
        df.at[idx, "type"] = final_type
        df.at[idx, "sex"] = final_sex if "sex" not in editor["hidden_fields"] and "sex" in editor["initialized_fields"] else np.nan
        df.at[idx, "age"] = final_age if "age" not in editor["hidden_fields"] and "age" in editor["initialized_fields"] else np.nan
        df.at[idx, "emoCat"] = final_emotion if "emoCat" not in editor["hidden_fields"] and "emoCat" in editor["initialized_fields"] else np.nan
        if "arousal__valence__dominance" not in editor["hidden_fields"] and "arousal__valence__dominance" in editor["initialized_fields"]:
            df.at[idx, "arousal"] = round(final_arousal, 2)
            df.at[idx, "valence"] = round(final_valence, 2)
            df.at[idx, "dominance"] = round(final_dominance, 2)
        else:
            for field in ["arousal", "valence", "dominance"]:
                df.at[idx, field] = np.nan

        project["df"] = df
        st.session_state["editor"]["dirty"] = False

        try:
            save_data(df)
            # st.session_state[dirty_key] = False
            st.session_state['project']['df']= df
            st.success("Successfully committed workspace changes!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Write operation failure encountered: {e}")


        # st.success("Saved successfully")
        # st.rerun()
    
     # -----------------------------
    # SIDEBAR: RESTORE FIELDS UTILITY
    # -----------------------------
    if editor["hidden_fields"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Hidden Interface Blocks")
        for field in sorted(list(editor["hidden_fields"])):
            if st.sidebar.button(f"↩️ Restore {field.split('__')[0]}", key=f"restore_{field}"):

                if editor["current_state"] is None:
                    st.session_state["editor"]["current_state"] = {}
                if field == "arousal__valence__dominance":
                    for k in ["arousal", "valence", "dominance"]:
                        if k not in md_cols:
                            continue
                        st.session_state["editor"]["current_state"].pop(k, None)
                else:
                    if field not in md_cols:
                        continue
                    st.session_state["editor"]["current_state"].pop(field, None)
                st.session_state["editor"]["hidden_fields"].remove(field)
                check_dirty_callback()
                st.rerun()


# -----------------------------
# CORE APP ENGINE STATE INITIALIZATION
# -----------------------------
def run_annotation() -> None:
    
    st.set_page_config(page_title="Visual Timeline Audio Trimmer", initial_sidebar_state="expanded", layout="wide")
    
    st.title("✂️ Editing Annotation")
    if st.session_state['project'].get("df") is None:
        if st.session_state.get("result"):
            load_data()
    
    run_annotation_gui()
