import io
import wave
import os
import time
import pandas as pd
import streamlit as st
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
METADATA_FILE = "outputs/dyad/raw_agesex.txt"
UPDATED_METADATA_FILE = "outputs/dyad/updated_labels.txt"

st.set_page_config(page_title="Visual Timeline Audio Trimmer", layout="wide")
st.title("✂️ Visual Audio Range Trimmer & Editor")


# -----------------------------
# AUDIO SLICER
# -----------------------------
def slice_wav_bytes(file_path, start_sec, end_sec):
    if not os.path.exists(file_path):
        return None

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


# -----------------------------
# HEADER & HIDDEN FIELDS MANAGEMENT
# -----------------------------
if "hidden_fields" not in st.session_state:
    st.session_state.hidden_fields = set()

def field_header(field_name, title):
    col1, col2 = st.columns([6, 1])

    # ALWAYS convert to a stable string key
    if isinstance(field_name, list):
        field_key = "__".join(field_name)
    else:
        field_key = str(field_name)

    with col1:
        st.markdown(f"##### {title}")

    with col2:
        if st.button("✖", key=f"hide_{field_key}", type="tertiary"):
            st.session_state.hidden_fields.add(field_key)
            st.rerun()


# -----------------------------
# INTEGRATED DIRTY CHECK CALLBACK
# -----------------------------
def check_dirty_callback():
    """
    Evaluates changes instantaneously whenever a widget value changes,
    comparing current state values safely against baseline row data.
    """
    idx = st.session_state.selected_idx
    if idx is None or df is None:
        return

    row_data = df.iloc[idx]
    
    # Extract data securely from state or fallbacks
    w_range = st.session_state.get(f"slider_{idx}", (float(row_data["start_sec"]), float(row_data["end_sec"])))
    w_text = st.session_state.get(f"transc_{idx}", str(row_data["transcription"])).strip()
    w_spk = st.session_state.get(f"spk_select_{idx}", str(row_data.get("speaker", ""))).strip()
    w_type = st.session_state.get(f"type_select_{idx}", str(row_data.get("type", ""))).strip()
    w_sex = st.session_state.get(f"sex_select_{idx}", str(row_data.get("sex", "")))
    w_age = st.session_state.get(f"age_input_{idx}", int(row_data.get("age", 25)))
    w_emo = st.session_state.get(f"emo_select_{idx}", str(row_data.get("emoCat", "Neutral")))
    w_arousal = st.session_state.get(f"arousal_input_{idx}", float(row_data.get("arousal", 0.5)))
    w_valence = st.session_state.get(f"valence_input_{idx}", float(row_data.get("valence", 0.5)))
    w_dom = st.session_state.get(f"dom_input_{idx}", float(row_data.get("dominance", 0.5)))

    is_dirty = (
        (abs(w_range[0] - float(row_data["start_sec"])) > 0.01) or
        (abs(w_range[1] - float(row_data["end_sec"])) > 0.01) or
        (w_text != str(row_data["transcription"]).strip()) or
        (w_spk != str(row_data.get("speaker", "")).strip()) or
        (w_type != str(row_data.get("type", "")).strip()) or
        ("sex" not in st.session_state.hidden_fields and w_sex != str(row_data.get("sex", ""))) or
        ("age" not in st.session_state.hidden_fields and int(w_age) != (int(row_data["age"]) if pd.notna(row_data.get("age")) else 25)) or
        ("emoCat" not in st.session_state.hidden_fields and w_emo != str(row_data.get("emoCat", "Neutral"))) or
        ("emotional_state" not in st.session_state.hidden_fields and abs(w_arousal - float(row_data.get("arousal", 0.5))) > 0.01) or
        ("emotional_state" not in st.session_state.hidden_fields and abs(w_valence - float(row_data.get("valence", 0.5))) > 0.01) or
        ("emotional_state" not in st.session_state.hidden_fields and abs(w_dom - float(row_data.get("dominance", 0.5))) > 0.01)
    )
    st.session_state[f"dirty_{idx}"] = is_dirty


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    target_file = UPDATED_METADATA_FILE if os.path.exists(UPDATED_METADATA_FILE) else METADATA_FILE

    if not os.path.exists(target_file):
        st.error(f"Metadata file not found: {target_file}")
        return None

    df = pd.read_csv(target_file, sep="\t")
    if len(df.columns) <= 1:
        df = pd.read_csv(target_file, sep=",")

    df.columns = df.columns.str.strip()
    return df


def load_snapshot(row):
    return {
        "transcription": row["transcription"],
        "speaker": row.get("speaker", ""),
        "type": row.get("type", ""),
        "sex": row.get("sex", ""),
        "age": row.get("age", 25),
        "emoCat": row.get("emoCat", "Neutral"),
        "arousal": row.get("arousal", 0.5),
        "valence": row.get("valence", 0.5),
        "dominance": row.get("dominance", 0.5),
    }


# -----------------------------
# SESSION INIT
# -----------------------------
if "df_segments" not in st.session_state:
    st.session_state.df_segments = load_data()

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

if "last_selected_idx" not in st.session_state:
    st.session_state.last_selected_idx = None

if "pending_idx" not in st.session_state:
    st.session_state.pending_idx = None

if "show_discard" not in st.session_state:
    st.session_state.show_discard = False

if "current_state" not in st.session_state:
    st.session_state.current_state = None

df = st.session_state.df_segments


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⏳ Audio Slices Queue")

def switch_segment(new_idx):
    current = st.session_state.selected_idx
    if current == new_idx:
        return
    
    dirty = st.session_state.get(f"dirty_{current}", False) if current is not None else False

    if dirty:
        st.session_state.pending_idx = new_idx
        st.session_state.show_discard = True
    else:
        st.session_state.selected_idx = new_idx
        st.session_state.hidden_fields = set()
        st.session_state.current_state = None
        st.rerun()


if df is not None:
    for idx, row in df.iterrows():
        speaker = row.get("speaker", "??")
        start = row.get("start_sec", 0.0)
        end = row.get("end_sec", 0.0)
        label = f"[{start:.2f}s - {end:.2f}s] {speaker}"

        if st.sidebar.button(label, key=f"seg_{idx}", use_container_width=True):
            switch_segment(idx)


# -----------------------------
# DISCARD POPUP DIALOG
# -----------------------------
if st.session_state.show_discard:
    @st.dialog("⚠️ Unsaved Changes")
    def dialog():
        st.write("You have unsaved changes. Do you want to discard them?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Discard"):
                old_idx = st.session_state.selected_idx
                # Wipe temporary working memory configurations 
                keys_to_clear = [
                    f"slider_{old_idx}", f"transc_{old_idx}", f"spk_select_{old_idx}",
                    f"type_select_{old_idx}", f"sex_select_{old_idx}", f"age_input_{old_idx}",
                    f"emo_select_{old_idx}", f"arousal_input_{old_idx}", f"valence_input_{old_idx}",
                    f"dom_input_{old_idx}"
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                st.session_state[f"dirty_{old_idx}"] = False
                st.session_state.selected_idx = st.session_state.pending_idx
                st.session_state.show_discard = False
                st.session_state.pending_idx = None
                st.session_state.hidden_fields = set()
                st.session_state.current_state = None
                st.rerun()

        with col2:
            if st.button("Cancel"):
                # Simply exit window without modifying or purging values in current_state
                st.session_state.pending_idx = None
                st.session_state.show_discard = False
                st.rerun()

    dialog()
    st.stop()


# -----------------------------
# MAIN VIEW
# -----------------------------
if df is not None and st.session_state.selected_idx is not None:
    idx = st.session_state.selected_idx
    row_data = df.iloc[idx]
    origin_rel_path = row_data["origin_filename"]

    st.markdown(
        f"<p style='padding-top: 32px; color: gray; margin: 0; font-size: 14px;'>"
        f"📁 <b>Source:</b> <code>{origin_rel_path}</code> &nbsp;|&nbsp; "
        f"💾 <b>Save:</b> <code>{row_data['seg_filename']}</code>"
        f"</p>",
        unsafe_allow_html=True
    )

    if not os.path.exists(origin_rel_path):
        st.error("Audio file missing")
        st.stop()

    with wave.open(origin_rel_path, "rb") as w:
        total_file_seconds = w.getnframes() / w.getframerate()

    st.write("---")
    st.markdown("### 🎚️ Drag & Cut Audio Boundaries")

    slider_key = f"slider_{idx}"
    dirty_key = f"dirty_{idx}"
    default_range = (float(row_data["start_sec"]), float(row_data["end_sec"]))

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_range

    if dirty_key not in st.session_state:
        st.session_state[dirty_key] = False

    if st.session_state.last_selected_idx != idx:
        st.session_state[slider_key] = default_range
        st.session_state[dirty_key] = False
        st.session_state.last_selected_idx = idx

    def reset_slider():
        st.session_state[slider_key] = default_range
        st.session_state[dirty_key] = False

    col_slider, col_refresh = st.columns([5, 1])

    with col_slider:
        time_range = st.slider(
            "Isolate Audio Playback Area",
            min_value=0.0,
            max_value=float(total_file_seconds),
            value=st.session_state[slider_key],
            step=0.05,
            format="%.2f seconds",
            key=slider_key,
            on_change=check_dirty_callback
        )

    with col_refresh:
        st.write("")
        st.button("🔄 Reset Value", on_click=reset_slider)

    selected_start, selected_end = time_range
    audio = slice_wav_bytes(origin_rel_path, selected_start, selected_end)

    if audio:
        st.audio(audio, format="audio/wav")
        st.caption(f"🎵 Playing an isolated audio chunk. Length: {(selected_end - selected_start):.2f}s")

    # -----------------------------
    # FORM BLOCK CONTEXT ENGINE
    # -----------------------------
    edit_container = st.container(border=True)

    with edit_container:
        with st.expander("📝 Edit Transcription Notes", expanded=True):          
            if st.session_state.get("current_state") and st.session_state["current_state"].get('transcription') is not None:
                current_text = str(st.session_state.current_state["transcription"])
            else:
                current_text = str(row_data["transcription"])
                
            new_text = st.text_area(
                "Transcription text content:",
                value=current_text,
                key=f"transc_{idx}",
                on_change=check_dirty_callback,
                height=max(100, 50 * int(len(current_text) / 100))
            )

        with st.expander("Characteristics", expanded=True):
            cols = st.columns(2)

            with cols[0]:
                st.markdown("##### 👥 Speaker Attribution")
                existing_speakers = sorted(list(df["speaker"].dropna().unique()))
                speaker_options = existing_speakers + ["➕ Add New Speaker..."]

                if st.session_state.get("current_state") and st.session_state["current_state"].get('speaker') is not None:
                    current_speaker = str(st.session_state.current_state["speaker"])
                else:
                    current_speaker = str(row_data.get("speaker", ""))
                
                default_spk_idx = existing_speakers.index(current_speaker) if current_speaker in existing_speakers else 0
                final_speaker = st.selectbox(
                    "Select Speaker identity:",
                    speaker_options,
                    index=default_spk_idx,
                    key=f"spk_select_{idx}",
                    on_change=check_dirty_callback
                )

            with cols[1]:
                st.markdown("##### 🏷️ Interaction Type Classification")
                existing_types = sorted(list(df["type"].dropna().unique()))
                type_options = existing_types + ["➕ Add New Type..."]
                
                if st.session_state.get("current_state") and st.session_state["current_state"].get('type') is not None:
                    current_type = str(st.session_state.current_state["type"])
                else:
                    current_type = str(row_data.get("type", ""))
                    
                default_type_idx = existing_types.index(current_type) if current_type in existing_types else 0
                final_type = st.selectbox(
                    "Select Interaction Type:",
                    type_options,
                    index=default_type_idx,
                    key=f"type_select_{idx}",
                    on_change=check_dirty_callback
                )

            cols = st.columns(2)

            if "sex" not in st.session_state.hidden_fields:
                with cols[0]:
                    st.markdown("##### 🧑Sex")
                    sex_options = ["Female", "Male", "Other"]
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('sex') is not None:
                        current_sex = st.session_state.current_state["sex"]
                    else:
                        current_sex = str(row_data.get("sex", ""))
                    default_sex_idx = sex_options.index(current_sex) if current_sex in sex_options else 0
                    final_sex = st.selectbox(
                        "Select Sex:",
                        sex_options,
                        index=default_sex_idx,
                        key=f"sex_select_{idx}",
                        on_change=check_dirty_callback
                    )
            else:
                final_sex = np.nan 

            if "age" not in st.session_state.hidden_fields:
                with cols[1]:
                    st.markdown("##### ⏳Age")
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('age') is not None:
                        current_age = int(st.session_state.current_state["age"])
                    else:
                        current_age = int(row_data.get("age", 25))
                    final_age = st.number_input(
                        "Enter Age:",
                        0, 120,
                        value=current_age,
                        key=f"age_input_{idx}",
                        on_change=check_dirty_callback
                    )
            else:
                final_age = np.nan 
            
            cols = st.columns(4)
            current_emotion = (
                st.session_state.current_state["emoCat"]
                if st.session_state.get("current_state") and st.session_state["current_state"].get("emoCat") is not None
                else row_data.get("emoCat", "Neutral")
            )

            if "emoCat" not in st.session_state.hidden_fields:
                with cols[0]:
                    field_header("emoCat", "🎭 Emotion")
                    emotion_options = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise','Other']
                    default_emotion_idx = emotion_options.index(current_emotion) if current_emotion in emotion_options else 0
                    final_emotion = st.selectbox(
                        "Select Emotion:",
                        options=emotion_options,
                        index=default_emotion_idx,
                        key=f"emo_select_{idx}",
                        on_change=check_dirty_callback
                    )
            else:
                final_emotion = current_emotion
            
            current_arousal = (
                float(st.session_state.current_state["arousal"])
                if st.session_state.get("current_state") and st.session_state["current_state"].get("arousal") is not None
                else float(row_data.get("arousal", 0.5))
            )
            current_valence = (
                float(st.session_state.current_state["valence"])
                if st.session_state.get("current_state") and st.session_state["current_state"].get("valence") is not None
                else float(row_data.get("valence", 0.5))
            )
            current_dominance = (
                float(st.session_state.current_state["dominance"])
                if st.session_state.get("current_state") and st.session_state["current_state"].get("dominance") is not None
                else float(row_data.get("dominance", 0.5))
            )
            if "arousal__valence__dominance" not in st.session_state.hidden_fields:
                with cols[1]:
                    st.markdown("##### 📈 Arousal")
                    final_arousal = st.slider(
                        "Arousal Level:", 0.0, 1.0,
                        value=current_arousal,
                        key=f"arousal_input_{idx}",
                        on_change=check_dirty_callback
                    )
                with cols[2]:
                    st.markdown("##### 📉 Valence")
                    final_valence = st.slider(
                        "Valence Level:", 0.0, 1.0,
                        value=float(current_valence),
                        key=f"valence_input_{idx}",
                        on_change=check_dirty_callback
                    )
                with cols[3]:
                    field_header(["arousal", "valence", "dominance"], "👑 Dominance")
                    final_dominance = st.slider(
                        "Dominance Level:", 0.0, 1.0,
                        value=float(current_dominance),
                        key=f"dom_input_{idx}",
                        on_change=check_dirty_callback
                    )
            else:
                final_arousal = current_arousal 
                final_valence = current_valence
                final_dominance = current_dominance

        # Update running in-memory object safely
        st.session_state.current_state = {
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
        
        origin_state = load_snapshot(row_data)
        st.session_state[dirty_key] = ((origin_state != st.session_state.current_state) or (time_range != default_range))

        st.sidebar.markdown("---")
        st.sidebar.subheader("Hidden Fields")

        for field in sorted(st.session_state.hidden_fields):
            if st.sidebar.button(f"Restore {field}", key=f"restore_{field}"):
                st.session_state.hidden_fields.remove(field)
                if st.session_state.current_state and field in st.session_state.current_state:
                    st.session_state.current_state[field] = None
                st.session_state[dirty_key] = False
                st.rerun()

        # -----------------------------
        # SAVE ACTION
        # -----------------------------
        if st.button("💾 Save", key=f"save_{idx}"):
            df.at[idx, "start_sec"] = round(selected_start, 2)
            df.at[idx, "end_sec"] = round(selected_end, 2)
            df.at[idx, "transcription"] = new_text
            df.at[idx, "speaker"] = final_speaker
            df.at[idx, "type"] = final_type
            df.at[idx, "sex"] = final_sex
            df.at[idx, "age"] = final_age
            df.at[idx, "emoCat"] = final_emotion
            if "arousal__valence__dominance" not in st.session_state.hidden_fields:
                df.at[idx, "arousal"] = round(final_arousal, 2)
                df.at[idx, "valence"] = round(final_valence, 2)
                df.at[idx, "dominance"] = round(final_dominance, 2)

            try:
                with open(UPDATED_METADATA_FILE, "w", encoding="utf-8", newline="") as f:
                    csv_data = df.to_csv(index=False, sep="\t")
                    f.write(csv_data)
                    f.flush()        
                    os.fsync(f.fileno()) 
            except Exception as e:
                st.error(f"Critical writing exception encountered: {e}")
                st.stop()
                
            st.session_state[dirty_key] = False
            st.session_state.df_segments = load_data()
            st.success("Saved!")
            time.sleep(1)
            st.rerun()
else:
    st.info("👈 Please select a segment from the sidebar")