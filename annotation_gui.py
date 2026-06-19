import io
import wave
import os
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
# Header
# -----------------------------
if "hidden_fields" not in st.session_state:
    st.session_state.hidden_fields = set()
def field_header(field_name, title):
    col1, col2 = st.columns([6, 1])

    with col1:
        st.markdown(f"##### {title}")

    with col2:
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            if st.button("✖", key=f"hide_{field_name}", type="tertiary"):
                st.session_state.hidden_fields.add(field_name)
                st.rerun()

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


def load_snapshot(idx, row):
    return {
        "transcription": row["transcription"],
        "speaker": row.get("speaker", ""),
        "type": row.get("type", ""),
        "sex": row.get("sex", ""),
        "age": row.get("age", 25),
        "emotion": row.get("emoCat", "Neutral"),
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


df = st.session_state.df_segments


# -----------------------------
# SIDEBAR (YOUR STYLE KEPT)
# -----------------------------
st.sidebar.header("⏳ Audio Slices Queue")

def switch_segment(new_idx):
    current = st.session_state.selected_idx
    if current == new_idx:
        return
    if current is not None:
        dirty = st.session_state.get(f"dirty_{current}", False)
    else: 
        dirty = False

    if dirty:
        st.session_state.pending_idx = new_idx
        st.session_state.show_discard = True
    else:
        st.session_state.selected_idx = new_idx
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
# DISCARD POPUP (STREAMLIT DIALOG)
# -----------------------------
if st.session_state.show_discard:

    @st.dialog("⚠️ Unsaved Changes")
    def dialog():
        st.write("You have unsaved changes. Do you want to discard them?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Discard"):
                st.session_state.selected_idx = st.session_state.pending_idx
                st.session_state.show_discard = False
                st.session_state.pending_idx = None
                st.session_state.current_state = None
                st.rerun()

        with col2:
            if st.button("Cancel"):
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

    # snap_key = f"snapshot_{idx}"

    # if st.session_state.get("last_selected_idx") != idx:
    #     st.session_state[snap_key] = load_snapshot(idx, row_data)
    #     st.session_state["last_selected_idx"] = idx
    # -----------------------------
    # HEADER (YOUR STYLE KEPT)
    # -----------------------------
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

    # -----------------------------
    # TOTAL LENGTH
    # -----------------------------
    with wave.open(origin_rel_path, "rb") as w:
        total_file_seconds = w.getnframes() / w.getframerate()

    st.write("---")
    st.markdown("### 🎚️ Drag & Cut Audio Boundaries")

    slider_key = f"slider_{idx}"
    dirty_key = f"dirty_{idx}"
    default_range = (float(row_data["start_sec"]), float(row_data["end_sec"]))

    # -----------------------------
    # INIT STATE
    # -----------------------------
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_range

    if dirty_key not in st.session_state:
        st.session_state[dirty_key] = False

    # reset when switching segment
    if st.session_state.last_selected_idx != idx:
        st.session_state[slider_key] = default_range
        st.session_state[dirty_key] = False
        st.session_state.last_selected_idx = idx

    # -----------------------------
    # RESET FUNCTION
    # -----------------------------
    def reset_slider():
        st.session_state[slider_key] = default_range
        st.session_state[dirty_key] = False

    # -----------------------------
    # SLIDER (YOUR ORIGINAL STYLE)
    # -----------------------------
    col_slider, col_refresh = st.columns([5, 1])

    with col_slider:
        time_range = st.slider(
            "Isolate Audio Playback Area (Left handle = start boundary | Right handle = end boundary)",
            min_value=0.0,
            max_value=float(total_file_seconds),
            value=st.session_state[slider_key],
            step=0.05,
            format="%.2f seconds",
            key=slider_key
        )


    with col_refresh:
        st.write("")
        st.button("🔄 Reset Value", on_click=reset_slider)

    selected_start, selected_end = time_range
    selected_duration = selected_end - selected_start

    # -----------------------------
    # AUDIO PREVIEW
    # -----------------------------
    audio = slice_wav_bytes(origin_rel_path, selected_start, selected_end)

    if audio:
        st.audio(audio, format="audio/wav")
        st.caption(f"🎵 Playing an isolated audio chunk. Length: {selected_duration:.2f}s")

    # -----------------------------
    # FORM (YOUR ORIGINAL LAYOUT UNCHANGED)
    # -----------------------------
    
    edit_container = st.container(border=True)

    with edit_container:
        with st.expander("📝 Edit Transcription Notes", expanded=True):

            new_text = st.text_area(
                "Transcription text content:",
                value=str(row_data["transcription"]),
                height=max(100, 50 * int(len(str(row_data["transcription"])) / 100))
            )


        with st.expander("Characteristics", expanded=True):
            cols = st.columns(2)

            with cols[0]:
                st.markdown("##### 👥 Speaker Attribution")

                existing_speakers = sorted(list(df["speaker"].dropna().unique()))
                speaker_options = existing_speakers + ["➕ Add New Speaker..."]

                current_speaker = str(row_data.get("speaker", ""))
                default_spk_idx = existing_speakers.index(current_speaker) if current_speaker in existing_speakers else 0

                selected_speaker_choice = st.selectbox(
                    "Select Speaker identity:",
                    speaker_options,
                    index=default_spk_idx,
                    key=f"spk_select_{idx}"
                )

                final_speaker = selected_speaker_choice

            with cols[1]:
                st.markdown("##### 🏷️ Interaction Type Classification")

                existing_types = sorted(list(df["type"].dropna().unique()))
                type_options = existing_types + ["➕ Add New Type..."]

                current_type = str(row_data.get("type", ""))
                default_type_idx = existing_types.index(current_type) if current_type in existing_types else 0

                selected_type_choice = st.selectbox(
                    "Select Interaction Type:",
                    type_options,
                    index=default_type_idx,
                    key=f"type_select_{idx}"
                )

                final_type = selected_type_choice

            cols = st.columns(2)

            if "sex" not in st.session_state.hidden_fields:
                with cols[0]:
                    st.markdown("##### 🧑Sex")
                    sex_options = ["Female", "Male", "Other"]
                    
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('sex'):
                        current_sex = st.session_state.current_state["sex"]
                    else:
                        current_sex = str(row_data.get("sex", ""))
                    default_sex_idx = sex_options.index(current_sex) if current_sex in sex_options else 0
                    final_sex = st.selectbox(
                        "Select Sex:",
                        sex_options,
                        index=default_sex_idx,
                        key=f"sex_select_{idx}"
                    )
            else:
                final_sex = np.nan 

            if "age" not in st.session_state.hidden_fields:
                with cols[1]:
                    st.markdown("##### ⏳Age")

                    if st.session_state.get("current_state") and st.session_state["current_state"].get('age'):
                        current_age = st.session_state.current_state["age"]
                    else:
                        current_age = int(row_data.get("age", 25))
                    final_age = st.number_input(
                        "Enter Age:",
                        0, 120,
                        value=current_age,
                        key=f"age_input_{idx}"
                    )
            else:
                final_age = np.nan 
            
            cols = st.columns(4)
            if "emotion" not in st.session_state.hidden_fields:
                with cols[0]:
                    field_header("emotion", "🎭 Emotion")
                    emotion_options = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise','Other']
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('emotion'):
                        current_emotion = st.session_state.current_state["emotion"]
                    else:
                        current_emotion = str(row_data.get("emoCat", "Neutral"))
                    default_emotion_idx = emotion_options.index(current_emotion) if current_emotion in emotion_options else 0
                    final_emotion = st.selectbox(
                        "Select Emotion:",
                        options=emotion_options,
                        index= default_emotion_idx
                    )
            else:
                final_emotion = np.nan 
            
            if "arousal" not in st.session_state.hidden_fields:
                with cols[1]:
                    field_header("arousal", "📈 Arousal")
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('arousal'):
                        current_arousal = st.session_state.current_state["arousal"]
                    else:
                        current_arousal = float(row_data.get("arousal", 0.5))
                    final_arousal = st.slider("Arousal Level:", 0.0, 1.0, current_arousal)
            else:
                final_arousal = np.nan 
            
            if "valence" not in st.session_state.hidden_fields:
                with cols[2]:
                    field_header("valence", "📉 Valence")
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('valence'):
                        current_valence = st.session_state.current_state["valence"]
                    else:
                        current_valence = float(row_data.get("valence", 0.5))
                    final_valence = st.slider("Valence Level:", 0.0, 1.0, current_valence)
            else:
                final_valence = np.nan 

            if "dominance" not in st.session_state.hidden_fields:
                with cols[3]:
                    field_header("dominance", "👑 Dominance")
                    if st.session_state.get("current_state") and st.session_state["current_state"].get('dominance'):
                        current_dominance = st.session_state.current_state["dominance"]
                    else:
                        current_dominance = float(row_data.get("dominance", 0.5))
                    final_dominance = st.slider("Dominance Level:", 0.0, 1.0, current_dominance)
            else:
                final_dominance = np.nan 

        st.session_state.current_state = {
            "transcription": new_text,
            "speaker": final_speaker,
            "type": final_type,
            "sex": final_sex,
            "age": final_age,
            "emotion": final_emotion,
            "arousal": final_arousal,
            "valence": final_valence,
            "dominance": final_dominance,
        }
        
        origin_state = load_snapshot(idx, row_data)
        st.session_state[dirty_key] = ((origin_state != st.session_state.current_state) or (time_range != default_range))

        st.sidebar.markdown("---")
        st.sidebar.subheader("Hidden Fields")

        for field in sorted(st.session_state.hidden_fields):
            if st.sidebar.button(
                f"Restore {field}",
                key=f"restore_{field}"
            ):
                st.session_state.hidden_fields.remove(field)
                st.session_state.current_state[field] = None
                st.session_state[dirty_key]=False
                st.rerun()


        if st.button("💾 Save", key=f"save_{idx}"):
            print(final_emotion)
            df.at[idx, "start_sec"] = round(selected_start, 2)
            df.at[idx, "end_sec"] = round(selected_end, 2)
            df.at[idx, "transcription"] = new_text

            df.at[idx, "sex"] = final_sex
            df.at[idx, "age"] = final_age
            df.at[idx, "emotion"] = final_emotion
            df.at[idx, "arousal"] = round(final_arousal, 2)
            df.at[idx, "valence"] = round(final_valence, 2)
            df.at[idx, "dominance"] = round(final_dominance, 2)

            df.to_csv(UPDATED_METADATA_FILE, index=False)

            st.session_state[dirty_key] = False
            st.success("Saved!")
            st.rerun()

else:
    st.info("👈 Please select a segment from the sidebar")