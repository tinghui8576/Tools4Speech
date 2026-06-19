# import os
# import pandas as pd
# import streamlit as st

# # 1. Configuration - Set your audio folder and annotation save file
# AUDIO_DIR = "outputs/dyad"  # Replace with your folder path
# ANNOTATION_FILE = "annotations.csv"

# st.title("🎵 Audio Annotation Dashboard")
import io
import wave
import os
import pandas as pd
import streamlit as st

# 1. CONFIGURATION
METADATA_FILE = "outputs/dyad/raw_agesex.txt"  # Path to your transcription CSV
UPDATED_METADATA_FILE = "outputs/dyad/updated_labels.txt"     # Where edited results are saved

st.set_page_config(page_title="Visual Timeline Audio Trimmer", layout="wide")
st.title("✂️ Visual Audio Range Trimmer & Editor")

# 2. HELPER TO SLICE WAV IN MEMORY
def slice_wav_bytes(file_path, start_sec, end_sec):
    """Reads a master WAV file and returns only the sliced segment as bytes on-the-fly."""
    if not os.path.exists(file_path):
        return None
        
    with wave.open(file_path, 'rb') as wav:
        params = wav.getparams()
        sample_rate = wav.getframerate()
        
        # Calculate frame boundaries
        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        num_frames_to_read = max(1, end_frame - start_frame)
        
        # Extract isolated sound frames
        wav.setpos(start_frame)
        audio_frames = wav.readframes(num_frames_to_read)
        
    # Recompile into a structural valid WAV byte buffer header
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, 'wb') as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(audio_frames)
        
    return output_buffer.getvalue()


# 3. DATA LOADERS & STATE INITIALIZATION
def load_data():
    target_file = UPDATED_METADATA_FILE if os.path.exists(UPDATED_METADATA_FILE) else METADATA_FILE
    if not os.path.exists(target_file):
        st.error(f"Metadata file not found at: `{target_file}`")
        return None
    try:
        df = pd.read_csv(target_file, sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(target_file, sep=",")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

if "df_segments" not in st.session_state:
    st.session_state.df_segments = load_data()

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

df = st.session_state.df_segments

# 4. SIDEBAR NAVIGATION
st.sidebar.header("⏳ Audio Slices Queue")
if df is not None:
    for idx, row in df.iterrows():
        speaker = row.get("speaker", "??")
        start = row.get("start_sec", 0.0)
        end = row.get("end_sec", 0.0)
        text_preview = str(row.get("transcription", ""))[:20] + "..."
        
        is_selected = idx == st.session_state.selected_idx
        prefix = "▶️ " if is_selected else ""
        label = f"{prefix}[{start:.2f}s - {end:.2f}s] {speaker}"
        
        if st.sidebar.button(label, key=f"seg_{idx}", use_container_width=True):
            st.session_state.selected_idx = idx
            st.rerun()

# 5. MAIN EDITOR HUB
if df is not None and st.session_state.selected_idx is not None:
    idx = st.session_state.selected_idx
    row_data = df.iloc[idx]
    
    origin_rel_path = row_data["origin_filename"]
    
    # st.subheader(f"Trimming Segment #{idx}")
    # st.caption(f"📁 Source Master Track: `{origin_rel_path}`")
    
    st.markdown(
        f"<p style='padding-top: 32px; color: gray; margin: 0; font-size: 14px;'>"
        f"📁 <b>Source:</b> <code>{origin_rel_path}</code> &nbsp;|&nbsp; "
        f"💾 <b>Save:</b> <code>{row_data['seg_filename']}</code>"
        f"</p>", 
        unsafe_allow_html=True
    )
    
    if not os.path.exists(origin_rel_path):
        st.error(f"Original audio file missing at expected path: `{origin_rel_path}`")
    else:
        # Get overall file attributes to set boundaries automatically
        with wave.open(origin_rel_path, 'rb') as w:
            total_file_seconds = w.getnframes() / w.getframerate()

        st.write("---")
        st.markdown("### 🎚️ Drag & Cut Audio Boundaries")
        
        # DOUBLE-ENDED RANGE SLIDER
        time_range = st.slider(
            "Isolate Audio Playback Area (Left handle = start boundary | Right handle = end boundary)",
            min_value=0.0,
            max_value=float(total_file_seconds),
            value=(float(row_data["start_sec"]), float(row_data["end_sec"])),
            step=0.05,
            format="%.2f seconds"
        )
        
        selected_start, selected_end = time_range
        selected_duration = selected_end - selected_start

        # DYNAMIC REAL-TIME SLICE 
        # Grabs ONLY the frames between the two slider handles
        sliced_audio_bytes = slice_wav_bytes(origin_rel_path, selected_start, selected_end)

        if sliced_audio_bytes:
            # 1. Create a clearable layout slot
            audio_placeholder = st.empty()
            
            # 2. Inject the fresh audio stream straight into it
            audio_placeholder.audio(sliced_audio_bytes, format="audio/wav")
            
            st.caption(f"🎵 Playing an isolated audio chunk. Length: **{selected_duration:.2f} seconds**.")
        
        # # Display Statistics Tracker
        # col_stat1, col_stat2 = st.columns(2)
        # with col_stat1:
        #     st.metric("Timeline Window Marks", f"{selected_start:.2f}s — {selected_end:.2f}s")
        # with col_stat2:
        #     st.metric("Extracted Duration Size", f"{selected_duration:.2f}s")

        st.write("---")
        
        # Form Submission Environment
        with st.form(key=f"visual_edit_form_{idx}", clear_on_submit=False):
            st.markdown("### 📝 Edit Transcription Notes")

            new_text = st.text_area(
                "Transcription text content:", 
                value=str(row_data["transcription"]), 
                height=max(100, 50* int(len(row_data["transcription"])/100))
            )
            
            st.write("---")
            
            # --- SPLIT LAYOUT: SPEAKER & TYPE SIDE-BY-SIDE ---
            cols = st.columns(2)
            
            # --- LEFT COLUMN: SPEAKER ATTRIBUTION ---
            with cols[0]:
                st.markdown("##### 👥 Speaker Attribution")
                
                # Gather unique speakers dynamically from data
                existing_speakers = sorted(list(df["speaker"].dropna().unique()))
                speaker_options = existing_speakers + ["➕ Add New Speaker..."]
                
                current_speaker = str(row_data.get("speaker", ""))
                default_spk_idx = existing_speakers.index(current_speaker) if current_speaker in existing_speakers else 0

                selected_speaker_choice = st.selectbox(
                    "Select Speaker identity:",
                    options=speaker_options,
                    index=default_spk_idx,
                    key=f"spk_select_{idx}"
                )
                
                # Reveal structural text input underneath dropdown if triggered
                if selected_speaker_choice == "➕ Add New Speaker...":
                    final_speaker = st.text_input(
                        "Enter New Speaker Name/ID:", 
                        placeholder="e.g., P3, Moderator",
                        key=f"spk_text_{idx}"
                    ).strip()
                else:
                    final_speaker = selected_speaker_choice

            with cols[1]:
                st.markdown("##### 🏷️ Interaction Type Classification")
                
                # Gather unique interaction types dynamically from data
                existing_types = sorted(list(df["type"].dropna().unique()))
                type_options = existing_types + ["➕ Add New Type..."]
                
                current_type = str(row_data.get("type", ""))
                default_type_idx = existing_types.index(current_type) if current_type in existing_types else 0
                
                selected_type_choice = st.selectbox(
                    "Select Interaction Type:",
                    options=type_options,
                    index=default_type_idx,
                    key=f"type_select_{idx}"
                )
                
                # Reveal structural text input underneath dropdown if triggered
                if selected_type_choice == "➕ Add New Type...":
                    final_type = st.text_input(
                        "Enter New Interaction Type:", 
                        placeholder="e.g., laughter, interruption",
                        key=f"type_text_{idx}"
                    ).strip()
                else:
                    final_type = selected_type_choice

            cols = st.columns(2)

            with cols[0]:
                st.markdown("##### 🧑Sex")
                sex_options = ["Female", "Male", "Other"]
                raw_current = str(row_data.get("sex", "")).strip().capitalize()
    
                # Assign the default dropdown index based on your fixed list
                default_sex_idx = sex_options.index(raw_current) if raw_current in sex_options else 2

                final_sex = st.selectbox(
                    "Select Sex:",
                    options=sex_options,
                    index=default_sex_idx,
                    key=f"sex_select_{idx}"
                )

            with cols[1]:
                st.markdown("##### ⏳Age")
                current_age = row_data.get("age", 25) # Default placeholder if empty
                
                # Using a number input box for precise integer age tracking
                final_age = st.number_input(
                    "Enter Age:",
                    min_value=0,
                    max_value=120,
                    value=int(current_age) if pd.notna(current_age) else 25,
                    step=1,
                    key=f"age_input_{idx}"
                )

            # st.markdown("### 📊 Continuous and Categorical Metadata")

            # Create 6 equal-width horizontal columns
            cols = st.columns(4)
           
            # --- COLUMN 2: EMOTION CATEGORY ---
            with cols[0]:
                st.markdown("##### 🎭Emotion")
                unique_emotions = list(df["emotion"].dropna().unique()) if "emotion" in df.columns else ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
                if "Other" not in unique_emotions:
                    unique_emotions.append("Other")
                emotion_options = sorted(unique_emotions)
                
                current_emotion = str(row_data.get("emotion", "Neutral"))
                default_emo_idx = emotion_options.index(current_emotion) if current_emotion in emotion_options else 0

                final_emotion = st.selectbox(
                    "Select Emotion:",
                    options=emotion_options,
                    index=default_emo_idx,
                    key=f"emotion_select_{idx}"
                )

            # --- COLUMN 3: AROUSAL ---
            with cols[1]:
                st.markdown("##### Arousal")
                current_arousal = row_data.get("arousal", 5.0)
                
                # Typically rated on a 1-9 or 1-10 scale; adjust min/max if yours differs
                final_arousal = st.slider(
                    "Arousal Level:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_arousal) if pd.notna(current_arousal) else 5.0,
                    step=0.1,
                    key=f"arousal_slider_{idx}"
                )

            # --- COLUMN 4: VALENCE ---
            with cols[2]:
                st.markdown("##### Valence")
                current_valence = row_data.get("valence", 5.0)
                
                final_valence = st.slider(
                    "Valence Level:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_valence) if pd.notna(current_valence) else 5.0,
                    step=0.1,
                    key=f"valence_slider_{idx}"
                )

            # --- COLUMN 5: DOMINANCE ---
            with cols[3]:
                st.markdown("##### Dominance")
                current_dominance = row_data.get("dominance", 5.0)
                
                final_dominance = st.slider(
                    "Dominance Level:",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_dominance) if pd.notna(current_dominance) else 5.0,
                    step=0.1,
                    key=f"dominance_slider_{idx}"
                )

            st.write("---")
            submit_btn = st.form_submit_button("💾 Save Trimmed Segment Data")
            
            if submit_btn:
                # Existing time and text code...
                st.session_state.df_segments.at[idx, "start_sec"] = round(selected_start, 2)
                st.session_state.df_segments.at[idx, "end_sec"] = round(selected_end, 2)
                st.session_state.df_segments.at[idx, "transcription"] = new_text

                # ADD THESE ROWS TO ATTACH YOUR NEW 6 COLUMNS:
                st.session_state.df_segments.at[idx, "sex"] = final_sex
                st.session_state.df_segments.at[idx, "age"] = final_age
                st.session_state.df_segments.at[idx, "emotion"] = final_emotion
                st.session_state.df_segments.at[idx, "arousal"] = round(final_arousal, 2)
                st.session_state.df_segments.at[idx, "valence"] = round(final_valence, 2)
                st.session_state.df_segments.at[idx, "dominance"] = round(final_dominance, 2)

                # Save CSV
                st.session_state.df_segments.to_csv(UPDATED_METADATA_FILE, index=False)
                st.success("All multi-column metadata successfully updated!")
                st.rerun()
else:
    st.info("👈 Please select a segmented track slice from the left sidebar index to display its audio frame timeline control properties.")