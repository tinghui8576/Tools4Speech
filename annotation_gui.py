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
METADATA_FILE = "outputs/dyad/final_labels.txt"  # Path to your transcription CSV
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
    
    st.subheader(f"Trimming Segment #{idx}")
    st.caption(f"📁 Source Master Track: `{origin_rel_path}`")
    
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
        
        # Display Statistics Tracker
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Timeline Window Marks", f"{selected_start:.2f}s — {selected_end:.2f}s")
        with col_stat2:
            st.metric("Extracted Duration Size", f"{selected_duration:.2f}s")

        st.write("---")
        
        # Form Submission Environment
        with st.form(key=f"visual_edit_form_{idx}", clear_on_submit=False):
            st.markdown("### 📝 Edit Transcription Notes")
            
            new_text = st.text_area(
                "Transcription text content:", 
                value=str(row_data["transcription"]), 
                height=100
            )
            
            st.write("---")
            
            # --- SPLIT LAYOUT: SPEAKER & TYPE SIDE-BY-SIDE ---
            col_left, col_right = st.columns(2)
            
            # --- LEFT COLUMN: SPEAKER ATTRIBUTION ---
            with col_left:
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

            # --- RIGHT COLUMN: INTERACTION TYPE CLASSIFICATION ---
            with col_right:
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

            st.write("---")
            submit_btn = st.form_submit_button("💾 Save Trimmed Segment Data")
            
            if submit_btn:
                # Validation safeguards
                if not final_speaker:
                    st.error("❌ Please enter a valid name for the new speaker.")
                elif not final_type:
                    st.error("❌ Please enter a valid classification name for the new interaction type.")
                else:
                    # Assign coordinates to session memory dataframe
                    st.session_state.df_segments.at[idx, "start_sec"] = round(selected_start, 2)
                    st.session_state.df_segments.at[idx, "end_sec"] = round(selected_end, 2)
                    st.session_state.df_segments.at[idx, "duration_sec"] = round(selected_duration, 2)
                    st.session_state.df_segments.at[idx, "transcription"] = new_text
                    
                    # Store variables
                    st.session_state.df_segments.at[idx, "speaker"] = final_speaker
                    st.session_state.df_segments.at[idx, "type"] = final_type
                    
                    # Flush data matrix to disk storage
                    st.session_state.df_segments.to_csv(UPDATED_METADATA_FILE, index=False)
                    st.success("Changes permanently saved side-by-side!")
                    st.rerun()
else:
    st.info("👈 Please select a segmented track slice from the left sidebar index to display its audio frame timeline control properties.")