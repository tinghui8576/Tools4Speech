"""
Local GUI for the Speech VAD / Diarization / Transcription pipeline.

Wraps :func:`src.conversation.process_conversation` in a Streamlit form so
the pipeline can be run without writing any code.

Launch with:
    streamlit run app_gui.py
"""

from __future__ import annotations

# from upload import run_upload
from page.setup import init_state
from page.pipeline import run_pipeline
from page.annotation_gui import run_annotation
from page.upload import run_upload
import streamlit as st



# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Speech Pipeline",
        page_icon="🗣️",
        layout="wide",
    )

    tab_upload, tab_pipeline, tab_annotation = st.tabs(
        [
            "📤 Upload",
            "🗣️ Pipeline",
            "✂️ Annotation",
        ],
        default="🗣️ Pipeline",
        on_change= "rerun",
        key="mode"
    )
    
    
    init_state()
    with tab_upload:
        if st.session_state.mode == "📤 Upload":
            run_upload()

    with tab_pipeline:
        if st.session_state.mode == "🗣️ Pipeline":
            run_pipeline()

    with tab_annotation:
        if st.session_state.mode == "✂️ Annotation":
            run_annotation()
    

if __name__ == "__main__":
    main()
