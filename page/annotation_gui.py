import io
import wave
import os
import time
import pandas as pd
import streamlit as st
import numpy as np
# from page.setup import merge_datasets, normalize_schema, _next_available_name
from src.annotation import start_labelstudio, build_label_config, create_project, import_tasks
import json

# -----------------------------
# CORE APP ENGINE STATE INITIALIZATION
# -----------------------------
def run_annotation() -> None:
    
    # st.title("✂️ Editing Annotation")
    # if st.session_state['project'].get("df") is None:
    #     if st.session_state.get("result"):


    # st.set_page_config(page_title="Visual Timeline Audio Trimmer", initial_sidebar_state="expanded", layout="wide")
    _labelstudio_token = os.environ.get("TOKEN", "").strip()
    _labelstudio_user = os.environ.get("USERNAME", "").strip()
    _labelstudio_password = os.environ.get("PASSWORD", "").strip()
    _labelstudio_workspace_id = os.environ.get("WORKSPACE_ID", "").strip()

    _label_ready = all([
        _labelstudio_token,
        _labelstudio_user,
        _labelstudio_password,
    ])

    if not _label_ready:
        network_url = st.session_state.get("ls_url")
        if not network_url:
            st.subheader("Wait for label studio to open")
            client, local_url, network_url  = start_labelstudio()
            st.session_state['ls_url'] = network_url
            st.rerun()
        
        url = (
            f"{network_url}/user/login/"
        )
        st.components.v1.iframe(
            url,
            height=1000,
            scrolling=True
        )
    elif st.session_state.get("result") and "labelstudio_export" in st.session_state["result"]:
            
        # Extract directories safely
        _export_dir = st.session_state["result"]["labelstudio_export"]
        st.session_state['labelstudio_storage_path'] = _export_dir # Fixed typo here
        
        _task_json = os.path.join(_export_dir, "data", "labelstudio_tasks.json")

        # Safe File Loading wrapper
        try:
            with open(_task_json, encoding="utf8") as f:
                tasks = json.load(f)
                st.session_state['annotation_tasks'] = tasks
                
            # Parse available fields efficiently
            available_fields = set()
            for task in tasks:
                # Combining lists gracefully with addition avoids deep nesting layers
                annotations_and_predictions = task.get("annotations", []) + task.get("predictions", [])
                for ann in annotations_and_predictions:
                    for res in ann.get("result", []):
                        if "from_name" in res:
                            available_fields.add(res["from_name"])
                            
            print(f"Detected LabelStudio fields: {available_fields}")
            st.session_state["label_config"] = build_label_config(available_fields)
            
        except FileNotFoundError:
            st.error(f"Could not find data file at: {_task_json}")
        except json.JSONDecodeError:
            st.error("The task JSON file appears to be corrupted or invalid.")

        
        if "labelstudio_storage_path" in st.session_state:
            client, local_url, network_url = start_labelstudio(
                USERNAME=_labelstudio_user, 
                PASSWORD = _labelstudio_password,
                TOKEN=_labelstudio_token)
            st.session_state["ls_client"] = client
            st.session_state["ls_url"] = network_url
            print(st.session_state.get("ls_url"))

            if "annotation_tasks" in st.session_state:

                project_id = st.session_state["project_id"] if "project_id" in st.session_state else create_project(client, _labelstudio_workspace_id)

                import_tasks(
                    client,
                    project_id
                )


                url = (
                    f"{network_url}/projects/{project_id}/data"
                )

                print(
                    "Open:",
                    url
                )
                st.components.v1.iframe(
                    url,
                    height=1000,
                    scrolling=True
                )
                # webbrowser.open(url)

    else:
        project = st.session_state["project"]

        default_dir = st.session_state.get("_output_dir_suggestion", "outputs")

        output_dir = st.text_input(
            "Output directory",
            value=default_dir,
            key="update_output_dir"
        )

        project["files"]["output_dir"] = output_dir
        print("No results found in session state.")