
from label_studio_sdk import LabelStudio
import os
import json
import time
import subprocess
import requests
import webbrowser
import socket
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


def is_port_available(port):
    """Check whether a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def get_available_port(preferred_port=8080):
    """
    Use preferred port if available.
    Otherwise let OS choose a free port.
    """

    if is_port_available(preferred_port):
        return preferred_port

    # Find another free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]

def get_network_ip():
    """
    Similar to Streamlit Network URL detection
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Does not send data, only selects interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]

    except Exception:
        ip = "127.0.0.1"

    finally:
        s.close()

    return ip


def start_labelstudio(
    USERNAME=None,
    PASSWORD=None,
    TOKEN=None,
):
    env = os.environ.copy()

    port = get_available_port()

    local_url = (
        f"http://localhost:{port}"
    )

    network_ip = get_network_ip()

    network_url = (
        f"http://{network_ip}:{port}"
    )

    cmd = [
        "label-studio",
        "start",
        "--host", network_ip,
        "--port", str(port),
        "--no-browser",
    ]


    if USERNAME and PASSWORD and TOKEN:
        cmd += [
            "--username", USERNAME,
            "--password", PASSWORD,
            "--user-token", TOKEN,
        ]

        EXPORT_DIR = st.session_state.get("labelstudio_storage_path")
        

        env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
        env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = EXPORT_DIR


    process = subprocess.Popen(cmd, env=env)
    if TOKEN:
        client = LabelStudio(base_url=network_url, api_key=TOKEN)
        wait_for_labelstudio(client)
        me = client.users.whoami()

        print("")
        print("Label Studio running!")
        print("")
        print(f"Local URL:   {local_url}")
        print(f"Network URL: {network_url}")
        print("Username:", me.username)
        print("Email:", me.email)
        print("")
    else:
        client = None
        wait_for_server(network_url)
        
    # process = subprocess.Popen(
    #     [
    #         "label-studio",
    #         "start",
    #         "--host",
    #         network_ip,
    #         "--port",
    #         str(port),
    #         "--username",
    #         USERNAME,
    #         "--password",
    #         PASSWORD,
    #         "--no-browser",
    #     ],
    #     env=env,
    # )



    return client, local_url, network_url

    # return LS_URL


def wait_for_server(url, timeout=60):
    """
    Wait until Label Studio web server is available.
    No API key required.
    """

    for i in range(timeout):
        try:
            r = requests.get(
                url,
                timeout=2
            )

            # Label Studio login page normally returns 200
            if r.status_code == 200:
                print("Label Studio server ready")
                return True

        except requests.exceptions.RequestException:
            pass

        time.sleep(1)

    raise RuntimeError(
        "Label Studio did not start"
    )

def wait_for_labelstudio(client, timeout=30):
    last_error = None

    for _ in range(timeout):
        try:
            version = client.versions.get()
            print("Label Studio ready:", version)
            return

        except Exception as e:
            last_error = e
            time.sleep(1)

    raise RuntimeError(
        f"Label Studio did not start.\nLast error: {last_error}"
    )

# ==========================
# Create Local Storage
# ==========================

def create_storage(client, project_id):
    STORAGE_PATH = os.path.join(st.session_state.get("labelstudio_storage_path"), "data")
    storage = client.import_storage.local.create(
        project=project_id,
        title="Generated Audio",
        path=STORAGE_PATH,
        regex_filter=".*\\.wav",
        use_blob_urls=False,
    )

    client.import_storage.local.sync(
        id=storage.id
    )
    
    print(
        "Storage synced:",
        storage.id
    )


    # return storage.id


# ==========================
# Label Studio XML
# ==========================
def build_label_config(columns):

    xml = [
        """
<View>
    <Header value="Conversation Audio"/>
    <Audio name="audio" value="$audio"/>

    <Labels name="speaker" toName="audio">
        <Label value="P1" background="#FF0000"/>
        <Label value="P2" background="#0000FF"/>
        <Label value="P3" background="#00AA00"/>
    </Labels>
"""
    ]

    if "transcript" in columns:
        xml.append("""
    <TextArea
        name="transcript"
        toName="audio"
        perRegion="true"
        maxSubmissions="1"
        editable="true"
        rows="3"/>
""")

    if "emoCat" in columns:
        xml.append("""
    <Choices name="emoCat" toName="audio" perRegion="true">
        <Choice value="Neutral"/>
        <Choice value="Happy"/>
        <Choice value="Sadness"/>
        <Choice value="Angry"/>
        <Choice value="Other"/>
    </Choices>
""")

    if "sex" in columns:
        xml.append("""
    <Choices name="sex" toName="audio" perRegion="true">
        <Choice value="Male"/>
        <Choice value="Female"/>
    </Choices>
""")

    if "age" in columns:
        xml.append("""
    <Number
        name="age"
        toName="audio"
        perRegion="true"/>
""")

    for dim in ["arousal", "valence", "dominance"]:
        if dim in columns:
            xml.append(f"""
    <Number
        name="{dim}"
        toName="audio"
        perRegion="true"/>
""")

    xml.append("</View>")

    return "\n".join(xml)


# ==========================
# Create Project
# ==========================

def create_project(client, WORKSPACE_ID):

    label_config = st.session_state.get("label_config") 
    project = client.projects.create(
        title="Speech Annotation",
        label_config=label_config,
        workspace=WORKSPACE_ID,
    )
    return project.id

# ==========================
# Import Predictions
# ==========================

def import_tasks(client, project_id):

    create_storage(
            client,
            project_id
        )
    
    tasks = st.session_state.get("annotation_tasks")

    if tasks:
        client.projects.import_tasks(
            id=project_id,
            request=tasks
        )
        print(
            "Imported:",
            len(tasks)
        )

