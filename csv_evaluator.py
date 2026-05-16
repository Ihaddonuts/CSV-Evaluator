# streamlit_app.py
import io
import json
import os 
import requests
import streamlit as st
import logging
from logging.handlers import RotatingFileHandler
import time 

API_BASE = "http://127.0.0.1:8000"  

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok = True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ui_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "ui.log"),
    maxBytes=1_500_000, 
    backupCount=3,
    encoding="utf-8"
)

ui_file_handler.setLevel(logging.INFO)
ui_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "%Y-%m-%d %H:%M:%S",
))

logging.getLogger().addHandler(ui_file_handler)
log = logging.getLogger("ui")
log.info("UI starting")

def fetch_datasets():
    try:
        r = requests.get(f"{API_BASE}/datasets/", timeout=30)
        if r.status_code == 200:
            items =  r.json().get("items", [])
            log.info(f"/datasets fail | status = {r.status_code} | count={len(items)}")
            return items
        else:
            log.warning(f"/datasets fail | status={r.status_code} | body={r.text[:200]}")
            st.warning(f"Could not fetch datasets ({r.status_code}).")
            return []
    except requests.RequestException as e:
        log.error(f"/datasets error | err={e}")
        st.warning(f"Dataset list error: {e}")
        return []

def fetch_latest_metrics(dataset_id: str):
    try:
        r = requests.get(f"{API_BASE}/metrics/latest/", params={"dataset_id": dataset_id}, timeout=30)
        if r.status_code == 200:
            log.info(f"/metrics/latest ok | dataset_id={dataset_id}")
            return r.json()
        else:
            log.warning(f"/metrics/latest fail | dataset_id={dataset_id} | status={r.status_code}")
            st.error(f"Latest metrics fetch failed ({r.status_code}): {r.text}")
            return None
    except requests.RequestException as e:
        log.error(f"/metrics/latest error | dataset_id={dataset_id} | err={e}")
        st.error(f"Latest metrics error: {e}")
        return None
def render_stream_fake(text: str, delay_s: float = 0.02):
    if not text:
        return
    ph = st.empty()
    out = []
    for token in text.split():       
        out.append(token)
        ph.markdown(" ".join(out))
        time.sleep(delay_s)


st.set_page_config(page_title="Model Evaluator", layout="centered")
st.title("CSV Evaluator")

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

st.header("1) Upload CSV")
csv_file = st.file_uploader(
    "Choose a .csv file",
    type=["csv"],
    help="Required columns: Ground Truth, Score, Threshold"
)

col_up_a, col_up_b = st.columns([1,1])
with col_up_a:
    score_orientation = st.selectbox(
        "Score orientation for ROC AUC",
        options=["positive", "negative"],
        index=0,
        help="positive: higher score ⇒ more positive; negative: higher score ⇒ more negative"
    )
with col_up_b:
    uploaded_filename_placeholder = st.empty()

if st.button("Upload to server", use_container_width=True, type="primary"):
    if not csv_file:
        st.error("Please choose a .csv file first.")
        log.warning("upload click with no file")
    else:
        try:
            files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
            log.info(f"upload attempt | filename={csv_file.name} | size={len(files['file'][1])}B")
            resp = requests.post(f"{API_BASE}/csv-upload/", files=files, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.dataset_id = data.get("dataset_id")
                uploaded_filename_placeholder.write(f"Uploaded: **{data.get('filename','(unknown)')}**")
                st.success(f"Stored dataset_id: {st.session_state.dataset_id}")
            else:
                st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                log.warning(f"upload fail | status={resp.status_code} | body={resp.text[:200]}")
        except requests.RequestException as e:
            st.error(f"Upload error: {e}")
            log.error(f"upload error | err={e}")

st.divider()

st.header("2) Select a dataset")
datasets = fetch_datasets()

if datasets:
    options = {
        f"{d.get('filename','(unknown)')} — {d.get('uploaded_at','')}" : d["dataset_id"]
        for d in datasets
    }
    label = st.selectbox("Pick a dataset", options=list(options.keys()))
    picked_dataset_id = options[label]

    st.session_state.dataset_id = picked_dataset_id
    log.info(f"dataset picked | dataset_id={picked_dataset_id}")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("View latest metrics", use_container_width=True):
            latest = fetch_latest_metrics(picked_dataset_id)
            if latest:
                if not latest.get("has_metrics"):
                    st.info("No metrics computed yet for this dataset. Use 'Evaluate metrics' to compute now.")
                    log.info(f"latest metrics | dataset_id={picked_dataset_id} | none")
                else:
                    st.subheader("Latest Metrics (from database)")
                    cols = st.columns(3)
                    cols[0].metric("Rows (total)", f"{latest.get('rows_total',0)}")
                    cols[1].metric("Rows (valid)", f"{latest.get('rows_valid',0)}")
                    cols[2].metric("Rows (invalid)", f"{latest.get('rows_invalid',0)}")

                    m = latest.get("metric", {})
                    mcols = st.columns(5)
                    mcols[0].metric("Accuracy", f"{m.get('accuracy',0.0):.6f}")
                    mcols[1].metric("Sensitivity (TPR)", f"{m.get('sensitivity',0.0):.6f}")
                    mcols[2].metric("Specificity (TNR)", f"{m.get('specificity',0.0):.6f}")
                    mcols[3].metric("Precision (PPV)", f"{m.get('precision',0.0):.6f}")
                    mcols[4].metric("ROC AUC", f"{m.get('roc_auc',0.0):.6f}")

                    st.caption(f"Computed at: {latest.get('computed_at','')}")
                    st.caption(f"Orientation (predictions): {m.get('orientation','score >= threshold: Positive')}")
                    st.caption(f"Score orientation (AUC): {m.get('score_orientation','positive')}")

                    cm = latest.get("confusion", {})
                    st.subheader("Confusion Matrix")
                    st.table({
                        "": ["Predicted Positive", "Predicted Negative"],
                        "Ground Truth Positive": [cm.get("tp",0), cm.get("fn",0)],
                        "Ground Truth Negative": [cm.get("fp",0), cm.get("tn",0)]
                    })
else:
    st.info("No datasets uploaded yet. Use 'Upload to server' above to add one.")
    log.info("No datasets available")

st.divider()


st.header("3) Evaluate")
st.write("Press the button to compute Accuracy, Sensitivity, Specificity, Precision, and ROC AUC.")

if st.button("Evaluate metrics", use_container_width=True):
    if not st.session_state.dataset_id:
        st.error("No dataset uploaded yet. Please upload a CSV first.")
        log.warning("evaluate click without dataset_id")
    else:
        try:
            params = {
                "dataset_id": st.session_state.dataset_id,
                "score_orientation": score_orientation
            }
            log.info(f"evaluate attempt | dataset_id={st.session_state.dataset_id} | score_orientation={score_orientation}")
            resp = requests.post(f"{API_BASE}/evaluate/", params=params, timeout=60)
            if resp.status_code != 200:
                st.error(f"Evaluate failed ({resp.status_code}): {resp.text}")
                log.warning(f"evaluate fail | status={resp.status_code} | body={resp.text[:200]}")
            else:
                data = resp.json()

               
                st.subheader("Summary")
                cols = st.columns(3)
                cols[0].metric("Rows (total)", f"{data.get('rows_total',0)}")
                cols[1].metric("Rows (valid)", f"{data.get('rows_valid',0)}")
                cols[2].metric("Rows (invalid)", f"{data.get('rows_invalid',0)}")

               
                m = data.get("metric", {})
                st.subheader("Metrics")
                mcols = st.columns(5)
                mcols[0].metric("Accuracy", f"{m.get('accuracy',0.0):.6f}")
                mcols[1].metric("Sensitivity (TPR)", f"{m.get('sensitivity',0.0):.6f}")
                mcols[2].metric("Specificity (TNR)", f"{m.get('specificity',0.0):.6f}")
                mcols[3].metric("Precision (PPV)", f"{m.get('precision',0.0):.6f}")
                mcols[4].metric("ROC AUC", f"{m.get('roc_auc',0.0):.6f}")

                st.caption(f"Orientation (predictions): {m.get('orientation','score >= threshold: Positive')}")
                st.caption(f"Score orientation (AUC): {m.get('score_orientation', score_orientation)}")

              
                st.subheader("Confusion Matrix")
                cm = data.get("confusion", {})
                st.table({
                    "": ["Predicted Positive", "Predicted Negative"],
                    "Ground Truth Positive": [cm.get("tp",0), cm.get("fn",0)],
                    "Ground Truth Negative": [cm.get("fp",0), cm.get("tn",0)]
                })

                
                with st.expander("See raw response"):
                    st.json(data)

                log.info(
                    "evaluate ok | "
                    f"dataset_id={st.session_state.dataset_id} | "
                    f"acc={m.get('accuracy')} auc={m.get('roc_auc')}"
                )

        except requests.RequestException as e:
            st.error(f"Evaluate error: {e}")
            log.error(f"evaluate error | err = {e}")



st.divider()
st.header("AI assistance")

def render_stream_fake(text: str, delay_s: float = 0.02):
    if not text:
        return
    ph = st.empty()
    out = []
    for token in text.split(): 
        out.append(token)
        ph.markdown(" ".join(out))
        time.sleep(delay_s)

current_dataset_id = st.session_state.get("dataset_id")

has_metrics = False
if current_dataset_id:
    latest = fetch_latest_metrics(current_dataset_id)
    has_metrics = bool(latest and latest.get("has_metrics"))

if not current_dataset_id:
    st.info("Pick or upload a dataset above, then evaluate to enable AI assistance.")
elif not has_metrics:
    st.info("No metrics yet for this dataset. Click **Evaluate metrics** first, then ask your question.")
else:
    col_ai_a, col_ai_b = st.columns([2,1])
    with col_ai_a:
        user_msg = st.chat_input("Tell me about the metrics'")
    with col_ai_b:
        model_name = st.text_input("Model", value="gemma3:4b")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        delay_s = st.slider("Stream delay (s)", 0.0, 0.05, 0.02, 0.005,
                            help="Visual typing speed only (frontend).")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.subheader("Chat")
        for role, content in reversed(st.session_state.chat_history):
             if role == "user":
                st.markdown(f"**You:** {content}")
             else:
                st.markdown(f"**AI:** {content}")

       
    if user_msg is not None:
        log.info(
            "ai chat send | combined agent | "
            f"model={model_name} | temp={temperature} | dataset_id={current_dataset_id} | len={len(user_msg)}"
        )

        st.session_state.chat_history.append(("user", user_msg))

        params = {
            "question": user_msg,
            "model": model_name,
            "temperature": temperature
        }
        if current_dataset_id:
            params["dataset_id"] = current_dataset_id

        try:
            r = requests.post(f"{API_BASE}/agent/web-assist/", params=params, timeout=120)
            if r.status_code == 200:
                reply = (r.json() or {}).get("answer", "")
                log.info(f"/agent/web-assist ok | len={len(reply)}")
            else:
                log.warning(f"/agent/web-assist fail | status={r.status_code} | body={r.text[:200]}")
                st.error(f"Chat failed ({r.status_code})")
                reply = ""
        except requests.RequestException as e:
            log.error(f"/agent/web-assist error | err={e}")
            st.error(f"Chat error: {e}")
            reply = ""

        render_stream_fake(reply, delay_s=delay_s)
        if reply:
            st.session_state.chat_history.append(("assistant", reply))
        st.rerun()
st.caption("Tip: Start your FastAPI server first, then run this Streamlit app.")
