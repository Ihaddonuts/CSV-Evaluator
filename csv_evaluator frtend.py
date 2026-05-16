from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import csv, io, uuid, sqlite3, time, os, logging 
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from db import get_conn, init_db
from ollama import chat as ollama_chat, ChatResponse
import requests 

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok = True)

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s |%(levelname)s | %(name)s | %(message)s ",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes= 2_000_000,
    backupCount = 5,
    encoding = "utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "%Y-%m-%d %H:%M:%S"
))
logging.getLogger().addHandler(file_handler)
log = logging.getLogger("app")

app = FastAPI()
app_version = "0.1.0"
DB_PATH = "data.db"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
GENERIC_SYS_PROMPT = (
    "You are a concise evaluator. Answer briefly (<=60 words) in bullet points. "
    "Explain classification metrics simply and relate to given values."
)
def _wiki_search_top(query: str, timeout_s: int = 10) -> str | None:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"},
            timeout=timeout_s
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) >= 2:
            items = data[1] or []
            return items[0] if items else None
    except Exception as e:
        log.warning(f"wiki search fail | q={query} | err={e}")
    return None


def _wiki_summary(title: str, timeout_s: int = 10, max_chars: int = 800) -> str:
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
            timeout=timeout_s,
            headers={"accept": "application/json"}
        )
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        data = r.json()
        txt = (data.get("extract") or "").strip()
        return (txt[:max_chars] + "…") if len(txt) > max_chars else txt
    except Exception as e:
        log.warning(f"wiki summary fail | title={title} | err={e}")
        return ""
def _web_context_wikipedia(query: str) -> str:
    title = _wiki_search_top(query)
    if not title:
        return ""
    summary = _wiki_summary(title)
    if not summary:
        return ""
    return f"Wikipedia | {title}: {summary}"

def _agent_web_system_prompt_full(m: dict | None, d: dict | None, web_ctx: str) -> str:
    base = (
        "You are a concise evaluator. Answer in ≤60 words with bullet points. "
        "Prioritize the dataset metrics; use the web note only for general context."
    )
    parts = [base]
    if m:
        parts.append(
            f"Counts: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}. "
            f"Core: acc={m['accuracy']}, sens={m['sensitivity']}, spec={m['specificity']}, "
            f"prec={m['precision']}, auc={m['roc_auc']}."
        )
    if d:
        parts.append(
            f"Derived: f1={d['f1']}, fpr={d['fpr']}, fnr={d['fnr']}, "
            f"bal_acc={d['balanced_accuracy']}, prev={d['prevalence']}."
        )
    if web_ctx:
        parts.append(f"Web note: {web_ctx}")
    return " ".join(parts)


def _latest_metrics_row(dataset_id: str):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT tp, tn, fp, fn,
                   rows_total, rows_valid, rows_invalid,
                   accuracy, sensitivity, specificity, precision, roc_auc,
                   computed_at
            FROM metrics
            WHERE dataset_id = ?
            ORDER BY datetime(computed_at) DESC
            LIMIT 1
        """, (dataset_id,))
        row = cur.fetchone()
        if not row:
            return None
        (tp, tn, fp, fn,
         rows_total, rows_valid, rows_invalid,
         acc, sens, spec, prec, auc, when) = row
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "rows_total": rows_total, "rows_valid": rows_valid, "rows_invalid": rows_invalid,
            "accuracy": acc, "sensitivity": sens, "specificity": spec, "precision": prec, "roc_auc": auc,
            "computed_at": when
        }
    except Exception as e:
        log.error(f"latest metrics db error | dataset_id={dataset_id} | err={e}", exc_info=True)
        return None
    finally:
        conn.close()


def _derived_from_confusion(tp: int, tn: int, fp: int, fn: int):
    
    def _safe(a, b): return round(a / b, 6) if b else 0.0
    support_pos = tp + fn
    support_neg = tn + fp
    tpr = _safe(tp, support_pos)        
    tnr = _safe(tn, support_neg)       
    ppv = _safe(tp, tp + fp)            
    npv = _safe(tn, tn + fn)
    fpr = _safe(fp, fp + tn)
    fnr = _safe(fn, fn + tp)
    f1  = _safe(2 * tp, 2 * tp + fp + fn)
    bal_acc = round((tpr + tnr) / 2.0, 6)
    prevalence = _safe(support_pos, support_pos + support_neg)
    return {
        "tpr": tpr, "tnr": tnr, "ppv": ppv, "npv": npv,
        "fpr": fpr, "fnr": fnr, "f1": f1, "balanced_accuracy": bal_acc,
        "prevalence": prevalence
    }


def _agent_system_prompt_basic(m: dict, d: dict) -> str:
    return (
        "You are a concise evaluator. Answer in ≤60 words using bullet points. "
        "Explain the metrics with respect to the given values. Be specific, no fluff. "
        f"Counts: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}. "
        f"Core: acc={m['accuracy']}, sens={m['sensitivity']}, spec={m['specificity']}, "
        f"prec={m['precision']}, auc={m['roc_auc']}. "
        f"Derived: f1={d['f1']}, fpr={d['fpr']}, fnr={d['fnr']}, bal_acc={d['balanced_accuracy']}, prev={d['prevalence']}."
    )

def _metrics_context(dataset_id: str) -> str:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT tp, tn, fp, fn,
                   accuracy, sensitivity, specificity, precision, roc_auc,
                   computed_at
            FROM metrics
            WHERE dataset_id = ?
            ORDER BY datetime(computed_at) DESC
            LIMIT 1
        """, (dataset_id,))
        row = cur.fetchone()
    except Exception as e:
        log.error(f"metrics context db error | dataset_id={dataset_id} | err={e}", exc_info=True)
        return ""
    finally:
        conn.close()

    if not row:
        log.info(f"metrics context | dataset_id={dataset_id} | none_found")
        return ""

    (tp, tn, fp, fn,
     acc, sens, spec, prec, auc,
     when) = row

    return (
        f"Latest metrics (at {when}): "
        f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
        f"acc={acc}, sens={sens}, spec={spec}, prec={prec}, auc={auc}."
    )

def _ollama_chat_roles(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    try:
        res: ChatResponse = ollama_chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            options={"temperature": temperature},
            stream=False
        )
        msg_obj = getattr(res, "message", None)
        if msg_obj and getattr(msg_obj, "content", ""):
            return msg_obj.content or ""
        try:
            return (res["message"]["content"]) or ""
        except Exception:
            return ""
    except Exception as e:
        log.error(f"ollama chat error | err={e}", exc_info=True)
        raise HTTPException(status_code=502, detail="ollama chat error")


def _expire_dataset_db(dataset_id: str, ttl_seconds: int = 60):
    log.info(f"expire scheduled | dataset_id = {dataset_id} | ttl = {ttl_seconds}s")
    time.sleep(ttl_seconds)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM metrics WHERE dataset_id = ?", (dataset_id,))
        cur.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        conn.commit()
        log.info(f"expire done | dataset_id = {dataset_id}")
    except Exception as e:
        log.error(f"expire error | dataset_id = {dataset_id} | err = {e}" , exc_info = True)
    finally:
        conn.close()


@app.on_event("startup")
def on_startup():
    log.info("server starting")
    init_db()

@app.get("/")
def read_root():
    return{"Hello" : "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query":q}

@app.get("/products/")
def list_products(skip: int = 0, limit: int = 10):
    return{"skip" : skip, "limit": limit}

@app.get("/ping/")
def ping():
    return {"status": "ok"}

@app.get("/version/")
def version():
    return {"version": app_version}

@app.post("/csv-upload/")
def csv_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    ttl_seconds = 60):
    if file.content_type not in {"text/csv", "application/vnd.ms-excel"} or not file.filename.lower().endswith(".csv"):
        log.warning(f"upload reject | reason = bad_content_type | filename = {file.filename}")
        raise HTTPException(status_code=415, detail="Only .csv files are allowed")
    
    raw = file.file.read()
    if not raw:
        log.warning(f"upload reject | reason = empty_file | filename = {file.filename}")
        raise HTTPException(status_code = 400, detail = "Empty File")
        
    dataset_id = str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds") 

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (dataset_id, filename, content_type, raw, uploaded_at) VALUES (?,?,?,?,?)",
        (dataset_id, file.filename, file.content_type, sqlite3.Binary(raw), uploaded_at)
    )
    conn.commit()
    conn.close()

    log.info(f"upload ok | dataset_id = {dataset_id} | filename = {file.filename} | size = {len(raw)}B")

    if background_tasks is not None  and ttl_seconds > 0:
        background_tasks.add_task(_expire_dataset_db, dataset_id, ttl_seconds)


    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "detail": f"CSV stored (will auto-expire in {ttl_seconds}s)"

    }

@app.get("/datasets/")
def list_datasets():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT dataset_id, filename, uploaded_at
        FROM datasets
        ORDER BY uploaded_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    items = [{"dataset_id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]
    log.info(f"datasets list | count = {len(items)}")
    return {"items": items, "count": len(items)}
@app.get("/metrics/latest/")
def metrics_latest(dataset_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT orientation, score_orientation, tp, tn, fp, fn,
               rows_total, rows_valid, rows_invalid,
               accuracy, sensitivity, specificity, precision, roc_auc, computed_at
        FROM metrics
        WHERE dataset_id = ?
        ORDER BY datetime(computed_at) DESC
        LIMIT 1
    """, (dataset_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        log.info(f"metrics latest | dataset_id = {dataset_id} | has_metrics = False")
        return {
            "dataset_id": dataset_id,
            "has_metrics": False,
            "detail": "No metrics computed yet for this dataset"
        }

    (orientation, score_orientation, tp, tn, fp, fn,
     rows_total, rows_valid, rows_invalid,
     accuracy, sensitivity, specificity, precision, roc_auc, computed_at) = row
    
    log.info(f"metrics latest | dataset_id = {dataset_id} | at = {computed_at} | acc = {accuracy}")

    return {
        "dataset_id": dataset_id,
        "has_metrics": True,
        "computed_at": computed_at,
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_invalid": rows_invalid,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metric": {
            "name": "accuracy",
            "orientation": orientation,
            "score_orientation": score_orientation,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "roc_auc": roc_auc
        }
    }
@app.post("/evaluate/")
def evaluate(dataset_id: str, score_orientation: str = "positive"):
    if score_orientation not in ("positive", "negative"):
        log.warning(f"evaluate reject | dataset_id = {dataset_id} | reason = bad_score_orientation = {score_orientation}")
        raise HTTPException(status_code= 400, detail= "score_orientation must be positive or negative")
    
    log.info(f"evaluate start | dataset_id = {dataset_id} | score_orientation = {score_orientation}")
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT filename, content_type, raw FROM datasets WHERE dataset_id = ?", (dataset_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        log.error(f"evaluate error | dataset_id = {dataset_id} | reason = dataset_not_found")
        raise HTTPException(status_code=404, detail="dataset_id not found")

    filename, content_type, raw = row
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        conn.close()
        log.error(f"evaluate error | dataset_id = {dataset_id} | reason = no_header")
        raise HTTPException(status_code=400, detail="CSV must have a header row")

    headers = [h.strip() for h in reader.fieldnames]
    need = ["Ground Truth", "Score", "Threshold"]
    for col in need:
        if col not in headers:
            conn.close()
            log.error(f"evaluate error | dataset_id = {dataset_id} | reason = missing_column | col = {col}")
            raise HTTPException(status_code = 422, detail = f"Missing column: {col}")
        
    rows_total = 0
    rows_valid = 0
    rows_invalid = 0
    tp =  tn = fp = fn = 0

    score_label = []

    for row in reader:
        rows_total += 1

        gt = (row.get("Ground Truth") or "").strip().lower()
        s = row.get("Score")
        t = row.get("Threshold")

        try:
            s_val = float(s)
            t_val = float(t)
        except:
            rows_invalid += 1
            continue
        if gt not in ("positive", "negative"):
            rows_invalid += 1
            continue

        rows_valid += 1

        score_label.append((s_val, 1 if gt == "positive" else 0))

         
        prediction = "positive" if s_val >= t_val else "negative"

        if prediction == "positive" and gt == "positive":
            tp += 1
        elif prediction == "positive" and gt == "negative":
            fp += 1
        elif prediction == "negative" and gt == "negative":
            tn += 1
        else:
            fn += 1

    
    if rows_valid == 0:
        conn.close()
        log.warning(f"evaluate done | dataset_id = {dataset_id} | valid = 0")
        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "rows_total": rows_total,
            "rows_valid": rows_valid,
            "rows_invalid": rows_invalid,
            "detail": "No valid rows"
        }

   
    correct = tp + tn
    incorrect = fp + fn 


    accuracy = round(correct / rows_valid, 6)
    positive = tp + fn 
    sensitivity = round (tp/positive, 6  ) if positive else 0.0
    predicted_positive = tp + fp
    precision = round(tp/ predicted_positive, 6) if predicted_positive else 0.0
    negative = tn + fp 
    specificity = round(tn/negative, 6) if negative else 0.0

    def _roc_auc_from_scores(pairs):
        n = len(pairs)
        n_pos = sum(lbl for _, lbl in pairs)
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0

        
        sorted_pairs = sorted(pairs, key=lambda x: x[0])

        
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[k] = avg_rank
            i = j + 1

       
        sum_pos_ranks = sum(r for r, (_, lbl) in zip(ranks, sorted_pairs) if lbl == 1)
        auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
        return auc

    raw_auc = _roc_auc_from_scores(score_label)
    roc_auc = 1.0 - raw_auc if score_orientation == "negative" else raw_auc
    roc_auc = round(roc_auc, 6)

    computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds") 
    orientation = "score >= threshold: Positive"
    cur.execute("""
        INSERT INTO metrics (
            dataset_id, orientation, score_orientation,
            tp, tn, fp, fn, rows_total, rows_valid, rows_invalid,
            accuracy, sensitivity, specificity, precision, roc_auc, computed_at
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        dataset_id, orientation, score_orientation,
        tp, tn, fp, fn, rows_total, rows_valid, rows_invalid,
        accuracy, sensitivity, specificity, precision, roc_auc, computed_at
    ))
    conn.commit()
    conn.close()

    log.info(
        "evaluate done |"
        f"dataset_id = {dataset_id} | valid = {rows_valid}/{rows_total} |"
        f"tp = {tp} tn = {tn} fp = {fp} fn = {fn} |"
        f"acc  = {accuracy}  sens = {sensitivity} spec = {specificity} prec = {precision} auc = {roc_auc} "
    )        

    return {
        "dataset_id": dataset_id,
        "filename":  filename,
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_invalid": rows_invalid,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metric": {
            "name": "accuracy",
            "orientation": "score >= threshold: Positive",
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "precision": precision,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "score_orientation": score_orientation
        }
    } 
@app.post("/ollama/chat-roles/")
def ollama_chat_roles(
    user_prompt: str,
    dataset_id: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3
):
    sys_prompt = GENERIC_SYS_PROMPT
    if dataset_id:
        ctx = _metrics_context(dataset_id)
        if ctx:
            sys_prompt = GENERIC_SYS_PROMPT + " " + ctx

    log.info(
        "ollama chat start | "
        f"model={model} | temp={temperature} | dataset_id={dataset_id or '(none)'}"
    )

    reply = _ollama_chat_roles(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature
    )

    log.info(f"ollama chat ok | len={len(reply)}")
    return {"model": model, "dataset_id": dataset_id, "reply": reply}
@app.post("/agent/analyze/")
def agent_analyze(
    dataset_id: str,
    question: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3
):
    if not dataset_id:
        raise HTTPException(status_code=422, detail="dataset_id required")

    m = _latest_metrics_row(dataset_id)
    if not m:
        log.warning(f"agent analyze | dataset_id={dataset_id} | no_metrics")
        raise HTTPException(status_code=404, detail="No metrics found for dataset_id")

    d = _derived_from_confusion(m["tp"], m["tn"], m["fp"], m["fn"])
    sys_prompt = _agent_system_prompt_basic(m, d)
    user_prompt = (question or "Give a short analysis of the evaluation and why the metrics look like this.").strip()

    log.info(
        "agent analyze start | "
        f"dataset_id={dataset_id} | model={model} | temp={temperature}"
    )

    reply = _ollama_chat_roles(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature
    )

    log.info(f"agent analyze ok | len={len(reply)}")
    return {
        "dataset_id": dataset_id,
        "computed_at": m["computed_at"],
        "core": {
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "roc_auc": m["roc_auc"]
        },
        "counts": {"tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"]},
        "derived": d,
        "model": model,
        "answer": reply
    }
@app.post("/agent/web-assist/")
def agent_web_assist(
    question: str,
    dataset_id: str | None = None,
    web_query: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3
):
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="question required")

    m = None
    d = None
    if dataset_id:
        m = _latest_metrics_row(dataset_id)
        if m:
            d = _derived_from_confusion(m["tp"], m["tn"], m["fp"], m["fn"])

    wq = (web_query or q).strip()
    web_ctx = _web_context_wikipedia(wq)

    sys_prompt = _agent_web_system_prompt_full(m, d, web_ctx)

    log.info(
        "agent web assist start | "
        f"model={model} | temp={temperature} | dataset_id={dataset_id or '(none)'} | web_q='{wq}' "
        f"| have_local={'yes' if m else 'no'} | have_web={'yes' if web_ctx else 'no'}"
    )

    reply = _ollama_chat_roles(
        system_prompt=sys_prompt,
        user_prompt=q,
        model=model,
        temperature=temperature
    )

    log.info(f"agent web assist ok | len={len(reply)}")
    return {
        "dataset_id": dataset_id,
        "web_query": wq,
        "model": model,
        "answer": reply
    }
