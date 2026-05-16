# CSV Evaluator & AI Assistant

A beginner-friendly AI-powered evaluation platform built using FastAPI, Streamlit, SQLite3, and Ollama.

This project allows users to upload CSV datasets, compute classification evaluation metrics, visualize results, and interact with AI agents that explain the metrics in simple language.

---

# Features

- Upload `.csv` datasets
- Automatic dataset storage using SQLite3
- Compute:
  - Accuracy
  - Sensitivity (TPR)
  - Specificity (TNR)
  - Precision (PPV)
  - ROC AUC
- Confusion Matrix generation
- Streamlit dashboard UI
- AI-powered metric explanations using Ollama
- Web-assisted AI agent using Wikipedia context
- Dataset auto-expiration with background tasks
- Logging using `RotatingFileHandler`
- Beginner-friendly FastAPI architecture

---

# Tech Stack

## Backend
- Python
- FastAPI
- SQLite3
- Ollama
- Requests

## Frontend
- Streamlit

## AI Models
- Ollama local LLMs
- Default model: `gemma3:4b`

---

# Project Structure

```bash
project/
│
├── main2.py                  # FastAPI backend
├── csv_evaluator frtend.py   # Streamlit frontend
├── db.py                     # Database initialization
├── data.db                   # SQLite database
├── logs/
│   ├── app.log
│   └── ui.log
```

---

# API Features

## Dataset Management
- Upload CSV datasets
- Store datasets in SQLite
- Auto-delete datasets after a configurable TTL

---

## Evaluation Engine

Computes:
- TP / TN / FP / FN
- Accuracy
- Sensitivity
- Specificity
- Precision
- ROC AUC

---

## AI Assistance

### 1. Local Metrics Agent
Explains evaluation metrics based on uploaded datasets.

### 2. Web-Assisted Agent
Combines:
- Dataset metrics
- Wikipedia context
- Local LLM reasoning

---

# Required CSV Format

Your CSV file must contain these columns:

```csv
Ground Truth,Score,Threshold
```

Example:

```csv
Ground Truth,Score,Threshold
positive,0.91,0.50
negative,0.22,0.50
positive,0.87,0.50
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <project-folder>
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install fastapi "uvicorn[standard]" streamlit requests ollama
```

---

# Install Ollama

Download and install Ollama from:

https://ollama.com/download

Pull the model:

```bash
ollama pull gemma3:4b
```

---

# Running the Backend

```bash
uvicorn main2:app --reload
```

Backend runs at:

```bash
http://127.0.0.1:8000
```

Swagger Docs:

```bash
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

```bash
streamlit run "csv_evaluator frtend.py"
```

---

# Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/csv-upload/` | POST | Upload dataset |
| `/evaluate/` | POST | Compute metrics |
| `/metrics/latest/` | GET | Fetch latest metrics |
| `/datasets/` | GET | List datasets |
| `/ollama/chat-roles/` | POST | Local AI assistant |
| `/agent/analyze/` | POST | Dataset analysis agent |
| `/agent/web-assist/` | POST | Web-assisted AI agent |

---

# Logging

Two rotating log files are created automatically:

```bash
logs/app.log
logs/ui.log
```

Features:
- Timestamped logs
- Error tracking
- Request monitoring
- Dataset lifecycle logging

---

# Example Workflow

1. Start FastAPI backend
2. Start Streamlit frontend
3. Upload CSV dataset
4. Compute metrics
5. Ask AI questions like:
   - "Why is ROC AUC low?"
   - "Explain sensitivity"
   - "What do FP and FN mean?"

---

# Screenshots

You can add screenshots here later:

```md
![Dashboard](images/dashboard.png)
```

---

# Future Improvements

- Graph visualizations
- Authentication system
- Multi-user support
- PDF report generation
- Model comparison
- Live streaming responses
- Docker deployment
- Cloud deployment

---

# Learning Objectives

This project demonstrates:
- FastAPI backend development
- REST API design
- Streamlit UI development
- SQLite integration
- Logging systems
- Background tasks
- AI/LLM integration
- Evaluation metric computation

---

# References

- Backend implementation based on FastAPI endpoints and metric computation logic from uploaded project files.
- Frontend implementation based on the Streamlit dashboard and AI assistant workflow.
- Database schema and initialization based on SQLite setup.

