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
