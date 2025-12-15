# Fake News Detector – DistilBERT

A lightweight fake news detection system built with **DistilBERT**, featuring a
clean web interface and fully containerized deployment.

This project was developed as part of the *Deep Learning and Software Engineering*
course at the **Universidad Politécnica de Madrid (UPM)**.

---

## What it does

- Classifies news articles as **Real** or **Fake**
- Displays confidence scores and interpretability categories
- Stores predictions locally using **SQLite**
- Provides filtering and statistics over past predictions
- Runs entirely on **CPU** (no GPU required)

---

## Tech Stack

- **Model:** DistilBERT (fine-tuned)
- **Frameworks:** PyTorch, Hugging Face Transformers
- **UI:** Streamlit
- **Persistence:** SQLite
- **Deployment:** Docker

---

## Quick Start (Docker)

The easiest way to run the app:

docker build -t fake-news-detector .
docker run --rm -p 8501:8501 fake-news-detector

Open in your browser:
http://localhost:8501

---

## Local Setup (Optional)

- python -m venv .venv
- source .venv/bin/activate # Windows: ..venv\Scripts\Activate.ps1
- pip install -r requirements.txt
- export PYTHONPATH=. # Windows: $env:PYTHONPATH="."
- streamlit run app/ui.py

---

## Project Structure

- app/ # Streamlit UI, inference, database logic
- models/best/ # Final trained DistilBERT model
- notebooks/ # Training & experimentation
- data/ # SQLite database (auto-generated)
- Dockerfile

---

## Notes

- The model expects **text input** (title + body). Avoid pasting only URLs.
- All predictions are stored locally in `data/predictions.sqlite3`.
- The Docker image uses **CPU-only PyTorch** for maximum compatibility.

---

## Academic Context

This project demonstrates the full lifecycle of a deep learning system:
**training, evaluation, deployment, and usability**.

Developed at the Universidad Politécnica de Madrid (UPM) by students:
- Birk Bregendahl
- Karol Swiderski
- Agustin Tamagnone