# Fake News Detection with DistilBERT

This project is part of the *Deep Learning and Software Engineering* course at UPM.  
Our goal is to classify news articles as 'real' or 'fake' using a fine-tuned DistilBERT model.

# Fake News Detector – DistilBERT (M3)

This project provides a Dockerized DistilBERT-based Fake News Detector with a simple web UI (Streamlit).
The Docker image runs fully on CPU and is compatible with any machine with Docker installed.

Requirements
1. Docker (tested with Docker Desktop)
2. No GPU required


Steps to run with Docker (recommended):

1. Build the image:
docker build -t fake-news-detector .

2. Run the container:
docker run --rm -p 8501:8501 fake-news-detector

3. Open in browser:
http://localhost:8501


To run locally (without Docker):

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=.        # Windows PowerShell: $env:PYTHONPATH="."
streamlit run app/ui.py


Project structure:
app/            # inference + Streamlit UI
models/best/    # trained DistilBERT model (final configuration)
notebooks/      # training & experiments (M2)
Dockerfile      # Docker image definition
requirements.txt


Notes for team members:

- The UI can be freely modified in app/ui.py
- Do NOT commit .venv/
- After UI changes, rebuild Docker image:
    docker build -t fake-news-detector .


Sharing Docker image (demo compatibility test)

Export image:
docker save fake-news-detector -o fake-news-detector.tar

Load on another machine:
docker load -i fake-news-detector.tar
docker run --rm -p 8501:8501 fake-news-detector