FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLECORS=false

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only torch once
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install only app dependencies
COPY requirements.app.txt .
RUN pip install --no-cache-dir -r requirements.app.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
