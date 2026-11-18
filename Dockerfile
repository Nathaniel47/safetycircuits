# Feature browser only: no torch, no model. GPU work runs in Kaggle/Colab notebooks.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/browser.py", "--server.port=8501", "--server.address=0.0.0.0"]
