FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# Para servidor con GPU, cambia la línea siguiente por: pip install torch torchvision
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
COPY api_atajadas.py .
COPY modelo/ modelo/
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api_atajadas:app --host 0.0.0.0 --port ${PORT}"]
