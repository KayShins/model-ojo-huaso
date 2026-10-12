FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
COPY api_atajadas.py .
RUN mkdir -p modelo && \
    wget -O modelo/atajada_r2plus1d.pt \
    https://huggingface.co/spaces/Fibonhacci/ojo-huaso-modelo/resolve/main/atajada_r2plus1d.pt
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api_atajadas:app --host 0.0.0.0 --port ${PORT}"]
