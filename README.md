# API de atajadas

## Ejecutar con Docker
    docker build -t atajadas-api .
    docker run -p 8000:8000 -e API_KEY=mi_clave atajadas-api

## Ejecutar sin Docker (requiere ffmpeg)
    pip install torch torchvision -r requirements.txt
    uvicorn api_atajadas:app --host 0.0.0.0 --port 8000

## Uso
    # bytes del video en el cuerpo
    curl -X POST "http://localhost:8000/analizar" -H "Content-Type: application/octet-stream" \
         -H "X-API-Key: mi_clave" --data-binary @carrera.mp4

    # formulario multipart
    curl -X POST "http://localhost:8000/analizar?formato=resumen" -H "X-API-Key: mi_clave" -F "video=@carrera.mp4"

    # recibir directamente el MP4 procesado (puntaje y tipo en los headers)
    curl -X POST "http://localhost:8000/analizar?formato=video" -H "X-API-Key: mi_clave" \
         --data-binary @carrera.mp4 -D headers.txt -o carrera_ANOTADO.mp4

Documentación interactiva: http://localhost:8000/docs

Variables de entorno: MODEL_PATH, API_KEY, MAX_MB (300), MAX_DURACION_S (600), MAX_ANCHO (1280), DET_THRESHOLD, CORS_ORIGINS, PORT.
