# api_atajadas.py — API del detector de atajadas (generado por el notebook detector_atajadas_rodeo.ipynb)
# Ejecutar:  MODEL_PATH=modelo/atajada_r2plus1d.pt uvicorn api_atajadas:app --host 0.0.0.0 --port 8000
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import os, re, io, sys, json, math, time, random, hashlib, subprocess, unicodedata, base64, tempfile, shutil, threading
from pathlib import Path
import numpy as np
import cv2
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

# ---------- Video / modelo ----------
TARGET_FPS = 10        # cuadros por segundo que ve el modelo
CLIP_LEN = 16          # cuadros por clip (16 a 10 fps = 1,6 s de movimiento)
FRAME_SIZE = 128       # tamaño al que se reduce cada cuadro
CROP_SIZE = 112        # recorte de entrada a la red
TYPE_RADIUS_S = 0.4    # radio (s) alrededor del momento usado para decidir el tipo de atajada

# ---------- Análisis de videos ----------
DET_THRESHOLD = 0.35   # probabilidad mínima para afirmar que hay atajada
HIGHLIGHT_S = 0.6      # segundos antes/después del momento que se marcan en rojo
SLOWMO_WINDOW_S = 1.5  # repetición en cámara lenta: ± segundos alrededor de la atajada
SLOWMO_FACTOR = 4      # 4 = velocidad 1/4
FREEZE_S = 1.5         # congelar el cuadro exacto de la atajada

KB_PATH = os.environ.get('KB_PATH', '')  # opcional: ruta al skills.md; si está vacío se busca en /content y en MyDrive
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm', '.mpg', '.mpeg', '.3gp', '.mts'}

# 1.3 Base de conocimiento: escala de puntaje de atajadas (compendio FEROCHI §6.1)
REGLAS = {
    'costilla':  {'puntos': 4, 'nombre': 'Atajada de costilla libre (de íjar)'},
    'paleta':    {'puntos': 3, 'nombre': 'Atajada de paleta libre (de mitad)'},
    'cogote':    {'puntos': 2, 'nombre': 'Atajada de cogote libre'},
    'envoltura': {'puntos': 0, 'nombre': 'Envoltura (no se define la atajada)'},
}
TIPOS = list(REGLAS)
FUENTE_REGLAS = 'Compendio Normativo FEROCHI 2026-2027, §6.1 Escala de Valoración de Atajadas'

def sin_tildes(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

PALABRAS_TIPO = {
    'costilla':  ['costilla', 'ijar'],
    'paleta':    ['paleta', 'mitad'],
    'cogote':    ['cogote', 'pescuezo'],
    'envoltura': ['envoltura', 'envuelta', 'envolvio'],
}

def cargar_reglas_md(path):
    # Lee líneas tipo '**Atajada de Costilla Libre (de íjar)**: 4 puntos buenos.' y actualiza REGLAS
    texto = Path(path).read_text(encoding='utf-8')
    n = 0
    for m in re.finditer(r'\*\*([^*]+)\*\*\s*:\s*(\d+)\s*punto', texto):
        etiqueta, pts = m.group(1).strip(), int(m.group(2))
        e = sin_tildes(etiqueta).lower()
        for tipo, claves in PALABRAS_TIPO.items():
            if any(k in e for k in claves):
                REGLAS[tipo] = {'puntos': pts, 'nombre': etiqueta}
                n += 1
                break
    return n

_kb = KB_PATH or next((p for p in ['/content/skills.md', '/content/drive/MyDrive/skills.md'] if os.path.exists(p)), '')
if _kb and os.path.exists(_kb):
    print(f'Base de conocimiento leída de {_kb}: {cargar_reglas_md(_kb)} reglas de atajada.')
else:
    print('Usando la escala de atajadas del compendio incluida en el notebook.')
for t in TIPOS:
    print(f"  {REGLAS[t]['nombre']:<40} -> {REGLAS[t]['puntos']} pts")

# 1.4 Utilidades de video
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)   # normalización Kinetics
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)

def leer_video(path, fps_obj=None, size=None):
    # Devuelve cuadros RGB (N,size,size,3) muestreados a fps_obj, sus tiempos (s) y la duración total
    fps_obj = fps_obj or TARGET_FPS; size = size or FRAME_SIZE
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    frames, times = [], []
    i, prox, paso = 0, 0.0, 1.0 / fps_obj
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        if t + 1e-6 >= prox:
            fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
            frames.append(fr); times.append(t)
            while prox <= t + 1e-6:
                prox += paso
        i += 1
    cap.release()
    if not frames:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros(0, np.float32), 0.0
    return np.stack(frames), np.array(times, np.float32), i / fps

def indices_clip(n, c, L=None):
    # índices de un clip de L cuadros centrado en c (se repiten los bordes)
    L = L or CLIP_LEN
    return np.clip(np.arange(c - L // 2, c - L // 2 + L), 0, n - 1)

def preparar_clip(clip, train=False):
    # (L,H,W,3) uint8 -> tensor (3,L,CROP,CROP) normalizado; con aumentos si train
    L, H, W, _ = clip.shape
    if train:
        y0 = np.random.randint(0, H - CROP_SIZE + 1); x0 = np.random.randint(0, W - CROP_SIZE + 1)
    else:
        y0 = (H - CROP_SIZE) // 2; x0 = (W - CROP_SIZE) // 2
    x = clip[:, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE].astype(np.float32) / 255.0
    if train:
        if np.random.rand() < 0.5:          # atajadas hacia ambos lados de la medialuna
            x = x[:, :, ::-1]
        x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2)))

# 1.5 Modelo: red de video 3D que aprende el movimiento
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class AtajadaNet(nn.Module):
    # det  -> probabilidad de que el centro del clip sea el instante de la atajada
    # tipo -> costilla / paleta / cogote / envoltura
    def __init__(self, n_tipos, pretrained=True):
        super().__init__()
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.drop = nn.Dropout(0.4)
        self.det = nn.Linear(feat, 1)
        self.tipo = nn.Linear(feat, n_tipos)

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.det(f).squeeze(1), self.tipo(f)

@torch.no_grad()
def curva_atajada(model, frames, times, stride=2, batch=8):
    # Recorre todo el video con una ventana deslizante y devuelve la probabilidad de atajada en cada instante
    model.eval()
    n = len(frames)
    centros = np.arange(0, n, stride)
    P, T = [], []
    for s in range(0, len(centros), batch):
        x = torch.stack([preparar_clip(np.asarray(frames[indices_clip(n, c)])) for c in centros[s:s + batch]]).to(DEVICE)
        with torch.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
            det, tip = model(x)
        P.append(torch.sigmoid(det.float()).cpu().numpy())
        T.append(torch.softmax(tip.float(), 1).cpu().numpy())
    return times[centros], np.concatenate(P), np.concatenate(T)

def suavizar(p, k=3):
    if len(p) < k:
        return p
    return np.convolve(np.pad(p, k // 2, mode='edge'), np.ones(k) / k, mode='valid')

def localizar(ts, p, tp, radio=None):
    # Momento de máxima probabilidad + probabilidades de tipo promediadas alrededor de ese momento
    radio = radio or TYPE_RADIUS_S
    ps = suavizar(p)
    j = int(np.argmax(ps))
    cerca = np.abs(ts - ts[j]) <= radio
    w = ps[cerca][:, None]
    tipo_prob = (tp[cerca] * w).sum(0) / max(float(w.sum()), 1e-6)
    return {'t': float(ts[j]), 'prob': float(ps[j]), 'tipo_prob': tipo_prob, 'curva': ps}

def cargar_modelo(ckpt_path):
    global model, TIPOS, TIPOS_ENTRENADOS, TARGET_FPS, CLIP_LEN, FRAME_SIZE, CROP_SIZE
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    TIPOS = ck['tipos']; TIPOS_ENTRENADOS = ck['tipos_entrenados']; REGLAS.update(ck['reglas'])
    TARGET_FPS, CLIP_LEN, FRAME_SIZE, CROP_SIZE = (ck['config'][k] for k in ('TARGET_FPS', 'CLIP_LEN', 'FRAME_SIZE', 'CROP_SIZE'))
    model = AtajadaNet(len(TIPOS), pretrained=False).to(DEVICE)
    model.load_state_dict(ck['state_dict']); model.eval()
    ck.pop('state_dict', None)
    import gc; gc.collect()
    return ck

# 1.6 Análisis de un video: puntaje + video anotado con el momento de la atajada
import matplotlib.pyplot as plt

def _txt(img, texto, org, escala, color, grosor=2, fondo=(0, 0, 0)):
    texto = sin_tildes(texto)
    (w, h), b = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + b + 4), fondo, -1)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)

def _franja(W, alto, ts, curva, dur, t_at):
    img = np.full((alto, W, 3), 30, np.uint8)
    X = lambda t: int(np.clip(t / max(dur, 1e-6) * (W - 1), 0, W - 1))
    Y = lambda p: int(alto - 8 - p * (alto - 28))
    cv2.line(img, (0, Y(DET_THRESHOLD)), (W - 1, Y(DET_THRESHOLD)), (90, 90, 90), 1)
    pts = np.array([[X(t), Y(p)] for t, p in zip(ts, curva)], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)
    if t_at is not None:
        cv2.line(img, (X(t_at), 0), (X(t_at), alto - 1), (0, 0, 255), 2)
    _txt(img, 'Prob. de atajada', (8, 18), 0.45, (220, 220, 220), 1, fondo=(30, 30, 30))
    return img, X

def anotar_video(path, res, out_path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps >= 1 else 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    esc = max(W / 1280, 0.45)
    alto_f = max(int(90 * esc), 60)
    t_at = res['t'] if res['detectada'] else None
    franja, X = _franja(W, alto_f, res['ts'], res['curva'], res['duracion'], t_at)
    if not res['detectada']:
        titulo = 'Sin atajada clara'
    elif res['puntos'] is None:
        titulo = 'Atajada detectada (tipo no entrenado)'
    else:
        titulo = f"{res['nombre']}: {res['puntos']} pts"
    g = max(1, int(2 * esc))

    def componer(fr, t, repeticion=False):
        img = fr.copy()
        cerca = res['detectada'] and abs(t - res['t']) <= HIGHLIGHT_S
        if cerca:
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 0, 255), max(4, int(12 * esc)))
            _txt(img, f"ATAJADA  t = {res['t']:.1f} s", (int(20 * esc), H - int(30 * esc)), 1.3 * esc, (255, 255, 255), max(2, int(3 * esc)), fondo=(0, 0, 200))
        _txt(img, titulo, (int(20 * esc), int(45 * esc)), 0.9 * esc, (255, 255, 255), g)
        _txt(img, f't = {t:5.2f} s', (int(20 * esc), int(90 * esc)), 0.7 * esc, (200, 200, 200), g)
        if repeticion:
            _txt(img, f'REPETICION 1/{SLOWMO_FACTOR}', (W - int(330 * esc), int(45 * esc)), 0.8 * esc, (0, 255, 255), g)
        f = franja.copy()
        cv2.line(f, (X(t), 0), (X(t), alto_f - 1), (255, 255, 255), 1)
        return np.vstack([img, f])

    tmp = out_path + '.tmp.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H + alto_f))
    clave, mejor, i = None, 1e9, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        lienzo = componer(fr, t)
        vw.write(lienzo)
        if t_at is not None and abs(t - t_at) < mejor:
            mejor, clave = abs(t - t_at), lienzo.copy()
        i += 1
    # Repetición en cámara lenta con cuadro congelado en el instante exacto
    if t_at is not None:
        j = max(0, int((t_at - SLOWMO_WINDOW_S) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        congelado = False
        while j / fps <= t_at + SLOWMO_WINDOW_S:
            ok, fr = cap.read()
            if not ok:
                break
            lienzo = componer(fr, j / fps, repeticion=True)
            reps = SLOWMO_FACTOR
            if not congelado and j / fps >= t_at - 0.5 / fps:
                reps += int(FREEZE_S * fps); congelado = True
            for _ in range(reps):
                vw.write(lienzo)
            j += 1
    cap.release(); vw.release()
    try:   # H.264 para que se vea en el navegador
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '23', out_path], check=True)
        os.remove(tmp)
    except Exception as e:
        print('ffmpeg no disponible, se deja el video en mp4v:', e)
        os.replace(tmp, out_path)
    return clave

def analizar_video(path, mostrar=True, out_dir=None):
    frames, times, dur = leer_video(path)
    if len(frames) == 0:
        raise ValueError(f'No se pudo leer el video: {path}')
    ts, p, tp = curva_atajada(model, frames, times)
    loc = localizar(ts, p, tp)
    detectada = loc['prob'] >= DET_THRESHOLD
    tipo, conf_tipo, probs_tipo = None, None, {}
    if TIPOS_ENTRENADOS:
        idx = [TIPOS.index(t) for t in TIPOS_ENTRENADOS]
        pt = loc['tipo_prob'][idx]; pt = pt / max(pt.sum(), 1e-9)
        probs_tipo = {t: float(v) for t, v in zip(TIPOS_ENTRENADOS, pt)}
        tipo = max(probs_tipo, key=probs_tipo.get); conf_tipo = probs_tipo[tipo]
    res = {
        'video': str(path), 'duracion': float(dur), 'detectada': bool(detectada),
        't': loc['t'], 'prob': loc['prob'], 'ts': ts, 'curva': loc['curva'],
        'tipo': tipo if detectada else None, 'conf_tipo': conf_tipo, 'probs_tipo': probs_tipo,
        'nombre': REGLAS[tipo]['nombre'] if (detectada and tipo) else None,
        'puntos': (REGLAS[tipo]['puntos'] if tipo else None) if detectada else 0,
    }
    base = Path(path).stem
    out_dir = out_dir or f'{OUTPUT_DIR}/resultados'
    os.makedirs(out_dir, exist_ok=True)
    out_video = f'{out_dir}/{base}_ANOTADO.mp4'
    clave = anotar_video(path, res, out_video)

    # Justificación
    lineas = []
    if not detectada:
        lineas.append(f"No se detectó una atajada clara (máxima probabilidad {loc['prob']:.0%} < umbral {DET_THRESHOLD:.0%}).")
        lineas.append('Puntaje asignado por atajada: 0.')
    else:
        lineas.append(f"Momento de la atajada: {res['t']:.1f} s (confianza de detección {res['prob']:.0%}).")
        if tipo:
            lineas.append(f"Clasificación: {res['nombre']} -> {res['puntos']} puntos buenos (confianza {conf_tipo:.0%}).")
            lineas.append(f"Fundamento: {FUENTE_REGLAS}: «{res['nombre']}» = {res['puntos']} puntos buenos.")
            lineas.append('Probabilidad por tipo: ' + ', '.join(f"{t} {v:.0%}" for t, v in sorted(probs_tipo.items(), key=lambda kv: -kv[1])))
            if conf_tipo < 0.5:
                lineas.append('⚠ Confianza baja en el tipo de atajada: revisa la repetición en cámara lenta.')
        else:
            lineas.append('El modelo no fue entrenado con el tipo de atajada (carpetas sin costilla/paleta/cogote/envoltura), no puede asignar puntaje.')
    lineas.append('Nota: se evalúa solo la atajada; la salida del apiñadero (+1) y las faltas (tijera, ayuda, etc.) no se califican aquí.')
    res['justificacion'] = '\n'.join(lineas)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(ts, loc['curva'], color='tab:orange', label='prob. de atajada')
    ax.axhline(DET_THRESHOLD, color='gray', ls='--', lw=1, label='umbral')
    if detectada:
        ax.axvline(res['t'], color='red', lw=2, label=f"atajada {res['t']:.1f} s")
        ax.axvspan(res['t'] - HIGHLIGHT_S, res['t'] + HIGHLIGHT_S, color='red', alpha=0.12)
    ax.set_xlabel('segundos'); ax.set_ylim(0, 1); ax.legend(loc='upper left'); ax.set_title(base)
    fig.tight_layout(); fig.savefig(f'{out_dir}/{base}_curva.png', dpi=110)
    img_path = None
    if clave is not None:
        img_path = f'{out_dir}/{base}_momento_atajada.jpg'
        cv2.imwrite(img_path, clave)
    res['archivos'] = {'video': out_video, 'imagen': img_path, 'curva': f'{out_dir}/{base}_curva.png'}
    guardable = {k: v for k, v in res.items() if k not in ('ts', 'curva')}
    Path(f'{out_dir}/{base}_resultado.json').write_text(json.dumps(guardable, ensure_ascii=False, indent=2))

    if mostrar:
        from IPython.display import display, HTML, Image as IPImage
        print('=' * 70)
        print(f"PUNTAJE DE LA ATAJADA: {res['puntos'] if res['puntos'] is not None else 's/d'}")
        print('=' * 70)
        print(res['justificacion'])
        plt.show()
        if clave is not None:
            print('Cuadro exacto de la atajada:')
            display(IPImage(filename=f'{out_dir}/{base}_momento_atajada.jpg', width=720))
        if os.path.getsize(out_video) < 45 * 1024 * 1024:
            b64 = base64.b64encode(open(out_video, 'rb').read()).decode()
            display(HTML(f'<video width="760" controls src="data:video/mp4;base64,{b64}"></video>'))
        else:
            print('Video anotado (muy pesado para mostrarlo aquí):', out_video)
        print('Archivos guardados en:', out_dir)
    else:
        plt.close(fig)
    return res

# =============================== API ===============================
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Query, Header
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

OUTPUT_DIR = tempfile.gettempdir()
MODEL_PATH = os.environ.get('MODEL_PATH', str(Path(__file__).resolve().parent / 'modelo' / 'atajada_r2plus1d.pt'))
API_KEY = os.environ.get('API_KEY', '')                       # si se define, se exige el header X-API-Key
MAX_MB = float(os.environ.get('MAX_MB', '300'))               # tamaño máximo del video recibido
MAX_DURACION_S = float(os.environ.get('MAX_DURACION_S', '600'))
MAX_ANCHO = int(os.environ.get('MAX_ANCHO', '1280'))          # ancho máximo tras normalizar el video
DET_THRESHOLD = float(os.environ.get('DET_THRESHOLD', DET_THRESHOLD))

import sqlite3
from datetime import datetime

HISTORIAL_DIR = Path(os.environ.get('HISTORIAL_DIR', str(Path(__file__).resolve().parent / 'historial')))
HISTORIAL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = HISTORIAL_DIR / 'historial.db'


def _conectar_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _inicializar_db():
    con = _conectar_db()
    con.execute('''
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            nombre_video TEXT,
            detectada INTEGER,
            puntaje INTEGER,
            tipo_atajada TEXT,
            nombre_atajada TEXT,
            momento_segundos REAL,
            confianza_deteccion REAL,
            confianza_tipo REAL,
            justificacion TEXT
        )
    ''')
    con.commit()
    con.close()


_inicializar_db()


def _guardar_en_historial(nombre_video, resultado, video_bytes, img_bytes):
    con = _conectar_db()
    cur = con.execute(
        '''INSERT INTO historial
           (fecha, nombre_video, detectada, puntaje, tipo_atajada, nombre_atajada,
            momento_segundos, confianza_deteccion, confianza_tipo, justificacion)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (datetime.utcnow().isoformat(), nombre_video, int(resultado['detectada']), resultado['puntaje'],
         resultado['tipo_atajada'], resultado['nombre_atajada'], resultado['momento_segundos'],
         resultado['confianza_deteccion'], resultado['confianza_tipo'], resultado['justificacion']),
    )
    con.commit()
    registro_id = cur.lastrowid
    con.close()
    (HISTORIAL_DIR / f'{registro_id}_video.mp4').write_bytes(video_bytes)
    if img_bytes:
        (HISTORIAL_DIR / f'{registro_id}_imagen.jpg').write_bytes(img_bytes)
    return registro_id
    
CK = cargar_modelo(MODEL_PATH)
_lock = threading.Lock()                                      # una inferencia a la vez en la GPU


def normalizar_video(data, nombre, carpeta):
    # Bytes recibidos -> MP4 H.264 legible por OpenCV, con la orientación corregida y ancho <= MAX_ANCHO
    ext = Path(nombre or '').suffix.lower()
    crudo = os.path.join(carpeta, 'entrada' + (ext if ext in VIDEO_EXT else '.bin'))
    with open(crudo, 'wb') as f:
        f.write(data)
    destino = os.path.join(carpeta, 'video.mp4')
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', crudo, '-an',
                        '-vf', f"scale='trunc(min({MAX_ANCHO},iw)/2)*2':-2",
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20', destino],
                       check=True, timeout=900)
    except FileNotFoundError:            # sin ffmpeg instalado: se intenta leer el archivo tal cual
        destino = crudo
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ValueError('El archivo recibido no es un video válido o su formato no es compatible.')
    cap = cv2.VideoCapture(destino)
    ok, _ = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if not ok:
        raise ValueError('No se pudo leer ningún cuadro del video.')
    if fps > 0 and n / fps > MAX_DURACION_S:
        raise ValueError(f'El video dura {n / fps:.0f} s; el máximo permitido es {MAX_DURACION_S:.0f} s.')
    return destino


def procesar_video_bytes(data, nombre='video.mp4'):
    # Función principal: bytes de un video -> (resultado con puntaje y tipo, video procesado, imagen de la atajada)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        path = normalizar_video(data, nombre, td)
        with _lock:
            res = analizar_video(path, mostrar=False, out_dir=td)
        video = Path(res['archivos']['video']).read_bytes()
        img = Path(res['archivos']['imagen']).read_bytes() if res['archivos']['imagen'] else None
    resultado = {
        'detectada': bool(res['detectada']),
        'puntaje': res['puntos'],
        'tipo_atajada': res['tipo'],
        'nombre_atajada': res['nombre'],
        'momento_segundos': round(float(res['t']), 2) if res['detectada'] else None,
        'confianza_deteccion': round(float(res['prob']), 4),
        'confianza_tipo': round(float(res['conf_tipo']), 4) if (res['detectada'] and res['conf_tipo'] is not None) else None,
        'probabilidades_tipo': {k: round(float(v), 4) for k, v in res['probs_tipo'].items()},
        'justificacion': res['justificacion'],
        'fundamento': FUENTE_REGLAS,
        'duracion_video_s': round(float(res['duracion']), 2),
        'tiempo_proceso_s': round(time.time() - t0, 2),
    }
    return resultado, video, img


app = FastAPI(title='API de atajadas – Rodeo chileno', version='1.0',
              description='Recibe un video (bytes) de una carrera, detecta la atajada y devuelve puntaje, tipo y video procesado.')
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
                   allow_methods=['*'], allow_headers=['*'],
                   expose_headers=['X-Puntaje', 'X-Tipo-Atajada', 'X-Momento-Segundos', 'X-Detectada', 'X-Resultado'])


def _verificar_clave(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, 'API key inválida o ausente (header X-API-Key).')


@app.get('/salud')
def salud():
    return {'estado': 'ok', 'dispositivo': DEVICE, 'tipos_puntuables': TIPOS_ENTRENADOS,
            'escala': {t: REGLAS[t] for t in TIPOS}, 'umbral_deteccion': DET_THRESHOLD,
            'metricas_validacion': {k: v for k, v in CK.get('metricas', {}).items() if isinstance(v, (int, float))}}


@app.post('/analizar')
async def analizar(request: Request,
                   formato: str = Query('json', description='json | resumen | video'),
                   x_api_key: Optional[str] = Header(None),
                   x_filename: Optional[str] = Header(None)):
    _verificar_clave(x_api_key)
    if formato not in ('json', 'resumen', 'video'):
        raise HTTPException(400, "formato debe ser 'json', 'resumen' o 'video'.")
    if request.headers.get('content-type', '').lower().startswith('multipart/form-data'):
        form = await request.form()
        archivo = form.get('video') or next((v for v in form.values() if hasattr(v, 'read')), None)
        if archivo is None or not hasattr(archivo, 'read'):
            raise HTTPException(400, "Envía el video en el campo 'video' del formulario.")
        data, nombre = await archivo.read(), (archivo.filename or 'video.mp4')
    else:
        data, nombre = await request.body(), (x_filename or 'video.mp4')
    if not data:
        raise HTTPException(400, 'No se recibió ningún video (cuerpo vacío).')
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f'El video supera el máximo de {MAX_MB:.0f} MB.')
    try:
        resultado, video, img = await run_in_threadpool(procesar_video_bytes, data, nombre)
    except ValueError as e:
        raise HTTPException(422, str(e))

    base = re.sub(r'[^A-Za-z0-9_.-]+', '_', sin_tildes(Path(nombre).stem)) or 'video'
    if formato == 'video':
        cab = {'X-Puntaje': '' if resultado['puntaje'] is None else str(resultado['puntaje']),
               'X-Tipo-Atajada': resultado['tipo_atajada'] or '',
               'X-Momento-Segundos': '' if resultado['momento_segundos'] is None else str(resultado['momento_segundos']),
               'X-Detectada': str(resultado['detectada']).lower(),
               'X-Resultado': base64.b64encode(json.dumps(resultado, ensure_ascii=False).encode()).decode(),
               'Content-Disposition': f'attachment; filename="{base}_ANOTADO.mp4"'}
        return Response(video, media_type='video/mp4', headers=cab)
    if formato == 'json':
        resultado['video_anotado'] = {'nombre': f'{base}_ANOTADO.mp4', 'content_type': 'video/mp4',
                                      'base64': base64.b64encode(video).decode()}
        resultado['imagen_atajada'] = ({'nombre': f'{base}_momento_atajada.jpg', 'content_type': 'image/jpeg',
                                        'base64': base64.b64encode(img).decode()} if img else None)
    return JSONResponse(resultado)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '8000')))
