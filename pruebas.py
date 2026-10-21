"""
Prueba en tiempo real del modelo YOLO entrenado (best.pt) usando la cámara.

Uso:
    python pruebas.py                 -> usa la cámara
    python pruebas.py video.mp4       -> usa un video (o una imagen .jpg/.png)

Controles:
    q / ESC  -> salir
    s        -> guardar captura en la carpeta 'pruebas/'
    +  / -   -> subir / bajar el umbral de confianza
"""
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ------------------ CONFIGURACIÓN ------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs_sv" / "yolo_sv" / "weights" / "best.pt"
CAMERA_INDEX = 0          # 0 = cámara por defecto; prueba 1, 2... si tienes varias
CONF = 0.1              # umbral de confianza inicial
IMGSZ = 640               # mismo tamaño usado en el entrenamiento
FRAME_W, FRAME_H = 1280, 720
SAVE_DIR = BASE_DIR / "pruebas"
# ---------------------------------------------------


def abrir_camara(index: int) -> cv2.VideoCapture:
    # En Windows CAP_DSHOW abre la cámara mucho más rápido
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo: {MODEL_PATH}")
    print(f"Dispositivo: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    model = YOLO(str(MODEL_PATH))
    print("Clases:", model.names)

    fuente = sys.argv[1] if len(sys.argv) > 1 else None
    es_imagen = fuente is not None and fuente.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    if fuente is None:
        cap = abrir_camara(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {CAMERA_INDEX}")
    elif es_imagen:
        cap = None
        img = cv2.imread(fuente)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {fuente}")
    else:
        cap = cv2.VideoCapture(fuente)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {fuente}")

    SAVE_DIR.mkdir(exist_ok=True)
    conf = CONF
    prev = time.time()
    fps = 0.0
    win = "YOLO - Prueba de camara (q para salir)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if es_imagen:
            ok, frame = True, img.copy()
        else:
            ok, frame = cap.read()
        if not ok:
            print("No se pudo leer el frame de la cámara.")
            break

        results = model.predict(frame, conf=conf, imgsz=IMGSZ, device=device,
                                verbose=False)
        r = results[0]
        annotated = r.plot()

        # FPS suavizado
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        n = len(r.boxes)
        info = f"FPS: {fps:.1f} | Conf: {conf:.2f} | Detecciones: {n}"
        cv2.rectangle(annotated, (0, 0), (520, 35), (0, 0, 0), -1)
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Lista de clases detectadas en este frame
        y = 60
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            cv2.putText(annotated, f"{cls_name}: {float(box.conf):.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y += 25

        cv2.imshow(win, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = SAVE_DIR / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(fname), annotated)
            print(f"Captura guardada: {fname}")
        elif key in (ord("+"), ord("=")):
            conf = min(conf + 0.05, 0.95)
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)

        # Cerrar también al pulsar la X de la ventana
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
