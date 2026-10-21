"""
Probar el modelo YOLO26 entrenado (detección de armas).
Ubicación: C:\\Users\\Usuario\\PycharmProjects\\TRAIN_YOLO26\\MIGUEL_TRAIN\\probar_modelo.py

Uso (desde la terminal de PyCharm, dentro de MIGUEL_TRAIN):
    python probar_modelo.py                      -> webcam (cámara 0)
    python probar_modelo.py --source 1           -> otra cámara
    python probar_modelo.py --source foto.jpg    -> una imagen
    python probar_modelo.py --source video.mp4   -> un video
    python probar_modelo.py --source carpeta     -> todas las imágenes de una carpeta
    python probar_modelo.py --source test        -> imágenes del split test del dataset
    python probar_modelo.py --source "rtsp://usuario:clave@IP:554/stream"   -> cámara IP / IoT

Opciones:
    --model ruta.pt    usar un modelo concreto (por defecto busca el best.pt más reciente)
    --conf 0.35        umbral de confianza
    --imgsz 640        tamaño de entrada
    --save             guardar resultados en MIGUEL_TRAIN/models/predicciones

Teclas en la ventana: Q o ESC = salir | S = guardar captura | ESPACIO = pausa (imágenes: siguiente)
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE_DIR = Path(r"C:\Users\Usuario\PycharmProjects\TRAIN_YOLO26\MIGUEL_TRAIN")
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "data" / "weapons"
OUT_DIR = MODELS_DIR / "predicciones"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def buscar_modelo() -> Path:
    """Devuelve el best.pt más reciente (models/final o models/runs/*/weights)."""
    candidatos = list((MODELS_DIR / "final").glob("*best.pt")) + list((MODELS_DIR / "runs").glob("*/weights/best.pt"))
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró ningún best.pt en {MODELS_DIR}. Entrena primero con el notebook o usa --model ruta.pt"
        )
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def dibujar_info(frame, fps, n_det, device):
    texto = f"FPS: {fps:.1f} | Detecciones: {n_det} | {device}"
    cv2.rectangle(frame, (0, 0), (len(texto) * 11 + 10, 30), (0, 0, 0), -1)
    cv2.putText(frame, texto, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if n_det > 0:
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 40), (260, h), (0, 0, 255), -1)
        cv2.putText(frame, "ALERTA: ARMA DETECTADA", (8, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def probar_imagenes(model, rutas, args, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, ruta in enumerate(rutas, 1):
        r = model.predict(str(ruta), imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        frame = dibujar_info(r.plot(), 1000 / max(r.speed["inference"], 1e-6), len(r.boxes), device)
        print(f"[{i}/{len(rutas)}] {ruta.name}: {len(r.boxes)} detecciones ({r.speed['inference']:.1f} ms)")
        for b in r.boxes:
            print(f"     - {model.names[int(b.cls)]}: {float(b.conf):.2f}")
        if args.save:
            cv2.imwrite(str(OUT_DIR / f"pred_{ruta.name}"), frame)
        cv2.imshow("YOLO26 - Prueba (ESPACIO: siguiente, Q: salir)", frame)
        tecla = cv2.waitKey(0) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("s"):
            cv2.imwrite(str(OUT_DIR / f"captura_{ruta.name}"), frame)
    cv2.destroyAllWindows()


def probar_stream(model, source, args, device):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    writer = None
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        salida = OUT_DIR / f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        print("Guardando video en:", salida)

    fps, pausa, frame = 0.0, False, None
    while True:
        if not pausa:
            ok, img = cap.read()
            if not ok:
                print("Fin del video / sin señal.")
                break
            t0 = time.time()
            r = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
            fps = 0.9 * fps + 0.1 * (1 / max(time.time() - t0, 1e-6)) if fps else 1 / max(time.time() - t0, 1e-6)
            frame = dibujar_info(r.plot(), fps, len(r.boxes), device)
            if writer:
                writer.write(frame)

        cv2.imshow("YOLO26 - Prueba (Q: salir, S: captura, ESPACIO: pausa)", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord(" "):
            pausa = not pausa
        if tecla == ord("s"):
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ruta = OUT_DIR / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(ruta), frame)
            print("Captura guardada:", ruta)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Probar modelo YOLO26 de detección de armas")
    ap.add_argument("--source", default="0", help="0 (webcam), imagen, video, carpeta, 'test' o URL rtsp/http")
    ap.add_argument("--model", default=None, help="ruta a un .pt (por defecto el best.pt más reciente)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)d
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    modelo = Path(args.model) if args.model else buscar_modelo()
    print("Modelo :", modelo)
    print("Device :", torch.cuda.get_device_name(0) if device == 0 else "CPU")
    model = YOLO(str(modelo))
    print("Clases :", model.names)

    src = args.source
    if src.isdigit():
        probar_stream(model, int(src), args, device)
        return
    if src.lower() == "test":
        carpeta = DATASET_DIR / "test" / "images"
        if not carpeta.exists():
            carpeta = DATASET_DIR / "valid" / "images"
        src = str(carpeta)

    p = Path(src)
    if not p.is_absolute() and not p.exists():
        p = BASE_DIR / src
    if p.is_dir():
        rutas = sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT)
        if not rutas:
            raise FileNotFoundError(f"No hay imágenes en {p}")
        probar_imagenes(model, rutas, args, device)
    elif p.suffix.lower() in IMG_EXT:
        probar_imagenes(model, [p], args, device)
    elif p.suffix.lower() in VID_EXT:
        probar_stream(model, str(p), args, device)
    else:
        probar_stream(model, src, args, device)  # rtsp://, http://, etc.


if __name__ == "__main__":
    main()
