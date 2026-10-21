# TRAIN_YOLO26 – Detección de armas con YOLO26

Proyecto de entrenamiento de un modelo **YOLO26n** para la detección de armas en imágenes y video.

## Contenido del repositorio

Este repositorio contiene solo los **notebooks** y scripts del proyecto.
Los datasets, pesos (`.pt`) y resultados de entrenamiento **no se incluyen** por su tamaño.

| Archivo / carpeta | Descripción |
|---|---|
| `entrenamiento_yolo_sv.ipynb` | Notebook de entrenamiento principal |
| `MIGUEL_TRAIN/` | Notebooks de entrenamiento de armas |
| `TRAIN_FINAL_UCSM_YOLO26/` | Entrenamiento final (UCSM) |
| `probar_modelo.py` | Script para probar el modelo entrenado |
| `requirements.txt` | Dependencias del proyecto |

## Requisitos

- Python 3.10 o superior
- GPU NVIDIA con CUDA (recomendado)

## Instalación

```bash
git clone https://github.com/TU_USUARIO/TRAIN_YOLO26.git
cd TRAIN_YOLO26
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Descarga el dataset y colócalo en la carpeta `ds/` (con `train/`, `valid/` y `test/`).
2. Abre y ejecuta el notebook de entrenamiento.
3. Prueba el modelo con:

```bash
python probar_modelo.py
```

## Autor

Miguel – Universidad Católica de Santa María (UCSM)