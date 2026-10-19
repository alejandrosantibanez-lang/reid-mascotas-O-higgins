# 04_entrenar_modelo.py
# Entrena el modelo de Re-ID usando el split train/test generado antes.
# Usa: DatasetReID (datos) + ModeloReID (ViT + ArcFace) + AdamW (optimizador).

import os
import torch
from torch.utils.data import DataLoader
from torch import nn

from config import (
    DATA_ROOT, BASE_DIR, EPOCHS, BATCH_SIZE, NUM_WORKERS,
    LEARNING_RATE_BACKBONE, LEARNING_RATE_HEAD, MODEL_SAVE_PATH,
)
from utils.datos_utils import (
    DatasetReID, cargar_split, transform_entrenamiento,
)
from importlib import import_module

# Importamos el modelo (está en src/, un nivel arriba de utils/)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
modelo_modulo = import_module("03_construir_modelo")
ModeloReID = modelo_modulo.ModeloReID
construir_modelo = modelo_modulo.construir_modelo


# ─── Función: entrenar una época ────────────────────────────────────────────
def entrenar_una_epoca(modelo, dataloader, optimizador, criterio, dispositivo):
    """
    Recorre todos los batches del dataset una vez y ajusta los pesos.
    Devuelve la pérdida promedio de la época.
    """
    modelo.train()  # Activa modo entrenamiento (dropout, batchnorm, etc.)
    perdida_total = 0.0
    num_batches = 0

    for imagenes, etiquetas in dataloader:
        # Mover datos al dispositivo (GPU o CPU)
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        # 1) Forward: el modelo predice
        logits = modelo(imagenes, etiquetas)

        # 2) Calcular qué tan mal predijo
        perdida = criterio(logits, etiquetas)

        # 3) Backward: calcular gradientes
        optimizador.zero_grad()      # Limpiar gradientes anteriores
        perdida.backward()           # Calcular nuevos gradientes

        # 4) Step: ajustar los pesos
        optimizador.step()

        # Acumular para el promedio
        perdida_total += perdida.item()
        num_batches += 1

    return perdida_total / num_batches


# ─── Función principal ──────────────────────────────────────────────────────
def entrenar():
    # 1) Detectar dispositivo
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")

    # 2) Cargar el split train/test
    ruta_split = os.path.join(BASE_DIR, "data", "split.json")
    train_ids, test_ids = cargar_split(ruta_split)
    print(f"Train: {len(train_ids)} identidades")
    print(f"Test:  {len(test_ids)} identidades")

    # 3) Crear dataset de entrenamiento
    dataset_train = DatasetReID(
        data_root=DATA_ROOT,
        identidades=train_ids,
        transform=transform_entrenamiento,
    )
    print(f"Imágenes de entrenamiento: {len(dataset_train)}")

    # 4) Crear DataLoader (agrupa en batches)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,           # Mezclar en cada época
        num_workers=NUM_WORKERS,
    )

    # 5) Crear modelo y moverlo al dispositivo
    num_clases = len(train_ids)
    modelo = construir_modelo(num_clases=num_clases)
    modelo = modelo.to(dispositivo)
    print(f"Modelo creado con {num_clases} clases")

    # 6) Optimizador con dos grupos de LR (backbone bajo, cabeza alta)
    optimizador = torch.optim.AdamW([
        {"params": modelo.backbone.parameters(), "lr": LEARNING_RATE_BACKBONE},
        {"params": modelo.proyeccion.parameters(), "lr": LEARNING_RATE_HEAD},
        {"params": modelo.cabeza_arcface.parameters(), "lr": LEARNING_RATE_HEAD},
    ])

    # 7) Función de pérdida
    criterio = nn.CrossEntropyLoss()

    # 8) Loop de épocas
    print("\nIniciando entrenamiento...")
    for epoca in range(EPOCHS):
        perdida = entrenar_una_epoca(
            modelo, dataloader_train, optimizador, criterio, dispositivo
        )
        print(f"Época {epoca+1}/{EPOCHS} — Pérdida: {perdida:.4f}")

    # 9) Guardar el modelo entrenado
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    ruta_modelo = os.path.join(MODEL_SAVE_PATH, "modelo_reid.pth")
    torch.save(modelo.state_dict(), ruta_modelo)
    print(f"\nModelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    entrenar()