# src/utils/datos_utils.py
# Funciones para cargar, preprocesar y preparar imágenes para el modelo.
# Lo usan 04_entrenar_modelo.py, 05_evaluar_modelo.py y 06_extraer_embeddings.py.

import os
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_ROOT, IMAGE_SIZE


# Transformaciones 
# Las imágenes deben llegar al ViT como tensores de 224x224 normalizados.

# Para ENTRENAR: aplicamos "aumentación" (variations aleatorias) para que
# el modelo aprenda a reconocer al animal en distintas condiciones.
transform_entrenamiento = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),      # Recorte aleatorio y redimensionar
    transforms.RandomHorizontalFlip(),              # Espejo horizontal aleatorio
    transforms.ToTensor(),                          # Imagen PIL → tensor [0, 1]
    transforms.Normalize(                           # Normalizar (media/std de ImageNet)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Para EVALUAR: sin aleatoriedad, siempre el mismo preprocesamiento
# (así los resultados son reproducibles).
transform_evaluacion = transforms.Compose([
    transforms.Resize(256),                         # Lado menor a 256 px
    transforms.CenterCrop(IMAGE_SIZE),              # Recorte central de 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Dataset personalizado 
class DatasetReID(Dataset):
    """
    Dataset de re-identificación: cada item es (imagen, etiqueta_identidad).
    Lee las identidades desde el split.json generado por 02_dividir_dataset.py.
    """

    def __init__(self, data_root, identidades, transform):
        self.data_root = data_root
        self.transform = transform
        self.muestras = []       # Lista de (ruta_imagen, indice_identidad)
        self.etiqueta_a_indice = {}   # {"dogs/perro_001": 0, "dogs/perro_002": 1, ...}

        # Asignar un número entero a cada identidad (ArcFace necesita enteros)
        for indice, identidad in enumerate(sorted(identidades)):
            self.etiqueta_a_indice[identidad] = indice

        # Recorrer cada identidad y registrar sus imágenes
        for identidad in sorted(identidades):
            ruta_identidad = os.path.join(data_root, identidad)
            indice = self.etiqueta_a_indice[identidad]

            for archivo in sorted(os.listdir(ruta_identidad)):
                ruta = os.path.join(ruta_identidad, archivo)
                if os.path.isfile(ruta):
                    self.muestras.append((ruta, indice))

    def __len__(self):
        """Cuántas imágenes tiene el dataset."""
        return len(self.muestras)

    def __getitem__(self, indice):
        """
        Devuelve una muestra: (imagen_procesada, etiqueta).
        PyTorch llama a esta función automáticamente al pedir un batch.
        """
        ruta, etiqueta = self.muestras[indice]
        imagen = Image.open(ruta).convert("RGB")   # Forzar 3 canales (RGB)
        imagen = self.transform(imagen)
        return imagen, etiqueta


# Función para cargar el split
def cargar_split(ruta_split):
    """Lee el split.json y devuelve las listas de train y test."""
    with open(ruta_split, "r", encoding="utf-8") as f:
        split = json.load(f)
    return split["train"], split["test"]