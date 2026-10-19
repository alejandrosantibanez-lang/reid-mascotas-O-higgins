# Configuración central del proyecto PetFace O'Higgins

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del proyecto

# Rutas
DATA_ROOT = os.path.join(BASE_DIR, "data_ohiggins")    # Carpeta con las imágenes organizadas por identidad
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "modelos")       # Dónde se guarda el modelo entrenado
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings")    # Dónde se guardan los vectores extraídos


# División del dataset 
TRAIN_RATIO = 0.8                    # 80 de identidades para entrenar
TEST_RATIO = 0.2                     # 20 de identidades para probar

# Entrenamiento
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE_BACKBONE = 1e-5        # LR del ViT (bajo, para no romper lo preentrenado)
LEARNING_RATE_HEAD = 1e-4            # LR de la cabeza ArcFace (más alto)
NUM_WORKERS = 0                      # Procesos del DataLoader (0 en Windows)

# ArcFace 
ARCFACE_MARGIN = 0.5                 # Margen angular
ARCFACE_SCALE = 30.0                 # Escala (s)

# Modelo 
BACKBONE_NAME = "vit_base_patch16_224"
EMBEDDING_DIM = 512                  # Dimensión del vector final
IMAGE_SIZE = 224