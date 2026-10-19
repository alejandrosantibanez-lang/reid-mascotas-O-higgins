# 03_construir_modelo.py
# Construye el modelo de Re-ID: backbone ViT + cabeza ArcFace.
# - El backbone convierte una imagen en un vector (embedding).
# - La cabeza ArcFace solo se usa durante el entrenamiento para
#   enseñar al modelo a separar bien las identidades.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from config import (
    BACKBONE_NAME,
    EMBEDDING_DIM,
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
)


# Cabeza ArcFace 
class CabezaArcFace(nn.Module):
    """
    Clasificador con margen angular (ArcFace).

    Durante el entrenamiento, obliga a que el embedding de un animal
    esté "cerca" de su clase y "lejos" de las demás, con un margen
    angular extra (m) que hace el problema más difícil a propósito.

    Fórmula del logit para la clase correcta:
        z_y = s * cos(theta_y + m)
    Para las demás clases:
        z_j = s * cos(theta_j)
    """

    def __init__(self, dim_embedding, num_clases, margen=0.5, escala=30.0):
        super().__init__()
        self.margen = margen
        self.escala = escala
        # Un vector de pesos por cada clase (identidad)
        self.pesos_clases = nn.Parameter(torch.randn(num_clases, dim_embedding))

    def forward(self, embeddings, etiquetas):
        """
        embeddings: tensor (batch, dim_embedding)
        etiquetas:  tensor (batch,) con el índice de la clase correcta
        Devuelve logits listos para CrossEntropyLoss.
        """
        # Normalizar ambos para que el producto punto = coseno del ángulo
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        pesos_norm = F.normalize(self.pesos_clases, p=2, dim=1)

        # Coseno entre cada embedding y todas las clases: (batch, num_clases)
        coseno = embeddings_norm @ pesos_norm.t()

        # Aplicar el margen solo a la clase correcta
        margen_aplicado = torch.cos(
            torch.acos(coseno.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) + self.margen
        )

        # Crear una máscara one-hot de la clase correcta
        una_sola_clase = F.one_hot(etiquetas, num_classes=coseno.size(1)).float()

        # Donde la clase es la correcta, usar el coseno CON margen;
        # en las demás, el coseno normal.
        logits = coseno * (1 - una_sola_clase) + margen_aplicado * una_sola_clase

        # Escalar (s) para que el softmax no sea demasiado "plano"
        return self.escala * logits


# Modelo completo 
class ModeloReID(nn.Module):
    """
    Modelo completo de Re-Identificación:
    - backbone (ViT): imagen → vector de 512 dimensiones
    - cabeza ArcFace: vector + etiqueta → logits (solo para entrenar)
    """

    def __init__(self, num_clases):
        super().__init__()
        # Backbone ViT preentrenado (ya "sabe" ver imágenes)
        self.backbone = create_model(
            BACKBONE_NAME,
            pretrained=True,
            num_classes=0,          # Sin clasificador propio, queremos el vector
        )
        # Proyección al tamaño de embedding que usará ArcFace
        self.proyeccion = nn.Linear(self.backbone.num_features, EMBEDDING_DIM)

        # Cabeza ArcFace (solo para entrenamiento)
        self.cabeza_arcface = CabezaArcFace(
            dim_embedding=EMBEDDING_DIM,
            num_clases=num_clases,
            margen=ARCFACE_MARGIN,
            escala=ARCFACE_SCALE,
        )

    def forward(self, imagenes, etiquetas=None):
        """
        imagenes: tensor (batch, 3, 224, 224)
        etiquetas: tensor (batch,) — solo necesario para entrenar

        - Si se dan etiquetas → modo entrenamiento, devuelve logits ArcFace.
        - Si no → modo inferencia, devuelve solo el embedding (vector).
        """
        # 1) Backbone extrae características visuales
        caracteristicas = self.backbone(imagenes)          # (batch, 768)

        # 2) Proyección a la dimensión de embedding
        embeddings = self.proyeccion(caracteristicas)      # (batch, 512)

        # 3) Si estamos entrenando, pasar por ArcFace
        if etiquetas is not None:
            logits = self.cabeza_arcface(embeddings, etiquetas)
            return logits
        else:
            return embeddings


#Función auxiliar para crear el modelo fácilmente 
def construir_modelo(num_clases):
    """Crea y devuelve una instancia de ModeloReID."""
    return ModeloReID(num_clases=num_clases)


if __name__ == "__main__":
    # Prueba rápida: crear el modelo y pasarle una imagen falsa
    modelo = construir_modelo(num_clases=10)
    imagen_falsa = torch.randn(2, 3, 224, 224)   # 2 imágenes falsas
    etiquetas_falsas = torch.tensor([0, 1])

    # Modo entrenamiento
    logits = modelo(imagen_falsa, etiquetas_falsas)
    print(f"Logits (entrenamiento): {logits.shape}")   # (2, 10)

    # Modo inferencia
    embeddings = modelo(imagen_falsa)
    print(f"Embeddings (inferencia): {embeddings.shape}")  # (2, 512)