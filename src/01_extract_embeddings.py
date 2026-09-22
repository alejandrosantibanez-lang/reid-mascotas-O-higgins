import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path

#Configuración

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
MODEL_NAME = 'vit_base_patch16_224'  # ViT-B/16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Usando dispositivo: {DEVICE}')
print(f'Usando modelo: {MODEL_NAME}')

# Cargar modelo ViT-B/16 preentrenado en ImageNet

print('Cargando modelo ViT-B/16...')
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model = model.to(DEVICE)
model.eval()
print('Modelo cargado')

# Transformaciones para preprocesar imágenes (estándar para ViT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

# Extraer embeddings de todas las imagenes

print('Extrayendo embeddings...')

embeddings_list = []
labels_list = []
species_list = []
image_paths_list = []

# Recorrer todas las carpetas (dog_001, dog_002, ..., cat_005)
for species in ['dog', 'cat']:
    species_dir = DATA_DIR / species
    
    for identity_folder in sorted(species_dir.iterdir()):
        if not identity_folder.is_dir():
            continue
        
        identity_name = identity_folder.name  # ej: dog_001
        label = f'{species}_{identity_name.split("_")[1]}'  # ej: dog_1
        
        image_paths = sorted(
            path for path in identity_folder.iterdir()
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        )

        for img_path in image_paths:
            # Cargar imagen
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
            
            # Extraer embedding
            with torch.no_grad():
                embedding = model(img_tensor)  # (1, 768)
            
            embeddings_list.append(embedding.squeeze().cpu().numpy())
            labels_list.append(label)
            species_list.append(species)
            image_paths_list.append(str(img_path))

# Convertir a numpy arrays
embeddings = np.array(embeddings_list)  # (60, 768)
labels = np.array(labels_list)          # (60,)
species = np.array(species_list)        # (60,)

if not embeddings_list:
    raise RuntimeError(
        f'No se encontraron imagenes en {DATA_DIR}. '
        'Se esperaban archivos .png, .jpg o .jpeg.'
    )

print(f'Embeddings extraídos: {embeddings.shape}')
print(f'   - {len(labels)} imágenes')
print(f'   - {len(set(labels))} identidades únicas')
print(f'   - {embeddings.shape[1]} dimensiones por embedding')


# visualizacion con t-SNE (2D) ----- simplemente transformar los embeddings de 768D a 2D para poder graficarlos

print('Generando t-SNE...')

tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings_2d = tsne.fit_transform(embeddings)

# Crear directorio de resultados junto a este script, sin depender del directorio actual
results_dir = Path(__file__).resolve().parent / 'results'
results_dir.mkdir(exist_ok=True)

# Ploteamos por ESPECIE (perro vs gato)

plt.figure(figsize=(10, 8))

# Perros (azul)
dog_mask = species == 'dog'
plt.scatter(embeddings_2d[dog_mask, 0], embeddings_2d[dog_mask, 1], 
            c='blue', label='Perro', alpha=0.7, s=100, edgecolors='black', linewidth=1)

# Gatos (rojo)
cat_mask = species == 'cat'
plt.scatter(embeddings_2d[cat_mask, 0], embeddings_2d[cat_mask, 1], 
            c='red', label='Gato', alpha=0.7, s=100, edgecolors='black', linewidth=1)

plt.title('t-SNE de Embeddings (ViT-B/16)\nPerros (azul) vs Gatos (rojo)', fontsize=14)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'tsne_species.png', dpi=300)
print('Gráfico guardado: results/tsne_species.png')
plt.close()

# Ploteamos por IDENTIDAD (cada identidad con color diferente), es decir cada perro y cada gato  con un color diferente

plt.figure(figsize=(12, 10))

# Colores únicos para cada identidad
unique_labels = sorted(set(labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    mask = labels == label
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                c=[colors[i]], label=label, alpha=0.7, s=100, 
                edgecolors='black', linewidth=1)

plt.title('t-SNE de Embeddings (ViT-B/16)\nPor Identidad', fontsize=14)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'tsne_identities.png', dpi=300, bbox_inches='tight')
print('Gráfico guardado: results/tsne_identities.png')
plt.close()

# guardamos EMBEDDINGS en un archivo para observacion 

import csv

with open(results_dir / 'embeddings_metadata.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'species', 'tsne_1', 'tsne_2'])
    for i in range(len(labels)):
        writer.writerow([
            image_paths_list[i],
            labels[i],
            species[i],
            embeddings_2d[i, 0],
            embeddings_2d[i, 1]
        ])

print('Metadata guardada: results/embeddings_metadata.csv')

# Si pasa todo correctamente, mostramos el RESUMEN

print('\n' + '='*60)
print('RESUMEN')
print('='*60)
print(f'Imágenes procesadas: {len(labels)}')
print(f'Identidades únicas: {len(set(labels))}')
print(f'Dimensiones de embedding: {embeddings.shape[1]}')
print(f'Gráficos guardados en: results/')
print('   - tsne_species.png (perros vs gatos)')
print('   - tsne_identities.png (cada identidad)')
print('   - embeddings_metadata.csv (coordenadas + metadata)')
print('='*60)