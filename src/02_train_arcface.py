import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path
import random
import math


# CONFIGURACIÓN

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
MODEL_NAME = 'vit_base_patch16_224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
EMBED_DIM = 512          # dimensión final del embedding
ARCFACE_S = 30.0         # escala de ArcFace
ARCFACE_M = 0.30         # margen angular (más suave por tener pocas clases)
TRIPLET_MARGIN = 0.3
TRIPLET_WEIGHT = 0.5     # peso de la pérdida triplet

print(f'Usando dispositivo: {DEVICE}')
print(f'Usando modelo: {MODEL_NAME}')


# DATASET: LEE data/dog/... y data/cat/...


class PetIDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []  # (ruta_imagen, id_entero, etiqueta_texto)
        self.transform = transform

        label_to_index = {}
        index = 0

        for species in ['dog', 'cat']:
            species_dir = data_dir / species
            if not species_dir.exists():
                continue

            for identity_folder in sorted(species_dir.iterdir()):
                if not identity_folder.is_dir():
                    continue

                identity_name = identity_folder.name  # ej: dog_001

                if identity_name not in label_to_index:
                    label_to_index[identity_name] = index
                    index += 1

                label_id = label_to_index[identity_name]

                for img_path in sorted(identity_folder.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        self.samples.append((img_path, label_id, identity_name))

        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}
        self.num_classes = len(label_to_index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_id, label_name = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label_id, label_name


# Transformaciones: entrenamiento con aumentos suaves
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Transformaciones para evaluación (sin aumentos)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = PetIDDataset(DATA_DIR, transform=train_transform)

if len(dataset) == 0:
    raise RuntimeError(f'No se encontraron imágenes en {DATA_DIR}')

print(f'Dataset cargado: {len(dataset)} imágenes')
print(f'Identidades: {dataset.num_classes}')
print(f'Clases: {list(dataset.label_to_index.keys())}')


# MODELO: ViT + CAPA PROYECTORA + ARCFACE

class PetReIDModel(nn.Module):
    def __init__(self, model_name, embed_dim=512, num_classes=10):
        super().__init__()

        # Backbone ViT preentrenado, sin capa de clasificación
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features  # normalmente 768

        # Capa proyectora: 768 -> 512
        self.projection = nn.Linear(backbone_dim, embed_dim)

        # Parámetros de ArcFace: un vector por clase
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.s = ARCFACE_S
        self.m = ARCFACE_M

    def forward(self, x, labels=None):
        features = self.backbone(x)              # (B, 768)
        embeddings = F.normalize(self.projection(features), dim=1)  # (B, 512)

        if labels is None:
            return embeddings

        # Normalizar pesos de clase
        weight = F.normalize(self.weight, dim=1)  # (num_classes, 512)

        # cos(theta) entre embedding y cada clase
        cosine = F.linear(embeddings, weight)     # (B, num_classes)

        # ArcFace: sumar margen angular m al coseno de la clase correcta
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_cosine = torch.cos(theta + self.m)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = cosine * (1 - one_hot) + target_cosine * one_hot
        output = output * self.s

        return output, embeddings


model = PetReIDModel(
    model_name=MODEL_NAME,
    embed_dim=EMBED_DIM,
    num_classes=dataset.num_classes
).to(DEVICE)


# 4. PÉRDIDAS: ARCFACE (CrossEntropy) + TRIPLET

criterion_arcface = nn.CrossEntropyLoss()
criterion_triplet = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# DATALOADER

# Para un dataset tan pequeño, usamos batch aleatorio simple.
# Con 6 fotos por identidad, es probable que un batch de 8 tenga
# varias fotos de la misma identidad (bueno para triplet).
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


# FUNCIÓN PARA FORMAR TRIPLETOS DENTRO DEL BATCH

def build_triplets(embeddings, labels):
    """
    Para cada imagen (anchor), busca:
      - positive: otra imagen del mismo individuo en el batch
      - negative: una imagen de otro individuo en el batch
    Si no hay positive o negative válido, se omite ese anchor.
    """
    anchors, positives, negatives = [], [], []

    for i in range(len(labels)):
        same = (labels == labels[i])
        diff = (labels != labels[i])

        same_idx = torch.nonzero(same).squeeze(-1)
        diff_idx = torch.nonzero(diff).squeeze(-1)

        # Excluir el propio anchor como positive
        same_idx = same_idx[same_idx != i]

        if len(same_idx) == 0 or len(diff_idx) == 0:
            continue

        pos_idx = random.choice(same_idx.tolist())
        neg_idx = random.choice(diff_idx.tolist())

        anchors.append(embeddings[i])
        positives.append(embeddings[pos_idx])
        negatives.append(embeddings[neg_idx])

    if len(anchors) == 0:
        return None

    return (torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives))

# ENTRENAMIENTO

model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    total_arc = 0.0
    total_trip = 0.0
    correct = 0
    total = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, embeddings = model(imgs, labels=labels)

        # Pérdida ArcFace
        loss_arc = criterion_arcface(logits, labels)

        # Pérdida Triplet (si se pueden formar tripletos)
        triplets = build_triplets(embeddings, labels)
        if triplets is not None:
            a, p, n = triplets
            loss_trip = criterion_triplet(a, p, n)
        else:
            loss_trip = torch.tensor(0.0, device=DEVICE)

        loss = loss_arc + TRIPLET_WEIGHT * loss_trip

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_arc += loss_arc.item()
        total_trip += loss_trip.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'Loss total: {total_loss/len(loader):.4f} | '
          f'ArcFace: {total_arc/len(loader):.4f} | '
          f'Triplet: {total_trip/len(loader):.4f} | '
          f'Acc entrenamiento: {acc:.2f}%')


# EVALUACIÓN: RANK-1 Y RANK-5 (usando embeddings)

model.eval()

eval_dataset = PetIDDataset(DATA_DIR, transform=eval_transform)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels, _ in eval_loader:
        imgs = imgs.to(DEVICE)
        embeddings = model(imgs)  # sin labels -> solo embeddings
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings)  # (N, 512)
all_labels = torch.cat(all_labels)          # (N,)

# Matriz de similitud coseno entre todas las imágenes
similarity = all_embeddings @ all_embeddings.T  # (N, N)

rank1_correct = 0
rank5_correct = 0

N = all_labels.size(0)

for i in range(N):
    sims = similarity[i].clone()
    sims[i] = -1e9  # excluir la propia imagen

    ranked = torch.argsort(sims, descending=True)
    ranked_labels = all_labels[ranked]

    true_label = all_labels[i]

    if ranked_labels[0] == true_label:
        rank1_correct += 1

    if true_label in ranked_labels[:5]:
        rank5_correct += 1

rank1 = 100.0 * rank1_correct / N
rank5 = 100.0 * rank5_correct / N

print('\n' + '=' * 60)
print('EVALUACIÓN (usando las mismas imágenes del piloto)')
print('=' * 60)
print(f'Imágenes evaluadas: {N}')
print(f'Rank-1: {rank1:.2f}%')
print(f'Rank-5: {rank5:.2f}%')
print('=' * 60)

# GUARDAR MODELO ENTRENADO

models_dir = Path(__file__).resolve().parent.parent / 'models'
models_dir.mkdir(exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_index': dataset.label_to_index,
    'index_to_label': dataset.index_to_label,
    'embed_dim': EMBED_DIM,
    'num_classes': dataset.num_classes,
}, models_dir / 'vit_arcface_piloto.pth')

print(f'Modelo guardado en: {models_dir / "vit_arcface_piloto.pth"}')