import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path
import random
import csv

# 1. CONFIGURACIÓN

SPLITS_DIR = Path(__file__).resolve().parent.parent / 'splits'
MODEL_NAME = 'vit_base_patch16_224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 35
BATCH_SIZE = 8
LR = 1e-4
EMBED_DIM = 512
ARCFACE_S = 30.0
ARCFACE_M = 0.30
TRIPLET_MARGIN = 0.3
TRIPLET_WEIGHT = 0.5

print(f'Usando dispositivo: {DEVICE}')

#DATASET DESDE CSV

def load_split(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

train_rows = load_split(SPLITS_DIR / 'train.csv')
gallery_rows = load_split(SPLITS_DIR / 'gallery.csv')
probe_rows = load_split(SPLITS_DIR / 'probe.csv')

# Mapear identidades a índices (solo las de train)
identities = sorted(set(r['identity'] for r in train_rows))
label_to_index = {name: i for i, name in enumerate(identities)}
index_to_label = {i: name for name, i in label_to_index.items()}

print(f'Train: {len(train_rows)} imágenes, {len(identities)} identidades')
print(f'Gallery: {len(gallery_rows)} imágenes')
print(f'Probe: {len(probe_rows)} imágenes')

# 3. DATASET PYTORCH

class SplitDataset(Dataset):
    def __init__(self, rows, label_to_index, transform=None):
        self.samples = []
        self.transform = transform
        for row in rows:
            img_path = row['image_path']
            identity = row['identity']
            label_id = label_to_index[identity]
            self.samples.append((img_path, label_id, identity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_id, identity = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label_id, identity

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = SplitDataset(train_rows, label_to_index, train_transform)
gallery_dataset = SplitDataset(gallery_rows, label_to_index, eval_transform)
probe_dataset = SplitDataset(probe_rows, label_to_index, eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE, shuffle=False)
probe_loader = DataLoader(probe_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. MODELO

class PetReIDModel(nn.Module):
    def __init__(self, model_name, embed_dim=512, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features
        self.projection = nn.Linear(backbone_dim, embed_dim)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = ARCFACE_S
        self.m = ARCFACE_M

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = F.normalize(self.projection(features), dim=1)

        if labels is None:
            return embeddings

        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_cosine = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = cosine * (1 - one_hot) + target_cosine * one_hot
        output = output * self.s
        return output, embeddings

model = PetReIDModel(MODEL_NAME, EMBED_DIM, len(identities)).to(DEVICE)

criterion_arcface = nn.CrossEntropyLoss()
criterion_triplet = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

#  TRIPLETS 

def build_triplets(embeddings, labels):
    anchors, positives, negatives = [], [], []
    for i in range(len(labels)):
        same = (labels == labels[i])
        diff = (labels != labels[i])
        same_idx = torch.nonzero(same).squeeze(-1)
        diff_idx = torch.nonzero(diff).squeeze(-1)
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
    return (torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

# ENTRENAR (solo con train)

model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, embeddings = model(imgs, labels=labels)
        loss_arc = criterion_arcface(logits, labels)
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
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'Loss: {total_loss/len(train_loader):.4f} | '
          f'Acc train: {100.0*correct/total:.2f}%')

# EXTRAER EMBEDDINGS DE GALLERY Y PROBE


model.eval()

def extract_embeddings(loader):
    embs, labels, ids = [], [], []
    with torch.no_grad():
        for imgs, labels_batch, identities in loader:
            imgs = imgs.to(DEVICE)
            e = model(imgs)
            embs.append(e.cpu())
            labels.append(labels_batch)
            ids.extend(identities)
    return torch.cat(embs), torch.cat(labels), ids

gallery_embs, gallery_labels, gallery_ids = extract_embeddings(gallery_loader)
probe_embs, probe_labels, probe_ids = extract_embeddings(probe_loader)


# EVALUAR: PROBE vs GALLERY (Rank-1, Rank-5)

# Similitud coseno entre cada probe y todas las gallery
similarity = probe_embs @ gallery_embs.T  # (num_probe, num_gallery)

rank1_correct = 0
rank5_correct = 0

for i in range(len(probe_ids)):
    sims = similarity[i]
    ranked = torch.argsort(sims, descending=True)
    ranked_labels = [gallery_ids[j] for j in ranked]

    true_id = probe_ids[i]

    if ranked_labels[0] == true_id:
        rank1_correct += 1
    if true_id in ranked_labels[:5]:
        rank5_correct += 1

rank1 = 100.0 * rank1_correct / len(probe_ids)
rank5 = 100.0 * rank5_correct / len(probe_ids)

print('\n' + '=' * 60)
print('EVALUACIÓN HONESTA (probe vs gallery)')
print('=' * 60)
print(f'Probes evaluados: {len(probe_ids)}')
print(f'Rank-1: {rank1:.2f}%')
print(f'Rank-5: {rank5:.2f}%')
print('=' * 60)


#GUARDAR MODELO

models_dir = Path(__file__).resolve().parent.parent / 'models'
models_dir.mkdir(exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_index': label_to_index,
    'embed_dim': EMBED_DIM,
}, models_dir / 'vit_arcface_split.pth')
print(f'Modelo guardado: {models_dir / "vit_arcface_split.pth"}')