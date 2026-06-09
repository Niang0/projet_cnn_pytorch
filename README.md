# projet_cnn_pytorch
# 🧠 Deep Learning avec PyTorch — Classification d'images CNN

**Niveau :** Master / Ingénierie IA — Deep Learning  
**Auteur :** _Abdou Salam NIANG_  
**Date limite :** Mardi 09 Juin 2026 — 17h59

Classification binaire d'images avec trois architectures CNN :

- **LeNet-5** — from scratch (+ étude comparative des activations)
- **VGG16** — from scratch
- **ResNet18** — fine-tuning par transfer learning

---

## Structure du projet

```
projet_cnn/
├── data/
│   ├── train/
│   │   ├── classe_A/
│   │   └── classe_B/
│   └── val/
│       ├── classe_A/
│       └── classe_B/
├── models/
│   ├── lenet5.py          # LeNet-5 from scratch (Tanh / ReLU / Sigmoid)
│   ├── vgg16.py           # VGG16 from scratch
│   └── resnet18_ft.py     # ResNet18 fine-tuning
├── dataset.py             # CustomImageDataset + transforms
├── train.py               # train_epoch / eval_epoch
├── main.py                # Pipeline principal + wandb
├── utils.py               # Métriques, visualisations, seed
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/projet_cnn.git
cd projet_cnn
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**

```
torch>=2.0
torchvision
wandb
scikit-learn
matplotlib
tqdm
numpy
Pillow
```

---

## Dataset

Télécharger le dataset depuis Google Drive :  
 https://drive.google.com/file/d/10ID5k_1BR-CI1XuwQ_Y-qNot66249JLp/view?usp=sharing

Extraire dans le dossier `data/` en respectant la structure attendue :

```
data/
├── train/
│   ├── classe_A/   *.jpg | *.png
│   └── classe_B/   *.jpg | *.png
└── val/
    ├── classe_A/   *.jpg | *.png
    └── classe_B/   *.jpg | *.png
```

> Si le dataset n'est pas pré-splitté, le script `utils.py` effectue automatiquement un split 80/20 stratifié.

---

## Configuration Weights & Biases

```bash
wandb login
```

Entrez votre clé API disponible sur [https://wandb.ai](https://wandb.ai).
"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"
---

## Lancement de l'entraînement

### Entraîner les 3 modèles (pipeline complet)

```bash
python main.py
```

### Entraîner un modèle spécifique

```bash
python main.py --model lenet5        # LeNet-5 (toutes les variantes d'activation)
python main.py --model vgg16         # VGG16 from scratch
python main.py --model resnet18      # ResNet18 fine-tuning
```

### Options disponibles

| Argument       | Description                                        | Défaut          |
| -------------- | -------------------------------------------------- | --------------- |
| `--model`      | Modèle à entraîner (`lenet5`, `vgg16`, `resnet18`) | `all`           |
| `--epochs`     | Nombre d'époques                                   | voir ci-dessous |
| `--batch_size` | Taille des batches                                 | voir ci-dessous |
| `--lr`         | Learning rate                                      | voir ci-dessous |
| `--seed`       | Seed de reproductibilité                           | `42`            |
| `--data_dir`   | Chemin vers le dataset                             | `./data`        |

---

## Hyperparamètres de référence

| Modèle   | Learning Rate | Batch Size | Epochs | Input Size |
| -------- | ------------- | ---------- | ------ | ---------- |
| LeNet-5  | 1e-3          | 64         | 50     | 32×32      |
| VGG16    | 1e-4          | 16–32      | 30     | 224×224    |
| ResNet18 | 1e-3          | 32         | 50     | 224×224    |

- **Optimiseur :** Adam pour les trois modèles
- **Loss :** CrossEntropyLoss
- **Seed :** fixée pour la reproductibilité (random, numpy, torch)

---

## Architectures implémentées

### LeNet-5 (from scratch)

Architecture originale LeCun (1998), adaptée RGB 32×32, 2 classes.  
Trois variantes comparées : `Tanh` (original) | `ReLU` | `Sigmoid`

### VGG16 (from scratch)

5 blocs convolutionnels [64, 128, 256, 512, 512], tête FC (4096 → 4096 → 2), Dropout p=0.5, AdaptiveAvgPool2d(7,7).

### ResNet18 (fine-tuning)

Poids ImageNet pré-entraînés, backbone gelé, tête remplacée par `nn.Linear(512, 2)`.

---

## Suivi des expériences (wandb)

Les runs sont loggués automatiquement sur Weights & Biases.  
Noms des runs : `lenet5_tanh`, `lenet5_relu`, `lenet5_sigmoid`, `vgg16_scratch`, `resnet18_finetune`

Métriques loggées à chaque epoch : `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, `val_f1_score`, `learning_rate`

🔗 Lien vers le projet wandb : https://wandb.ai/abdoukillmonger-me/devoir-cnn-pytorch?nw=nwuserabdoukillmonger

---

## Points importants

- LeNet-5 : **pas de normalisation ImageNet** (pas de Normalize dans les transforms)
- Toujours appeler `model.eval()` + `torch.no_grad()` en validation
- La seed est fixée au début de `main.py` (random, numpy, torch)
- Les labels doivent être `torch.long` pour `CrossEntropyLoss`

---

# Livrables

- [x] Code Python structuré et commenté
- [x] README.md
- [x] Rapport PDF (4–6 pages)
- [x] Lien wandb partagé

⚠️ Tout plagiat entraîne la note de 0. Le code doit être personnel et commenté.
