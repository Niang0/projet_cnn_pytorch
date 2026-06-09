"""
dataset.py  —  Chargement des images et préparation des données

En PyTorch, pour charger ses propres données on crée une classe Dataset.
Elle doit obligatoirement implémenter :
  - __len__()     : combien d'images y a-t-il ?
  - __getitem__() : donne-moi l'image numéro i

Ensuite on enveloppe ce Dataset dans un DataLoader qui va créer les batchs
automatiquement et mélanger les données à chaque epoch.

On définit aussi ici les "transforms" : les transformations appliquées
à chaque image avant de la passer au réseau (redimensionnement, flip, normalisation...).
"""

import os
from PIL import Image                      # pour ouvrir les images
from torch.utils.data import Dataset
from torchvision import transforms


# Statistiques de normalisation ImageNet
# Ces valeurs (moyenne et écart-type par canal RGB) ont été calculées
# sur tout ImageNet. On les réutilise pour VGG16 et ResNet18 car ces
# modèles ont été conçus pour des images normalisées de cette façon.
# Pour LeNet-5, on ne les utilise PAS (le modèle est entraîné from scratch).
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]


def get_transforms(mode, model_type='vgg'):
    """
    Retourne les transformations à appliquer aux images.

    Paramètres :
        mode       : 'train' ou 'val'
        model_type : 'lenet' (32x32, pas de normalisation ImageNet)
                     'vgg' ou 'resnet' (224x224, normalisation ImageNet)

    En entraînement, on ajoute de l'augmentation de données (flip, crop aléatoire...)
    pour montrer des variations de la même image → le modèle généralise mieux.
    En validation, on applique juste ce qui est nécessaire pour avoir une taille fixe.
    """

    if model_type == 'lenet':
        # LeNet travaille sur des petites images 32x32
        # Pas de normalisation ImageNet car le modèle est entraîné from scratch
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((36, 36)),           # un peu plus grand pour le crop
                transforms.RandomCrop(32),             # crop aléatoire → simule différents cadrages
                transforms.RandomHorizontalFlip(),     # miroir horizontal aléatoire
                transforms.ColorJitter(brightness=0.2, contrast=0.2),   # variation de luminosité
                transforms.ToTensor(),                 # PIL Image → Tensor PyTorch (valeurs entre 0 et 1)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])

    else:
        # VGG16 et ResNet18 : images 224x224 normalisées selon les stats ImageNet
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),     # crop + zoom aléatoire sur 224x224
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                # La normalisation centre les valeurs autour de 0 avec un écart-type de ~1
                # → les gradients se propagent mieux dans le réseau
                transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),                # côté court → 256px
                transforms.CenterCrop(224),            # crop centré de 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
            ])


class CustomImageDataset(Dataset):
    """
    Dataset qui charge des images depuis un dossier organisé en sous-dossiers par classe.

    Structure attendue :
        dossier_racine/
            classe_A/    image1.jpg   image2.png   ...
            classe_B/    image1.jpg   ...

    Les labels sont attribués par ordre alphabétique des sous-dossiers :
        classe_A → 0
        classe_B → 1
    """

    def __init__(self, dossier, transform=None):
        """
        Paramètres :
            dossier   : chemin vers train/ ou val/
            transform : transformations à appliquer (résultat de get_transforms)
        """
        self.dossier   = dossier
        self.transform = transform

        # Extensions acceptées (on ignore les autres fichiers)
        ext_ok = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # Détection automatique des classes (= sous-dossiers)
        # sorted() → ordre alphabétique → labels reproductibles
        self.classes = sorted([
            nom for nom in os.listdir(dossier)
            if os.path.isdir(os.path.join(dossier, nom))
        ])

        if not self.classes:
            raise ValueError(f'Aucun sous-dossier trouvé dans {dossier}')

        # Dictionnaire : nom_classe → indice numérique
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Construction de la liste de tous les échantillons : [(chemin, label), ...]
        self.samples = []
        for classe in self.classes:
            dossier_classe = os.path.join(dossier, classe)
            label = self.class_to_idx[classe]
            for fichier in os.listdir(dossier_classe):
                if os.path.splitext(fichier)[1].lower() in ext_ok:
                    self.samples.append((os.path.join(dossier_classe, fichier), label))

        if not self.samples:
            raise ValueError(f'Aucune image trouvée dans {dossier}')

        print(f'  Dataset "{dossier}" : {len(self.samples)} images | classes : {self.class_to_idx}')

    def __len__(self):
        """ PyTorch appelle cette méthode pour savoir combien d'images il y a. """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        PyTorch appelle cette méthode pour récupérer un échantillon.
        Elle est appelée automatiquement par le DataLoader pour constituer chaque batch.

        Retour :
            (image_tensor, label)  — image sous forme de Tensor, label entier
        """
        chemin, label = self.samples[idx]

        # On force la lecture en RGB même si l'image est en niveaux de gris
        # → toujours 3 canaux, cohérent avec nos modèles
        image = Image.open(chemin).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def comptes_par_classe(self):
        """ Retourne le nombre d'images par classe (pour vérifier l'équilibre). """
        comptes = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            comptes[self.classes[label]] += 1
        return comptes
