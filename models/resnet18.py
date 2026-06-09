"""
models/resnet18.py  —  ResNet18 adapté par transfer learning

Le transfer learning c'est une idée simple mais très puissante :
au lieu d'entraîner un réseau de zéro (ce qui nécessite beaucoup de données
et beaucoup de temps), on réutilise un réseau déjà entraîné sur une grande tâche.

ResNet18 a été entraîné sur ImageNet : 1,2 million d'images, 1000 catégories.
Il a "appris" à reconnaître des formes, des textures, des contours...
Ces connaissances sont utiles pour presque n'importe quel problème de vision.

Notre stratégie :
  1. Charger ResNet18 avec ses poids ImageNet
  2. Geler le backbone (ne plus le modifier)
  3. Remplacer uniquement la dernière couche par notre classificateur binaire
  4. N'entraîner que cette nouvelle couche

Pourquoi geler le backbone ?
  → Avec peu de données, ré-entraîner 11M de paramètres mènerait à de l'overfitting.
  → La couche finale a seulement 512*2 + 2 = 1026 paramètres → rapide à converger.
  → Les features ImageNet sont génériques et suffisamment bonnes pour notre tâche.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18FineTune(nn.Module):
    """
    ResNet18 pré-entraîné sur ImageNet, adapté à notre classification binaire.

    On garde tout le backbone tel quel (gelé), et on change juste
    la dernière couche fc : Linear(512, 1000) → Linear(512, 2).
    """

    def __init__(self, nb_classes=2, geler_backbone=True):
        """
        Paramètres :
            nb_classes      : nombre de classes (2 pour notre problème binaire)
            geler_backbone  : True = on gèle le backbone (recommandé pour débuter)
        """
        super(ResNet18FineTune, self).__init__()

        # Chargement de ResNet18 avec les poids officiels ImageNet
        # weights=IMAGENET1K_V1 : première version des poids officiels PyTorch
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Gel du backbone si demandé
        # requires_grad=False signifie : "ne calcule pas le gradient pour ce paramètre"
        # Sans gradient → pas de mise à jour pendant backpropagation
        # → les poids restent exactement ceux d'ImageNet
        if geler_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Remplacement de la couche de sortie
        # ResNet18 termine par self.resnet.fc = Linear(512, 1000)
        # On la remplace par Linear(512, nb_classes)
        # Cette nouvelle couche a requires_grad=True par défaut → sera entraînée
        nb_features = self.resnet.fc.in_features   # = 512 pour ResNet18
        self.resnet.fc = nn.Linear(nb_features, nb_classes)

    def forward(self, x):
        """
        Paramètre :
            x : [batch, 3, 224, 224]
        Retour :
            logits : [batch, 2]
        """
        # On passe simplement par le ResNet complet (backbone gelé + nouvelle tête)
        return self.resnet(x)

    def stats_parametres(self):
        """
        Affiche combien de paramètres sont entraînés vs gelés.
        Utile pour vérifier que le backbone est bien gelé.
        """
        total       = sum(p.numel() for p in self.parameters())
        entraines   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        geles       = total - entraines
        return {'total': total, 'entrainables': entraines, 'geles': geles}

    def degeler_tout(self):
        """
        Optionnel : dégèle tout le réseau pour un fine-tuning complet.
        À utiliser après une première phase d'entraînement de la tête seule.
        Attention : nécessite beaucoup plus de données et un lr très faible (ex: 1e-5).
        """
        for param in self.resnet.parameters():
            param.requires_grad = True
        print('Backbone entièrement dégelé — pensez à réduire le learning rate !')


if __name__ == '__main__':
    print('=== Test ResNet18 Fine-tuning ===')
    modele = ResNet18FineTune(nb_classes=2, geler_backbone=True)
    stats  = modele.stats_parametres()
    print(f'  Total paramètres     : {stats["total"]:,}')
    print(f'  Paramètres entraînés : {stats["entrainables"]:,}  ← seulement la tête fc !')
    print(f'  Paramètres gelés     : {stats["geles"]:,}')

    x = torch.randn(4, 3, 224, 224)
    y = modele(x)
    print(f'  Sortie shape         : {list(y.shape)}')
