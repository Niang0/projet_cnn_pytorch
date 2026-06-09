"""
models/vgg16.py  —  VGG16 codé à la main en PyTorch

VGG16 (Oxford, 2014) a été une révélation à l'époque : une architecture
très simple (que des filtres 3x3) mais très profonde (16 couches).
L'idée clé : empiler plusieurs petits filtres 3x3 est plus efficace
qu'un seul grand filtre, car ça donne plus de profondeur avec moins de paramètres.

Résultat : 138 millions de paramètres et d'excellentes performances sur ImageNet.
Pour nous, on l'entraîne de zéro ("from scratch") sur notre dataset binaire.

IMPORTANT : pas touche à torchvision.models, tout est codé manuellement.
"""

import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    VGG16 from scratch pour 2 classes.

    L'architecture se divise en deux grandes parties :

    1) LES BLOCS CONVOLUTIFS (extraction de caractéristiques)
       5 blocs, chacun contenant 2 ou 3 convolutions + un MaxPool
       Les filtres deviennent de plus en plus nombreux (profondeur)
       mais l'image devient de plus en plus petite (résolution)

         Bloc 1 : 2 conv  3→64    + MaxPool   224x224 → 112x112
         Bloc 2 : 2 conv  64→128  + MaxPool   112x112 →  56x56
         Bloc 3 : 3 conv  128→256 + MaxPool    56x56  →  28x28
         Bloc 4 : 3 conv  256→512 + MaxPool    28x28  →  14x14
         Bloc 5 : 3 conv  512→512 + MaxPool    14x14  →   7x7

    2) LA TÊTE DE CLASSIFICATION (linéaire)
       AdaptiveAvgPool → aplatissement → 3 couches linéaires
    """

    def __init__(self):
        super(VGG16, self).__init__()

        # On construit les 5 blocs convolutifs
        # La fonction _bloc() s'occupe de créer les couches conv + relu + maxpool
        self.bloc1 = self._bloc(nb_conv=2, entree=3,   sortie=64)
        self.bloc2 = self._bloc(nb_conv=2, entree=64,  sortie=128)
        self.bloc3 = self._bloc(nb_conv=3, entree=128, sortie=256)
        self.bloc4 = self._bloc(nb_conv=3, entree=256, sortie=512)
        self.bloc5 = self._bloc(nb_conv=3, entree=512, sortie=512)

        # AdaptiveAvgPool : garantit que la sortie est toujours 7x7
        # Peu importe si l'image d'entrée était 224x224 ou autre,
        # on obtient toujours [batch, 512, 7, 7] en sortie.
        # C'est pratique pour rendre le modèle flexible.
        self.pool_adaptatif = nn.AdaptiveAvgPool2d((7, 7))

        # Tête de classification : 3 couches linéaires
        # Taille d'entrée : 512 * 7 * 7 = 25088 valeurs après aplatissement
        #
        # Le Dropout(0.5) désactive aléatoirement 50% des neurones pendant l'entraînement.
        # Pourquoi ? Pour forcer le réseau à ne pas trop mémoriser les données d'entraînement
        # (= régularisation, réduit l'overfitting).
        # Pendant l'évaluation, Dropout est automatiquement désactivé par model.eval().
        self.classification = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 2),    # 2 logits en sortie = 2 classes
        )

        # Initialisation des poids avec Xavier
        # Par défaut PyTorch utilise une initialisation aléatoire simple,
        # mais Xavier est mieux adaptée aux réseaux profonds car elle
        # maintient une variance stable à travers les couches.
        self._init_poids()

    def _bloc(self, nb_conv, entree, sortie):
        """
        Crée un bloc convolutif VGG : nb_conv couches [Conv+ReLU] puis MaxPool.

        Paramètres :
            nb_conv : 2 ou 3 selon le bloc
            entree  : nombre de canaux en entrée du bloc
            sortie  : nombre de canaux en sortie (tous les Conv du bloc ont ce nombre)

        Retour :
            nn.Sequential contenant toutes les couches du bloc

        Exemple pour bloc1 (nb_conv=2, entree=3, sortie=64) :
            Conv(3→64, 3x3) → ReLU → Conv(64→64, 3x3) → ReLU → MaxPool(2x2)
        """
        couches = []

        for i in range(nb_conv):
            # Premier conv du bloc : utilise le nombre de canaux d'entrée
            # Les suivants : sortie → sortie (même nombre de canaux)
            ch_in = entree if i == 0 else sortie

            # padding=1 avec un filtre 3x3 : conserve la résolution spatiale
            # Sans padding : 224 - 3 + 1 = 222  (rétrécit)
            # Avec padding=1 : 224 - 3 + 1 + 2 = 224  (taille conservée)
            couches.append(nn.Conv2d(ch_in, sortie, kernel_size=3, padding=1))
            couches.append(nn.ReLU(inplace=True))  # inplace=True économise la mémoire

        # MaxPool 2x2 à la fin du bloc : divise la résolution par 2
        # On prend le maximum dans chaque fenêtre 2x2 → garde l'information la plus forte
        couches.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*couches)

    def _init_poids(self):
        """
        Initialise les poids avec la méthode Xavier (Glorot).

        Sans bonne initialisation, les gradients peuvent exploser ou disparaître
        dans un réseau aussi profond. Xavier calcule la variance initiale
        en fonction du nombre de connexions entrantes et sortantes.
        """
        for couche in self.modules():
            if isinstance(couche, nn.Conv2d) or isinstance(couche, nn.Linear):
                nn.init.xavier_uniform_(couche.weight)
                nn.init.zeros_(couche.bias)

    def forward(self, x):
        """
        Chemin de l'image jusqu'aux logits de sortie.

        Paramètre :
            x : [batch, 3, 224, 224]
        Retour :
            logits : [batch, 2]
        """
        # Les 5 blocs convolutifs réduisent progressivement la résolution
        x = self.bloc1(x)            # [batch,  64, 112, 112]
        x = self.bloc2(x)            # [batch, 128,  56,  56]
        x = self.bloc3(x)            # [batch, 256,  28,  28]
        x = self.bloc4(x)            # [batch, 512,  14,  14]
        x = self.bloc5(x)            # [batch, 512,   7,   7]

        x = self.pool_adaptatif(x)   # [batch, 512,   7,   7]  (déjà 7x7, sécurité)

        # Aplatissement 4D → 2D avant les couches linéaires
        # [batch, 512, 7, 7]  →  [batch, 25088]
        x = x.view(x.size(0), -1)

        x = self.classification(x)   # [batch, 2]

        return x

    def nb_parametres(self):
        """ Nombre de paramètres entraînables. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print('=== Test VGG16 ===')
    modele = VGG16()
    x = torch.randn(2, 3, 224, 224)
    y = modele(x)
    print(f'  Sortie  : {list(y.shape)}')
    print(f'  Params  : {modele.nb_parametres():,}')
    print(f'  Mémoire : ~{modele.nb_parametres() * 4 / 1e6:.0f} MB (float32)')
