"""
models/lenet5.py  —  LeNet-5 codé à la main en PyTorch

LeNet-5 a été inventé par Yann LeCun en 1998 pour lire les chiffres
écrits à la main sur des chèques bancaires. C'est l'un des tout premiers
réseaux de neurones convolutifs qui a vraiment "marché".

On l'adapte ici pour nos images RGB 32x32 (au lieu de 28x28 niveaux de gris)
et on prédit 2 classes au lieu de 10 chiffres.

On peut aussi choisir la fonction d'activation : tanh, relu ou sigmoid.
C'est utile pour comparer leur comportement dans la section 3.2.1 du devoir.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    Architecture LeNet-5 adaptée à nos besoins.

    Un réseau CNN c'est deux grandes parties :
      1) La partie "extraction" : des filtres convolutifs qui détectent
         des formes dans l'image (bords, textures, motifs...)
      2) La partie "classification" : des couches linéaires qui décident
         à quelle classe appartient l'image

    Ici la partie extraction a 3 couches conv (C1, C3, C5)
    et la partie classification a 2 couches linéaires (F6 + sortie).
    """

    def __init__(self, activation='tanh'):
        """
        Construit le réseau couche par couche.

        Paramètre :
            activation : quelle fonction d'activation utiliser
                         'tanh'    = activation originale du papier (LeCun 1998)
                         'relu'    = la plus utilisée aujourd'hui, évite le vanishing gradient
                         'sigmoid' = écrase les valeurs entre 0 et 1, souffre du vanishing gradient
        """
        super(LeNet5, self).__init__()

        # On vérifie que l'utilisateur n'a pas fait de faute de frappe
        if activation not in ('tanh', 'relu', 'sigmoid'):
            raise ValueError(f"Activation '{activation}' inconnue. Choisir : tanh, relu ou sigmoid")

        self.activation_choisie = activation

        # ----------------------------------------------------------------
        # COUCHE C1 : première convolution
        # Entrée  : image RGB  → shape [batch, 3, 32, 32]
        # Sortie  : 6 cartes   → shape [batch, 6, 28, 28]
        #
        # Un filtre 5x5 "glisse" sur l'image et calcule des produits scalaires.
        # Avec 6 filtres différents, on obtient 6 "cartes de caractéristiques"
        # qui détectent chacune un type de motif différent.
        # Sans padding, 32 - 5 + 1 = 28  → la carte rétrécit de 32 à 28.
        # ----------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

        # ----------------------------------------------------------------
        # COUCHE S2 : premier sous-échantillonnage (pooling)
        # Entrée  : [batch, 6, 28, 28]
        # Sortie  : [batch, 6, 14, 14]
        #
        # On divise chaque carte en petits blocs 2x2 et on prend la moyenne.
        # Ça réduit la taille de moitié → moins de calculs, plus robuste
        # aux petits décalages dans l'image.
        # ----------------------------------------------------------------
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # ----------------------------------------------------------------
        # COUCHE C3 : deuxième convolution
        # Entrée  : [batch, 6, 14, 14]
        # Sortie  : [batch, 16, 10, 10]
        #
        # On passe de 6 à 16 filtres → le réseau détecte des motifs plus complexes
        # qui combinent les caractéristiques trouvées par C1.
        # ----------------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # ----------------------------------------------------------------
        # COUCHE S4 : deuxième sous-échantillonnage
        # Entrée  : [batch, 16, 10, 10]
        # Sortie  : [batch, 16, 5, 5]
        # ----------------------------------------------------------------
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # ----------------------------------------------------------------
        # COUCHE C5 : troisième convolution (= équivalent d'une couche linéaire)
        # Entrée  : [batch, 16, 5, 5]
        # Sortie  : [batch, 120, 1, 1]
        #
        # Le filtre 5x5 couvre toute la carte restante → chaque "neurone"
        # de sortie voit l'image entière. C'est le passage de "spatial" à "abstrait".
        # ----------------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # ----------------------------------------------------------------
        # COUCHE F6 : première couche fully-connected (linéaire)
        # Entrée  : 120 valeurs aplaties
        # Sortie  : 84 valeurs
        #
        # "Fully-connected" = chaque neurone est connecté à tous les neurones
        # de la couche précédente. C'est là que la classification se fait vraiment.
        # ----------------------------------------------------------------
        self.fc1 = nn.Linear(120, 84)

        # ----------------------------------------------------------------
        # COUCHE DE SORTIE
        # Entrée  : 84 valeurs
        # Sortie  : 2 valeurs (une par classe)
        #
        # Ces 2 valeurs s'appellent des "logits". Ce ne sont pas des probabilités.
        # CrossEntropyLoss s'en charge ensuite (elle applique softmax en interne).
        # ----------------------------------------------------------------
        self.fc2 = nn.Linear(84, 2)

        # On stocke la fonction d'activation choisie
        self.activation = self._get_activation(activation)

    def _get_activation(self, nom):
        """
        Retourne la couche d'activation correspondant au nom.
        Petite fonction helper pour éviter un gros if/elif dans __init__.
        """
        if nom == 'tanh':
            return nn.Tanh()      # sortie entre -1 et 1
        elif nom == 'relu':
            return nn.ReLU()      # max(0, x) — simple et efficace
        else:
            return nn.Sigmoid()   # sortie entre 0 et 1

    def forward(self, x):
        """
        Le "forward pass" : comment les données traversent le réseau.

        PyTorch appelle cette fonction automatiquement quand on fait model(images).
        On décrit juste le chemin de l'entrée vers la sortie.

        Paramètre :
            x : batch d'images de shape [batch, 3, 32, 32]
        Retour :
            logits de shape [batch, 2]
        """

        # --- Partie convolutive : on extrait les caractéristiques ---

        # conv1 → activation → pooling
        # L'ordre conv → activation est standard : on filtre d'abord, puis on "active"
        x = self.activation(self.conv1(x))   # [batch, 6, 28, 28]
        x = self.pool1(x)                    # [batch, 6, 14, 14]

        # conv2 → activation → pooling
        x = self.activation(self.conv2(x))   # [batch, 16, 10, 10]
        x = self.pool2(x)                    # [batch, 16, 5, 5]

        # conv3 → activation
        x = self.activation(self.conv3(x))   # [batch, 120, 1, 1]

        # --- Passage de 4D à 2D ---
        # x.size(0) = taille du batch (ex: 64)
        # -1 dit à PyTorch de calculer automatiquement la deuxième dimension
        # [batch, 120, 1, 1]  →  [batch, 120]
        x = x.view(x.size(0), -1)

        # --- Partie classification : on prédit la classe ---
        x = self.activation(self.fc1(x))   # [batch, 84]
        x = self.fc2(x)                    # [batch, 2]  ← logits finaux

        return x

    def nb_parametres(self):
        """ Compte le nombre de paramètres entraînables (utile pour comparer les modèles). """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test rapide si on lance directement ce fichier : python models/lenet5.py
if __name__ == '__main__':
    print('=== Test LeNet-5 ===')
    batch_fictif = torch.randn(4, 3, 32, 32)   # 4 images RGB 32x32 aléatoires

    for act in ['tanh', 'relu', 'sigmoid']:
        modele = LeNet5(activation=act)
        sortie = modele(batch_fictif)
        print(f'  LeNet5-{act:7s} | sortie: {list(sortie.shape)} | params: {modele.nb_parametres():,}')
