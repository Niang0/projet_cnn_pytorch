"""
utils.py  —  Fonctions utilitaires : seeds, visualisations, métriques

"""

import os
import random  # noqa: F401 — utilisé dans inspecter_dataset (random.sample)
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader


def fixer_seeds(seed=42):
    """
    Fixe toutes les sources d'aléatoire pour rendre les résultats reproductibles.

    Sans ça, chaque exécution du script donnerait des résultats différents
    (poids initiaux différents, ordre des données différent...).
    Avec une seed fixe, tout le monde qui exécute le code obtient les mêmes résultats.

    Il faut fixer 3 sources : Python, NumPy et PyTorch (CPU + GPU).
    """
    random.seed(seed)          # module random de Python standard
    np.random.seed(seed)       # NumPy (utilisé par scikit-learn)
    torch.manual_seed(seed)    # PyTorch CPU
    torch.cuda.manual_seed_all(seed)   # PyTorch GPU (tous les GPUs)

    # Mode déterministe pour CuDNN (bibliothèque NVIDIA pour les convolutions sur GPU)
    # Légèrement plus lent mais garantit la reproductibilité sur GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f'  Seeds fixées à {seed}')


def inspecter_dataset(dataset, nb_exemples=8, titre='Dataset'):
    """
    Affiche des infos sur le dataset et montre des exemples d'images.

    C'est important de faire ça avant d'entraîner pour :
    - Vérifier que les images se chargent correctement
    - Voir si les classes sont équilibrées (si une classe a 10x plus d'images → problème)
    - Avoir une idée visuelle de ce qu'on classifie

    Paramètres :
        dataset     : instance de CustomImageDataset
        nb_exemples : combien d'images afficher (8 par défaut)
        titre       : titre du graphique
    """
    print(f'\n{"─"*50}')
    print(f'  {titre}')
    print(f'{"─"*50}')

    comptes = dataset.comptes_par_classe()
    total   = len(dataset)

    print(f'  Total : {total} images')
    for classe, nb in comptes.items():
        print(f'    {classe} : {nb} images ({nb/total*100:.1f}%)')

    # Alerte si déséquilibre important
    valeurs = list(comptes.values())
    if max(valeurs) / max(1, min(valeurs)) > 2.0:
        print('  ⚠ Déséquilibre > 2:1 → envisager class_weight dans CrossEntropyLoss')
    else:
        print('  ✓ Classes équilibrées')

    # Affichage de nb_exemples images au hasard
    indices = random.sample(range(total), min(nb_exemples, total))
    cols = 4
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]

        # Conversion Tensor [C, H, W] → numpy [H, W, C] pour matplotlib
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # Dénormalisation ImageNet si l'image a été normalisée
        # (valeurs hors [0,1] → couleurs incorrectes sans cette étape)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        axes[i].set_title(dataset.classes[label], fontsize=9)
        axes[i].axis('off')

    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    plt.suptitle(titre, fontweight='bold')
    plt.tight_layout()
    nom_fichier = f'exemples_{titre.replace(" ", "_")}.png'
    plt.savefig(nom_fichier, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'  → Sauvegardé : {nom_fichier}')


def matrice_confusion(model, loader, device, noms_classes, nom_modele='modele', wandb_run=None):
    """
    Calcule et affiche la matrice de confusion sur le jeu de validation.

    La matrice de confusion montre :
    - Diagonale : prédictions correctes
    - Hors diagonale : erreurs (ex: ligne=chien, col=chat → images de chiens classées en chat)

    Paramètres :
        model        : modèle entraîné
        loader       : DataLoader de validation
        device       : 'cuda' ou 'cpu'
        noms_classes : liste des noms des classes
        nom_modele   : pour nommer le fichier sauvegardé
        wandb_run    : si fourni, upload la figure vers wandb
    """
    model.eval()
    preds_list  = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())

    mat = confusion_matrix(labels_list, preds_list)

    # Rapport texte avec précision, rappel, F1 par classe
    print(f'\n  Rapport — {nom_modele}')
    print(classification_report(labels_list, preds_list, target_names=noms_classes))

    # Affichage graphique de la matrice
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(mat, display_labels=noms_classes).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Matrice de confusion — {nom_modele}', fontweight='bold')
    plt.tight_layout()

    chemin = f'confusion_{nom_modele}.png'
    plt.savefig(chemin, dpi=120, bbox_inches='tight')
    plt.show()

    if wandb_run is not None:
        import wandb
        wandb_run.log({f'confusion/{nom_modele}': wandb.Image(chemin)})

    return mat


def tracer_courbes(historique, nom_modele='modele'):
    """
    Trace les courbes loss et accuracy (train vs val) sur 2 graphiques côte à côte.

    Ces courbes servent à diagnostiquer le comportement du modèle :
    - Train et val proches → bon équilibre
    - Train ↓ mais val ↗  → overfitting (le modèle mémorise au lieu d'apprendre)
    - Train et val élevées → underfitting (modèle pas assez puissant ou trop peu d'epochs)

    Paramètre :
        historique : dict retourné par entrainer_modele()
    """
    nb_epochs = len(historique['train_loss'])
    epochs    = range(1, nb_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Courbe de perte
    ax1.plot(epochs, historique['train_loss'], label='Train', color='steelblue')
    ax1.plot(epochs, historique['val_loss'],   label='Val',   color='tomato')
    ax1.set_title('Loss (perte)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropyLoss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Courbe d'accuracy
    ax2.plot(epochs, historique['train_acc'], label='Train', color='steelblue')
    ax2.plot(epochs, historique['val_acc'],   label='Val',   color='tomato')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(nom_modele, fontweight='bold')
    plt.tight_layout()
    chemin = f'courbes_{nom_modele}.png'
    plt.savefig(chemin, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'  → Courbes sauvegardées : {chemin}')


def comparer_activations_lenet(historiques):
    """
    Compare les courbes de val_loss et val_acc des 3 variantes d'activation.

    Paramètre :
        historiques : {'tanh': hist_tanh, 'relu': hist_relu, 'sigmoid': hist_sigmoid}
    """
    couleurs = {'tanh': 'steelblue', 'relu': 'forestgreen', 'sigmoid': 'darkorange'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for nom, hist in historiques.items():
        epochs  = range(1, len(hist['val_loss']) + 1)
        couleur = couleurs.get(nom, 'gray')
        ax1.plot(epochs, hist['val_loss'], label=f'LeNet5-{nom}', color=couleur)
        ax2.plot(epochs, hist['val_acc'],  label=f'LeNet5-{nom}', color=couleur)

    ax1.set_title('Val Loss — comparaison des activations')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title('Val Accuracy — comparaison des activations')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1)
    # Ligne rouge pointillée à 80% : seuil mentionné dans le sujet
    ax2.axhline(0.80, color='red', linestyle='--', alpha=0.4, label='seuil 80%')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('LeNet-5 : impact du choix de la fonction d\'activation', fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparaison_activations.png', dpi=120, bbox_inches='tight')
    plt.show()


def comparer_trois_modeles(resultats):
    """
    Trace val_loss, val_acc et val_f1 des 3 architectures sur le même graphique.

    Paramètre :
        resultats : {'LeNet5-relu': hist, 'VGG16': hist, 'ResNet18': hist}
    """
    couleurs = ['steelblue', 'darkorange', 'forestgreen']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metriques = [('val_loss', 'Val Loss'), ('val_acc', 'Val Accuracy'), ('val_f1', 'Val F1-score')]

    for ax, (cle, titre) in zip(axes, metriques):
        for (nom, hist), couleur in zip(resultats.items(), couleurs):
            epochs = range(1, len(hist[cle]) + 1)
            ax.plot(epochs, hist[cle], label=nom, color=couleur)
        ax.set_title(titre)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Comparaison LeNet-5 vs VGG16 vs ResNet18', fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparaison_trois_modeles.png', dpi=120, bbox_inches='tight')
    plt.show()


def tableau_parametres(modeles):
    """
    Affiche un tableau récapitulatif du nombre de paramètres par modèle.

    Paramètre :
        modeles : {'nom': instance_nn.Module, ...}
    """
    print(f'\n{"="*55}')
    print(f'  {"Modèle":<22} {"Entraînables":>15} {"Total":>12}')
    print(f'{"─"*55}')
    for nom, m in modeles.items():
        total  = sum(p.numel() for p in m.parameters())
        entraines = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f'  {nom:<22} {entraines:>15,} {total:>12,}')
    print(f'{"="*55}')
