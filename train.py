"""
train.py  —  Boucles d'entraînement et d'évaluation

Ce fichier contient le cœur de l'apprentissage.

Comment fonctionne un epoch d'entraînement ?
  Pour chaque batch d'images :
    1. On prédit les classes avec le modèle         (forward pass)
    2. On calcule l'erreur entre prédiction et vérité  (loss)
    3. On calcule comment modifier les poids pour réduire l'erreur  (backward)
    4. On met à jour les poids                          (optimizer.step)

C'est ce cycle qu'on répète des milliers de fois.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm   # affiche une barre de progression dans le terminal


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Entraîne le modèle pendant un epoch (= une passe complète sur toutes les données).

    Paramètres :
        model     : le réseau de neurones à entraîner
        loader    : DataLoader qui fournit les batchs d'images
        criterion : la fonction de perte (ex: CrossEntropyLoss)
        optimizer : l'algorithme qui met à jour les poids (ex: Adam)
        device    : 'cuda' (GPU) ou 'cpu'

    Retour :
        (loss_moyenne, accuracy)  sur tout l'epoch
    """

    # model.train() active certains comportements spécifiques à l'entraînement :
    # - Dropout : désactive aléatoirement des neurones (régularisation)
    # - BatchNorm : utilise les statistiques du batch courant
    # Sans ce mode, le modèle ne se comporterait pas correctement
    model.train()

    total_loss    = 0.0
    nb_corrects   = 0
    nb_total      = 0

    # tqdm affiche une barre de progression : [=====>    ] 50% ...
    for images, labels in tqdm(loader, desc='  Train', leave=False):

        # On déplace les données sur le bon appareil (GPU ou CPU)
        # Si le modèle est sur GPU mais les données sur CPU → erreur
        images = images.to(device)
        labels = labels.to(device)

        # ÉTAPE 1 : Remise à zéro des gradients
        # En PyTorch, les gradients s'ACCUMULENT par défaut entre les batchs.
        # Si on ne remet pas à zéro, les gradients du batch précédent
        # s'ajoutent à ceux du batch courant → les poids évoluent mal.
        optimizer.zero_grad()

        # ÉTAPE 2 : Forward pass — le modèle prédit les classes
        # logits = valeurs brutes avant softmax, shape [batch, nb_classes]
        logits = model(images)

        # ÉTAPE 3 : Calcul de la perte
        # CrossEntropyLoss mesure à quel point les prédictions sont éloignées
        # de la vérité. Elle attend : logits [B, C] et labels [B] (entiers).
        perte = criterion(logits, labels)

        # ÉTAPE 4 : Backward pass — calcul des gradients
        # PyTorch parcourt le graphe de calcul en sens inverse et calcule
        # ∂perte/∂w pour chaque poids w du réseau.
        perte.backward()

        # ÉTAPE 5 : Mise à jour des poids
        # Adam (ou SGD) utilise les gradients calculés pour modifier les poids :
        # w = w - lr * gradient
        optimizer.step()

        # Accumulation des statistiques pour calculer les métriques globales
        total_loss  += perte.item() * images.size(0)   # .item() convertit le Tensor en float Python

        # La prédiction = indice du logit le plus élevé
        predictions  = logits.argmax(dim=1)
        nb_corrects += (predictions == labels).sum().item()
        nb_total    += images.size(0)

    loss_moy = total_loss / nb_total
    accuracy = nb_corrects / nb_total

    return loss_moy, accuracy


def eval_epoch(model, loader, criterion, device):
    """
    Évalue le modèle sur le jeu de validation SANS modifier les poids.

    Paramètres :
        model     : le réseau de neurones à évaluer
        loader    : DataLoader de validation
        criterion : fonction de perte
        device    : 'cuda' ou 'cpu'

    Retour :
        (loss_moyenne, accuracy, f1_score)
    """

    # model.eval() désactive Dropout et met BatchNorm en mode évaluation
    # → les prédictions sont déterministes et comparables entre epochs
    model.eval()

    total_loss  = 0.0
    nb_corrects = 0
    nb_total    = 0
    toutes_preds  = []
    tous_labels   = []

    # torch.no_grad() désactive le calcul des gradients
    # Pourquoi ? En validation on ne fait pas de backward, donc inutile de stocker
    # les gradients. Ça économise ~50% de mémoire et accélère le calcul.
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Val  ', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            perte  = criterion(logits, labels)

            total_loss  += perte.item() * images.size(0)
            predictions  = logits.argmax(dim=1)
            nb_corrects += (predictions == labels).sum().item()
            nb_total    += images.size(0)

            # On garde les prédictions et labels pour calculer le F1 à la fin
            toutes_preds.extend(predictions.cpu().numpy())
            tous_labels.extend(labels.cpu().numpy())

    loss_moy = total_loss / nb_total
    accuracy = nb_corrects / nb_total

    # F1-score : meilleure métrique que l'accuracy quand les classes sont déséquilibrées
    # average='binary' convient pour 2 classes
    # zero_division=0 évite une erreur si une classe n'est jamais prédite
    score_f1 = f1_score(tous_labels, toutes_preds, average='binary', zero_division=0)

    return loss_moy, accuracy, score_f1


def entrainer_modele(model, train_loader, val_loader, criterion, optimizer,
                     nb_epochs, device, nom_modele, scheduler=None,
                     wandb_run=None, dossier_save='.'):
    """
    Boucle d'entraînement complète : lance train_epoch + eval_epoch pendant nb_epochs.

    Gère aussi :
    - La sauvegarde automatique du meilleur modèle (sur val_loss)
    - Le logging vers wandb à chaque epoch
    - L'affichage des métriques dans le terminal

    Paramètres :
        model        : réseau à entraîner
        train_loader : DataLoader entraînement
        val_loader   : DataLoader validation
        criterion    : fonction de perte
        optimizer    : optimiseur
        nb_epochs    : nombre d'epochs
        device       : 'cuda' ou 'cpu'
        nom_modele   : nom pour les sauvegardes et les logs (ex: 'lenet5_relu')
        scheduler    : (optionnel) scheduler qui réduit le lr si le modèle stagne
        wandb_run    : (optionnel) run wandb pour envoyer les métriques en temps réel
        dossier_save : où sauvegarder le fichier .pth du meilleur modèle

    Retour :
        historique : dict avec toutes les métriques par epoch
    """
    import os

    meilleure_val_loss  = float('inf')   # on veut minimiser la val_loss
    chemin_best = os.path.join(dossier_save, f'{nom_modele}_best.pth')

    # On stocke toutes les métriques pour tracer les courbes après
    historique = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_f1':     [], 'lr':        [],
    }

    print(f'\n{"="*60}')
    print(f'  {nom_modele}  —  {nb_epochs} epochs  —  device: {device}')
    print(f'{"="*60}')

    for epoch in range(1, nb_epochs + 1):

        # --- Entraînement ---
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # --- Évaluation ---
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        # Le scheduler réduit le lr si la val_loss n'améliore plus
        if scheduler is not None:
            scheduler.step(val_loss)

        # Récupération du lr APRÈS ajustement par le scheduler
        # (important : avant le step, on aurait l'ancien lr)
        lr_actuel = optimizer.param_groups[0]['lr']

        # Sauvegarde si c'est le meilleur modèle vu jusqu'ici
        if val_loss < meilleure_val_loss:
            meilleure_val_loss = val_loss
            torch.save(model.state_dict(), chemin_best)
            # state_dict() = dictionnaire {nom_couche: tensor_poids}
            # C'est la façon standard de sauvegarder un modèle PyTorch
            marque = ' ← meilleur'
        else:
            marque = ''

        # Affichage dans le terminal
        print(
            f'  Epoch {epoch:3d}/{nb_epochs} | '
            f'train loss={train_loss:.4f} acc={train_acc:.3f} | '
            f'val loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} | '
            f'lr={lr_actuel:.1e}{marque}'
        )

        # Envoi des métriques vers wandb (si connecté)
        if wandb_run is not None:
            wandb_run.log({
                'epoch':          epoch,
                'train/loss':     train_loss,
                'train/accuracy': train_acc,
                'val/loss':       val_loss,
                'val/accuracy':   val_acc,
                'val/f1':         val_f1,
                'learning_rate':  lr_actuel,
            })

        # Stockage dans l'historique local
        historique['train_loss'].append(train_loss)
        historique['train_acc'].append(train_acc)
        historique['val_loss'].append(val_loss)
        historique['val_acc'].append(val_acc)
        historique['val_f1'].append(val_f1)
        historique['lr'].append(lr_actuel)

    print(f'\n  Meilleur modèle sauvegardé → {chemin_best}  (val_loss={meilleure_val_loss:.4f})')

    if wandb_run is not None:
        wandb_run.save(chemin_best)   # upload vers wandb cloud

    return historique
