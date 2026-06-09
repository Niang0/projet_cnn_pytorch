"""
main.py  —  Pipeline principal du projet CNN

Ce fichier orchestre tout :
  1. Inspection du dataset
  2. LeNet-5 (3 variantes d'activation : tanh, relu, sigmoid)
  3. VGG16 from scratch
  4. ResNet18 fine-tuning
  5. Comparaison finale + logs wandb

Lancement : python main.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from dataset        import CustomImageDataset, get_transforms
from train          import entrainer_modele
from utils          import (fixer_seeds, inspecter_dataset, matrice_confusion,
                             tracer_courbes, comparer_activations_lenet,
                             comparer_trois_modeles, tableau_parametres)
from models.lenet5  import LeNet5
from models.vgg16   import VGG16
from models.resnet18 import ResNet18FineTune


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION CENTRALISÉE
#  Tous les hyperparamètres sont ici → facile à modifier et à retrouver
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Données
    'train_dir'    : 'data/train',
    'val_dir'      : 'data/val',
    'checkpoints'  : 'checkpoints',

    # Reproductibilité
    'seed'         : 42,

    # LeNet-5 : petites images, modèle léger → on peut se permettre un grand batch
    'lenet_epochs' : 50,
    'lenet_batch'  : 64,
    'lenet_lr'     : 1e-3,

    # VGG16 : très lourd (138M params) → petit batch pour tenir en VRAM
    'vgg_epochs'   : 30,
    'vgg_batch'    : 16,
    'vgg_lr'       : 1e-4,   # lr plus faible : VGG est plus difficile à entraîner

    # ResNet18 : seulement la tête est entraînée → convergence rapide
    'resnet_epochs': 50,
    'resnet_batch' : 32,
    'resnet_lr'    : 1e-3,

    # Nom du projet wandb
    'wandb_project': 'devoir-cnn-pytorch',

    # Nombre de threads pour le chargement des données
    'num_workers'  : 2,
}

os.makedirs(CONFIG['checkpoints'], exist_ok=True)


def creer_dataloaders(train_dir, val_dir, model_type, batch_size, num_workers):
    """
    Crée les DataLoaders pour entraînement et validation.

    Un DataLoader c'est ce qui "distribue" les données par batchs au modèle.
    shuffle=True en train : mélange les images à chaque epoch → meilleure généralisation
    shuffle=False en val  : ordre constant → résultats reproductibles

    Paramètres :
        train_dir   : chemin dossier train
        val_dir     : chemin dossier val
        model_type  : 'lenet', 'vgg' ou 'resnet' (détermine la taille d'image)
        batch_size  : nombre d'images par batch
        num_workers : threads parallèles pour le chargement (0 = pas de parallélisme)

    Retour :
        (train_loader, val_loader, noms_classes)
    """
    train_ds = CustomImageDataset(train_dir, get_transforms('train', model_type))
    val_ds   = CustomImageDataset(val_dir,   get_transforms('val',   model_type))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    # pin_memory=True : transfert CPU→GPU plus rapide (si GPU disponible)

    return train_loader, val_loader, train_ds.classes


def lancer_run_wandb(nom_run, config_dict, project):
    """
    Initialise un run wandb avec les hyperparamètres.

    wandb.config stocke tous les hyperparamètres → on peut les retrouver
    plus tard sur le dashboard et comparer les runs entre eux.
    """
    return wandb.init(
        project = project,
        name    = nom_run,
        config  = config_dict,
        reinit  = True,   # autorise plusieurs runs dans le même script
    )


def main():

    # ── Étape 0 : setup ───────────────────────────────────────────────────────
    fixer_seeds(CONFIG['seed'])

    # Utilise le GPU si disponible, sinon CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device : {device}')
    if device.type == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')

    # ── Étape 1 : inspection du dataset ───────────────────────────────────────
    print('\n' + '='*60)
    print('  INSPECTION DU DATASET')
    print('='*60)

    ds_temp = CustomImageDataset(CONFIG['train_dir'], get_transforms('val', 'vgg'))
    inspecter_dataset(ds_temp, nb_exemples=8, titre='Exemples du jeu d\'entraînement')
    noms_classes = ds_temp.classes
    print(f'\n  Classes détectées : {noms_classes}')

    # ── Étape 2 : LeNet-5 — comparaison des 3 activations ────────────────────
    print('\n' + '='*60)
    print('  PARTIE 1 — LeNet-5 (tanh vs relu vs sigmoid)')
    print('='*60)

    historiques_lenet = {}

    for activation in ['tanh', 'relu', 'sigmoid']:
        nom_run = f'lenet5_{activation}'
        print(f'\n  → LeNet5-{activation.upper()}')

        train_loader, val_loader, _ = creer_dataloaders(
            CONFIG['train_dir'], CONFIG['val_dir'],
            model_type='lenet', batch_size=CONFIG['lenet_batch'],
            num_workers=CONFIG['num_workers']
        )

        modele    = LeNet5(activation=activation).to(device)
        optimizer = torch.optim.Adam(modele.parameters(), lr=CONFIG['lenet_lr'])
        criterion = nn.CrossEntropyLoss()
        # ReduceLROnPlateau : divise le lr par 2 si val_loss ne baisse plus pendant 5 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=False
        )

        run = lancer_run_wandb(nom_run, {
            'model': 'LeNet5', 'activation': activation,
            'lr': CONFIG['lenet_lr'], 'batch': CONFIG['lenet_batch'],
            'epochs': CONFIG['lenet_epochs'], 'seed': CONFIG['seed'],
            'input': '32x32', 'classes': noms_classes,
        }, CONFIG['wandb_project'])

        hist = entrainer_modele(
            modele, train_loader, val_loader, criterion, optimizer,
            nb_epochs=CONFIG['lenet_epochs'], device=device,
            nom_modele=nom_run, scheduler=scheduler,
            wandb_run=run, dossier_save=CONFIG['checkpoints']
        )

        matrice_confusion(modele, val_loader, device, noms_classes, nom_run, run)
        tracer_courbes(hist, nom_run)
        historiques_lenet[activation] = hist
        run.finish()

    comparer_activations_lenet(historiques_lenet)

    # ── Étape 3 : VGG16 from scratch ──────────────────────────────────────────
    print('\n' + '='*60)
    print('  PARTIE 2 — VGG16 From Scratch')
    print('='*60)

    train_loader, val_loader, _ = creer_dataloaders(
        CONFIG['train_dir'], CONFIG['val_dir'],
        model_type='vgg', batch_size=CONFIG['vgg_batch'],
        num_workers=CONFIG['num_workers']
    )

    modele_vgg    = VGG16().to(device)
    optimizer_vgg = torch.optim.Adam(modele_vgg.parameters(), lr=CONFIG['vgg_lr'])
    scheduler_vgg = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_vgg, mode='min', patience=5, factor=0.5, verbose=False
    )

    run_vgg = lancer_run_wandb('vgg16_scratch', {
        'model': 'VGG16', 'from_scratch': True, 'dropout': 0.5,
        'lr': CONFIG['vgg_lr'], 'batch': CONFIG['vgg_batch'],
        'epochs': CONFIG['vgg_epochs'], 'seed': CONFIG['seed'],
        'input': '224x224', 'classes': noms_classes,
    }, CONFIG['wandb_project'])

    hist_vgg = entrainer_modele(
        modele_vgg, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer_vgg,
        nb_epochs=CONFIG['vgg_epochs'], device=device,
        nom_modele='vgg16_scratch', scheduler=scheduler_vgg,
        wandb_run=run_vgg, dossier_save=CONFIG['checkpoints']
    )

    matrice_confusion(modele_vgg, val_loader, device, noms_classes, 'vgg16_scratch', run_vgg)
    tracer_courbes(hist_vgg, 'vgg16_scratch')
    run_vgg.finish()

    # ── Étape 4 : ResNet18 fine-tuning ────────────────────────────────────────
    print('\n' + '='*60)
    print('  PARTIE 3 — ResNet18 Fine-tuning')
    print('='*60)

    train_loader, val_loader, _ = creer_dataloaders(
        CONFIG['train_dir'], CONFIG['val_dir'],
        model_type='resnet', batch_size=CONFIG['resnet_batch'],
        num_workers=CONFIG['num_workers']
    )

    modele_res = ResNet18FineTune(nb_classes=len(noms_classes), geler_backbone=True).to(device)

    # On passe uniquement les paramètres entraînables à l'optimiseur
    # filter(lambda p: p.requires_grad, ...) exclut les paramètres gelés
    # Ça évite qu'Adam leur alloue de la mémoire inutilement
    params_entrainables = filter(lambda p: p.requires_grad, modele_res.parameters())
    optimizer_res = torch.optim.Adam(params_entrainables, lr=CONFIG['resnet_lr'])
    scheduler_res = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_res, mode='min', patience=5, factor=0.5, verbose=False
    )

    stats = modele_res.stats_parametres()
    print(f'  Paramètres entraînés : {stats["entrainables"]:,} / {stats["total"]:,}')

    run_res = lancer_run_wandb('resnet18_finetune', {
        'model': 'ResNet18', 'pretrained': True, 'backbone_gele': True,
        'params_entraines': stats['entrainables'],
        'lr': CONFIG['resnet_lr'], 'batch': CONFIG['resnet_batch'],
        'epochs': CONFIG['resnet_epochs'], 'seed': CONFIG['seed'],
        'input': '224x224', 'classes': noms_classes,
    }, CONFIG['wandb_project'])

    hist_res = entrainer_modele(
        modele_res, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer_res,
        nb_epochs=CONFIG['resnet_epochs'], device=device,
        nom_modele='resnet18_finetune', scheduler=scheduler_res,
        wandb_run=run_res, dossier_save=CONFIG['checkpoints']
    )

    matrice_confusion(modele_res, val_loader, device, noms_classes, 'resnet18_finetune', run_res)
    tracer_courbes(hist_res, 'resnet18_finetune')
    run_res.finish()

    # ── Étape 5 : comparaison finale ──────────────────────────────────────────
    print('\n' + '='*60)
    print('  COMPARAISON FINALE')
    print('='*60)

    # On garde le meilleur LeNet5 (celui avec la meilleure val_acc finale)
    meilleur_act = max(historiques_lenet, key=lambda k: max(historiques_lenet[k]['val_acc']))
    print(f'  Meilleur LeNet5 : LeNet5-{meilleur_act}')

    comparer_trois_modeles({
        f'LeNet5-{meilleur_act}': historiques_lenet[meilleur_act],
        'VGG16'                : hist_vgg,
        'ResNet18'             : hist_res,
    })

    tableau_parametres({
        'LeNet5-tanh'   : LeNet5('tanh'),
        'LeNet5-relu'   : LeNet5('relu'),
        'LeNet5-sigmoid': LeNet5('sigmoid'),
        'VGG16'         : VGG16(),
        'ResNet18 (tête)': ResNet18FineTune(geler_backbone=True),
    })

    print('\n  ✓ Entraînement terminé ! Résultats dans wandb et dans les fichiers .png')


if __name__ == '__main__':
    main()
