import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from BPI import *
from Pente import *
from Rugosite import *


def b_bpi(BPI, PENTE, RUGOSITE, mnt):
    # === Étape 1 : Calculs des grilles ===
    broad_bpi = BPI.bpi_cercle(15)
    fine_bpi = BPI.bpi_cercle(5)
    fx, fy = PENTE.Evans(mnt)
    pente = PENTE.pente(fx, fy)
    rugosite = RUGOSITE.rugosite_ecart_type_analytique(13)

    # === Étape 2 : Création des masques ===
    broad_depression = broad_bpi <= -0.075
    broad_crest = broad_bpi >= 0.075
    broad_flat = (broad_bpi > -0.075) & (broad_bpi < 0.075)

    fine_depression = fine_bpi <= -0.075
    fine_crest = fine_bpi >= 0.075

    slope = pente >= 20
    roughness = rugosite >= 1.75

    # === Étape 3 : Carte des classes ===
    carte_classes = np.full_like(broad_bpi, np.nan)

    carte_classes[broad_depression & ~fine_crest] = 1  # Narrow depression
    carte_classes[broad_depression & fine_crest] = 2   # Local crest in depression
    carte_classes[broad_crest & fine_depression] = 3   # Depression on crest
    carte_classes[broad_crest & ~fine_depression] = 4  # Narrow crest
    carte_classes[slope & broad_flat] = 5              # Slope
    carte_classes[~slope & broad_flat & roughness] = 6  # Flat with roughness
    carte_classes[~slope & broad_flat & ~roughness] = 7 # Flat without roughness

    # === Étape 4 : Affichage ===
    noms_classes = {
        1: "Narrow depression",
        2: "Local crest in depression",
        3: "Depression on crest",
        4: "Narrow crest",
        5: "Slope",
        6: "Flat (rough)",
        7: "Flat (smooth)"
    }

    # Couleurs personnalisées
    couleurs = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3']
    cmap = ListedColormap(couleurs[:len(noms_classes)])

    unique_classes = sorted(int(k) for k in np.unique(carte_classes[~np.isnan(carte_classes)]))
    ticks = list(range(1, len(unique_classes)+1))
    labels = [noms_classes[k] for k in unique_classes]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # === BPI large ===
    im0 = axs[0].imshow(broad_bpi, origin='lower', cmap='bwr', vmin=-0.1, vmax=0.1)
    axs[0].set_title('BPI large')
    plt.colorbar(im0, ax=axs[0], label='BPI')

    # === Carte des classes ===
    im1 = axs[1].imshow(carte_classes, origin='lower', cmap=cmap, vmin=0.5, vmax=7.5)
    axs[1].set_title('Classification bathymétrique')
    cbar = plt.colorbar(im1, ax=axs[1], ticks=ticks)
    cbar.ax.set_yticklabels(labels)

    # === MNT terrain ===
    im2 = axs[2].imshow(mnt, origin='lower', cmap='terrain')
    axs[2].set_title('Terrain (MNT)')
    plt.colorbar(im2, ax=axs[2], label='Élévation (m)')

    for ax in axs:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    fichier = "lezardrieux_z.txt"
    mnt = np.loadtxt("MNT/" + fichier)
    pas = 1
    name = fichier[:-4]

    bpi = BPI(mnt, pas, name)
    pente = Pente(mnt, pas, name)
    rugosite = Rugosite(mnt, pas, name)

    b_bpi(bpi,pente,rugosite,mnt)