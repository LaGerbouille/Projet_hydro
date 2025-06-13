import numpy as np
import matplotlib.pyplot as plt
from BPI import *
from Pente import *
from Rugosite import *
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmcrameri 

def b_bpi(BPI, PENTE, RUGOSITE, mnt):
    # === Étapes de calcul ===
    broad_bpi = BPI.bpi_cercle(15)
    fine_bpi = BPI.bpi_cercle(5)
    fx, fy = PENTE.Evans(mnt)
    pente = PENTE.pente(fx, fy)
    rugosite = RUGOSITE.rugosite_ecart_type_analytique(13)

    # === Masques ===
    broad_depression = broad_bpi <= -0.075
    broad_crest = broad_bpi >= 0.075
    broad_flat = (broad_bpi > -0.075) & (broad_bpi < 0.075)

    fine_depression = fine_bpi <= -0.075
    fine_crest = fine_bpi >= 0.075

    slope = pente >= 10
    roughness = rugosite >= 0.1

    # === Carte de classes ===
    carte_classes = np.full_like(broad_bpi, np.nan)
    carte_classes[broad_depression & (~fine_crest)] = 1
    carte_classes[broad_depression & fine_crest] = 2
    carte_classes[broad_crest & fine_depression] = 3
    carte_classes[broad_crest & ~fine_depression] = 4
    carte_classes[slope & broad_flat] = 5
    carte_classes[~slope & broad_flat & roughness] = 6
    carte_classes[~slope & broad_flat & ~roughness] = 7

    noms_classes = [
        "Narrow depression",
        "Local crest in depression",
        "Depression on crest",
        "Narrow crest",
        "Slope",
        "Flat rough",
        "Flat smooth"
    ]

    vmin = int(np.nanmin(carte_classes))
    vmax = int(np.nanmax(carte_classes))
    N = vmax - vmin + 1


    # === Première figure : MNT brut + ombrage ===
    fig1, ax = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.cm.gist_earth
    resol = 1

    # MNT brut
    im0 = ax[0].imshow(mnt, origin='lower', cmap=cmap)
    ax[0].set_title('MNT - Affichage simple')

    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, label='Altitude [m]', cax=cax0)

    # Ombrage
    ls = LightSource(azdeg=-45, altdeg=35)
    mnt_mask = np.ma.masked_invalid(mnt)
    rgb = ls.shade(mnt_mask, cmap=cmap, vert_exag=4, dx=resol, dy=resol, blend_mode='soft')
    im1 = ax[1].imshow(rgb, origin='lower')
    ax[1].set_title('MNT - Estompage (ombrage)')

    # Ajouter colorbar 
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, label='Altitude [m]', cax=cax1)
    plt.tight_layout()
    plt.show()

    # === Deuxième figure : Classification + ombrage ===
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))

    cmap_classif = plt.get_cmap('tab10', N)
    im2 = ax2[0].imshow(carte_classes, origin='lower', cmap=cmap_classif, vmin=vmin - 0.5, vmax=vmax + 0.5)
    ax2[0].set_title('Classification BPI / pente / rugosité')

    divider2 = make_axes_locatable(ax2[0])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(im2, ticks=range(vmin, vmax + 1), cax=cax2)
    cb1.ax.set_yticklabels(noms_classes)

    im3 = ax2[1].imshow(rgb, origin='lower')
    ax2[1].set_title('MNT - Estompage')

    divider3 = make_axes_locatable(ax2[1])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, label='Altitude [m]', cax=cax3)

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