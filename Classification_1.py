import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from BPI import*
from Pente import *
from Rugosite import *


# depression = 1 = bleu
# crete = 2 = rouge
#
def b_bpi (BPI,PENTE):
    # BPI large
    broad_bpi = BPI.bpi_cercle(15)
    #plot du BPI large
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(broad_bpi, origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
    plt.title('BPI large')
    plt.colorbar(label='BPI')
    
    #filtre pour le BPI
    broad_depression = broad_bpi <= -0.075
    broad_crest = broad_bpi >= 0.075
    broad_flat = (broad_bpi > -0.075) & (broad_bpi < 0.075)
    #BPI fin
    fine_bpi = BPI.bpi_cercle(5)
    fine_depression = fine_bpi <= -0.075
    fine_crest = fine_bpi >= 0.075

    #filtre pour les pentes
    fx,fy = PENTE.Evans(mnt)
    pente = PENTE.pente(fx,fy)
    plt.figure
    plt.subplot(1,4,3)
    plt.imshow(pente, origin='lower', cmap='bwr')
    plt.title('Pente')
    plt.colorbar(label='Pente (degrés)')

    



    carte_classes_bpi = np.full_like(broad_bpi,np.nan)

    # carte_classes_bpi = BPI.bpi_cercle(23)

    carte_classes_bpi [broad_flat] = 0  # flat areas
    #classification des classes depression
    
    #narrow depression 
    carte_classes_bpi[(broad_depression) & (~fine_crest)] = 1

    #local crest in depression
    carte_classes_bpi[(broad_depression) & (fine_crest)] = 2

    #broad depression with an open bottom

    #classification des classes crest
    #Depression on crest
    carte_classes_bpi[(broad_crest) & (fine_depression)] = 3
    #narrow crest 
    carte_classes_bpi[(broad_crest) & (~fine_depression)] = 4

    #j'ai donc fait les premières branches  de classification
    #maitenant il faut faire d'autres classes pour affiner la classification

    noms_bpi = ['flat surface','narrow depression', 'local crest in depression','Derpression on crest', 'narrow crest']
    vmin = int(np.nanmin(carte_classes_bpi))
    vmax = int(np.nanmax(carte_classes_bpi))
    N = vmax - vmin + 1
    print(N)
    plt.subplot(1, 4, 2)

    cm = plt.get_cmap('Accent', N)
    plt.imshow(carte_classes_bpi, origin='lower', cmap=cm, vmin=vmin-.5,vmax=vmax+.5)
    cbar = plt.colorbar(ticks=range(vmin,vmax+1))
    cbar.ax.set_yticklabels(noms_bpi)
    plt.title('Segmentation')
    plt.tight_layout()

    #plot du terrain réel 
    plt.subplot(1,4,4)
    plt.imshow(mnt, origin='lower', cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Terrain Elevation')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

if __name__ == '__main__':
    fichier = "lezardrieux_z.txt"
    mnt = np.loadtxt("MNT/" + fichier)
    pas = 1
    name = fichier[:-4]

    bpi = BPI(mnt, pas, name)
    pente = Pente(mnt, pas, name)
    rugosite = Rugosite(mnt, pas, name)

    b_bpi(bpi,pente)