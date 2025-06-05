import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from BPI import*
from Pente import *
from Rugosite import *

# depression = 1 = bleu
# crete = 2 = rouge
#
def b_bpi (BPI):
    broad_bpi = BPI.bpi_cercle(23)
    masque_depression = broad_bpi <= -0.025
    masque_crete = broad_bpi >= 0.025

    carte_classes_bpi = np.full_like(broad_bpi,np.nan)

    carte_classes_bpi[masque_depression] = 1
    carte_classes_bpi[masque_crete] = 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.imshow(broad_bpi, origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
    plt.title('BPI large')
    plt.colorbar(label='BPI')

    plt.subplot(1,2,2)
    cm = plt.get_cmap('Accent', 2)
    plt.imshow(carte_classes_bpi, origin='lower', cmap=cm)
    plt.title('Masque dépression')
    plt.colorbar(label='Masque dépression')

    # plt.subplot(1,3,3)
    # plt.imshow(masque_crete, origin='lower', cmap='RdBu')
    # plt.title('Masque crête')
    # plt.colorbar(label='Masque crête')

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

    b_bpi(bpi)






