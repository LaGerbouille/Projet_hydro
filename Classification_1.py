import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from BPI import*
from Pente import *
from Rugosite import *



def b_bpi (BPI):
    broad_bpi = BPI.bpi_cercle(23)
    masque_depression = broad_bpi <= -0.025
    masque_crete = broad_bpi >= 0.025

    carte_classes_bpi = np.zeros_likes(broad_bpi)

    carte_classes_bpi[masque_depression] = 
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1,3,1)
    # plt.imshow(broad_bpi, origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
    # plt.title('BPI large')
    # plt.colorbar(label='BPI')

    # plt.subplot(1,3,2)
    # plt.imshow(masque_depression, origin='lower', cmap='RdBu')
    # plt.title('Masque dépression')
    # plt.colorbar(label='Masque dépression')

    # plt.subplot(1,3,3)
    # plt.imshow(masque_crete, origin='lower', cmap='RdBu')
    # plt.title('Masque crête')
    # plt.colorbar(label='Masque crête')

    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    fichier = "lezardrieux_z.txt"
    mnt = np.loadtxt("MNT/" + fichier)
    pas = 1
    name = fichier[:-4]

    bpi = BPI(mnt, pas, name)
    pente = Pente(mnt, pas, name)
    rugosite = Rugosite(mnt, pas, name)

    b_bpi(bpi)






