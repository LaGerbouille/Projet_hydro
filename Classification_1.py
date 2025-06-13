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
    # BPI large
    broad_bpi = BPI.bpi_cercle(23)
    broad_depression = broad_bpi <= -0.025
    broad_crest = broad_bpi >= 0.025

    #BPI fin
    fine_bpi = BPI.bpi_cercle(5)
    fine_depression = fine_bpi <= -0.25
    fine_crest = fine_bpi >= 0.25


    carte_classes_bpi = np.full_like(broad_bpi,np.nan)

    #classification des classes depression
    #narrow depression 
    carte_classes_bpi[broad_depression & fine_depression] = 1
    #narrow crest
    carte_classes_bpi[broad_crest & fine_crest] = 2
    #local crest in depression
    carte_classes_bpi[broad_depression & fine_crest] = 3
    #broad depression with an open bottom

    #classification des classes crest
    #Depression on crest
    carte_classes_bpi[broad_crest & fine_depression] = 4
    #narrow crest
    carte_classes_bpi[broad_crest & fine_crest] = 5




    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.imshow(broad_bpi, origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
    plt.title('BPI large')
    plt.colorbar(label='BPI')

    plt.subplot(1,2,2)
    cm = plt.get_cmap('tab20', 5)
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






