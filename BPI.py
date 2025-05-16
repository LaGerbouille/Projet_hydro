import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class BPI():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    def bpi_carre(self):
        noyau = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]) / 9
        voisins = convolve2d(self.mnt,noyau,mode='same',boundary = 'fill', fillvalue=0)
        bpi = self.mnt - voisins
        return bpi


    def bpi_cercle(self):
        noyau = np.array([
            [0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,0,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0],
        ]) / 25
        voisins = convolve2d(self.mnt,noyau,mode='same',boundary= 'fill', fillvalue=0)
        bpi = self.mnt - voisins
        return bpi

    def affichage_bpi_cercle(self):
        plt.figure()
        plt.imshow(self.bpi_cercle(), origin='lower', cmap='magma_r')
        plt.title(f'BPI cercle de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()

    def affichage_bpi_carre(self):

        plt.figure()
        plt.imshow(self.bpi_carre(), origin='lower', cmap='magma_r')
        plt.title(f'BPI de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()


