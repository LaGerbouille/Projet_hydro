import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import generic_filter
from matplotlib.colors import CenteredNorm

class BPI():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name
    def carre(self,n):
        carre=np.ones(n)
        carre[n/2][n/2]=0
        carre = carre/(n**2-1)
        return carre
    
    def cercle(self,r):
        x = np.arange(-r,r+1,1)
        y = np.arange(-r,r+1,1)
        X,Y = np.meshgrid(x,y)
        D = (X**2 + Y**2)**0.5
        M = (D<=r)*1
        return M

    def bpi_carre(self):
        noyau = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]) / 8
        voisins = convolve(self.mnt,noyau,mode='constant',cval=np.nan)
        bpi = self.mnt - voisins
        return bpi

    def bpi_cercle(self):
        noyau = np.array([
            [0,1,1,1,0],
            [1,1,1,1,1],
            [1,1,0,1,1],
            [1,1,1,1,1],
            [0,1,1,1,0],
        ]) / 20
        voisins = convolve(self.mnt,noyau,mode='constant',cval=np.nan)
        bpi = self.mnt - voisins
        return bpi

    def affichage_bpi_cercle(self):
        plt.figure()
        plt.imshow(self.bpi_cercle(), origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
        plt.title(f'BPI cercle de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()
        print('min:',np.nanmin(self.bpi_cercle()),'max:',np.nanmax(self.bpi_cercle()))

    def affichage_bpi_carre(self):

        plt.figure()
        plt.imshow(self.bpi_carre(), origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
        plt.title(f'BPI de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()

fichier = 'Dune2_Dunkerque_Extrait1_50cm.xyz' 

mnt = np.loadtxt("MNT/" + fichier)
pas = 8
pas_bpi=1
name = fichier[:-4]

bpi = BPI(mnt, pas_bpi, name)

