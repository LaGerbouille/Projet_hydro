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
        if n % 2 == 0:
            raise ValueError("La taille du noyau doit être impair")
        noyau=np.ones((n,n))
        centre = n //2
        noyau[centre,centre] = 0
        noyau = noyau/(n**2-1)
        return noyau
    
    def cercle(self,r):
        x = np.arange(-r, r + 1)
        y = np.arange(-r, r + 1)
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        
        masque = (D <= r).astype(int)
        centre = r 
        masque[centre, centre] = 0  
        masque = masque / np.sum(masque) 
        return masque

    def bpi_carre(self,n):
        noyau = self.carre(n)
        voisins = convolve(self.mnt,noyau,mode='constant',cval=np.nan)
        bpi = self.mnt - voisins
        return bpi

    def bpi_cercle(self,r):
        noyau = self.cercle(r)
        voisins = convolve(self.mnt,noyau,mode='constant',cval=np.nan)
        bpi = self.mnt - voisins
        return bpi

    def affichage_bpi_cercle(self,r):
        plt.figure()
        plt.imshow(self.bpi_cercle(r), origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
        plt.title(f'BPI cercle de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()
        
    def affichage_bpi_carre(self,n):

        plt.figure()
        plt.imshow(self.bpi_carre(n), origin='lower', cmap='bwr',norm=CenteredNorm(0,halfrange=0.1))
        plt.title(f'BPI de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()

fichier = 'Dune2_Dunkerque_Extrait1_50cm.xyz' 

mnt = np.loadtxt("MNT/" + fichier)
pas = 8
pas_bpi=1
name = fichier[:-4]
bpi = BPI(mnt, pas_bpi, name)
bpi.affichage_bpi_carre(3)
bpi.affichage_bpi_cercle(3)

