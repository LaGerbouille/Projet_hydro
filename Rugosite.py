import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve
from Pente import Pente

class Rugosite():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    def rug_ecart_type(self, n):
        # à ne pas utiliser pour calculer la rugosité 3*3,5*5,7*7 ... car cette méthode demande bcp trop de temps de calcul
        print('je suis au debut de rug_ecart_type')
        rugosite = generic_filter(self.mnt, np.nanstd, size=n, mode='constant', cval=np.nan)
        print('je suis a la fin de rug_ecart_type')
        return rugosite

    def rugosite_ecart_type_plot(self, n):
        rugosite = self.rug_ecart_type(n)
        print('je suis dans rug_ecart_type_plot')
        plt.figure()
        plt.imshow(rugosite, origin='lower', cmap='viridis')
        plt.title(f'Rugosité (écart-type) de {self.name}')
        plt.colorbar(label='Rugosité')
        plt.show()

    def rugosite_ecart_type_analytique(self, n):

        if n % 2 == 0:
            raise ValueError("La taille du noyau doit être impaire")
        noyau = np.ones((n, n)) / (n**2)
        
        mnt_carre = self.mnt ** 2
        
        voisins_mnt_carre = convolve(mnt_carre, noyau, mode='constant', cval=np.nan)
        
        voisins = convolve(self.mnt, noyau, mode='constant', cval=np.nan)
        
        rugosite = np.sqrt(np.clip(voisins_mnt_carre - voisins**2, 0, None)) #le np.clip permet d'éviter les racines carrés de nombres négatifs        
        return rugosite

    def subplot_rugosite_ecart_type(self, tailles_voisinage):
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        fig.suptitle(f'Rugosité (écart-type) pour différents voisinages - {self.name}', fontsize=16)        
        
        for ax, n in zip(axes.flat, tailles_voisinage):

            rug = self.rugosite_ecart_type_analytique(n)
            
            img = ax.imshow(rug, origin='upper', cmap='viridis')
            
            ax.set_title(f'Noyau {n}x{n}')
            
            ax.axis('off')
            
            fig.colorbar(img, ax=ax, shrink=0.6)
            
        plt.tight_layout()

        plt.subplots_adjust(top=0.90)

        plt.show()

    def vecteurs_normaux(self, Pente):

        fx, fy = Pente.Evans(Pente.mnt)

        pente = Pente.pente(fx,fy)

        exposition = Pente.exposition(fx, fy)

        x_i = np.sin(np.radians(pente)) * np.cos(np.radians(exposition))

        y_i = np.sin(np.radians(pente)) * np.sin(np.radians(exposition))

        z_i = np.cos(np.radians(pente))

        return x_i, y_i, z_i

    
    def sommes_vecteurs_normaux(self,n):

        x_i, y_i, z_i = self.vecteurs_normaux(Pente(self.mnt, self.pas, self.name))

        noyau = np.ones((n, n)) / (n**2)
    
        x_somme = convolve(x_i, noyau, mode='constant', cval=np.nan)

        y_somme = convolve(y_i, noyau, mode='constant', cval=np.nan)

        z_somme = convolve(z_i, noyau, mode='constant', cval=np.nan)
        
        return x_somme, y_somme, z_somme

    def rugosite_vecteurs_normaux(self,n):

        x_somme, y_somme, z_somme = self.sommes_vecteurs_normaux(n)

        norme = np.sqrt(x_somme**2 + y_somme**2 + z_somme**2)

        rugosite = 1 - (norme / (n**2))

        return rugosite
    
    def affichage_rugosite_vesteurs_normaux(self, n):
        
        rugosite = self.rugosite_vecteurs_normaux(n)
        plt.figure()
        plt.imshow(rugosite, origin='lower', cmap='viridis')
        plt.title(f'Rugosité (vecteurs normaux) de {self.name} avec noyau {n}x{n}')
        plt.colorbar(label='Rugosité')
        plt.show()