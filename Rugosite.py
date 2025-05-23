import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve

class Rugosite():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name
     # def rug_ecart_type(self, n):
    #     # à ne pas utiliser pour calculer la rugosité 3*3,5*5,7*7 ... car cette méthode demande bcp trop de temps de calcul
    #     print('je suis au debut de rug_ecart_type')
    #     rugosite = generic_filter(self.mnt, np.nanstd, size=n, mode='constant', cval=np.nan)
    #     print('je suis a la fin de rug_ecart_type')
    #     return rugosite

    # def rugosite_ecart_type_plot(self, n):
    #     rugosite = self.rug_ecart_type(n)
    #     print('je suis dans rug_ecart_type_plot')
    #     plt.figure()
    #     plt.imshow(rugosite, origin='lower', cmap='viridis')
    #     plt.title(f'Rugosité (écart-type) de {self.name}')
    #     plt.colorbar(label='Rugosité')
    #     plt.show()

    def rugosite_ecart_type_analytique(self, n):

        if n % 2 == 0:
            raise ValueError("La taille du noyau doit être impaire")
        noyau = np.ones((n, n)) / (n**2)
        
        mnt_carre = self.mnt ** 2
        
        voisins_mnt_carre = convolve(mnt_carre, noyau, mode='constant', cval=np.nan)
        
        voisins = convolve(self.mnt, noyau, mode='constant', cval=np.nan)
        
        rugosite = np.sqrt(np.clip(voisins_mnt_carre - voisins**2, 0, None)) #le np.clip permet d'éviter les racines carrés de nombres négatifs        
        return rugosite

    def subplot_rugosite(self, tailles):
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        fig.suptitle(f'Rugosité (écart-type) pour différents voisinages - {rugosite.name}', fontsize=16)        
        
        for ax, n in zip(axes.flat, tailles_voisinage):

            rug = self.rugosite_ecart_type_analytique(n)
            
            img = ax.imshow(rug, origin='lower', cmap='viridis')
            
            ax.set_title(f'Noyau {n}x{n}')
            
            ax.axis('off')
            
            fig.colorbar(img, ax=ax, shrink=0.6)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

fichier = 'Dune2_Dunkerque_Extrait1_50cm.xyz' 
mnt = np.loadtxt("MNT/" + fichier)
pas = 8
name = fichier[:-4]
rugosite = Rugosite(mnt, pas, name)

tailles_voisinage = [3, 5, 7, 9, 11, 13, 15,17, 19]
rugosite.subplot_rugosite(tailles_voisinage)